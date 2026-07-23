"""Round-trip test for to_webdataset worker_fn (sibling discovery + metadata)."""
from __future__ import annotations

import json
import tarfile

import numpy as np
import pandas as pd

from src.to_webdataset import load_metadata, resolve_output_dir, worker_fn


def test_resolve_output_dir_supports_separate_disk(tmp_path):
    podcasts_path = tmp_path / "ssd" / "dataset"
    custom = tmp_path / "hdd" / "exports" / "train"

    assert resolve_output_dir(podcasts_path, str(custom)) == custom
    assert resolve_output_dir(podcasts_path, "") == (
        podcasts_path.parent / "dataset_webdataset" / "train"
    )


def test_worker_fn_roundtrip(tmp_path):
    audio_dir = tmp_path / "pl" / "pod"
    audio_dir.mkdir(parents=True)

    a = audio_dir / "chunk_001.wav"
    a.write_bytes(b"RIFFfakewav")
    (audio_dir / "chunk_001_rover.txt").write_text("привет мир", encoding="utf-8")
    (audio_dir / "chunk_001_punct.txt").write_text("Привет, мир.", encoding="utf-8")
    # decoy: belongs to a DIFFERENT chunk whose stem shares the prefix
    (audio_dir / "chunk_0010_rover.txt").write_text("другое", encoding="utf-8")
    b = audio_dir / "chunk_0010.wav"
    b.write_bytes(b"RIFFotherwav")

    csv = tmp_path / "balalaika.csv"
    pd.DataFrame(
        {
            "filepath": [str(a), str(b)],
            "crest_factor": [2.5, np.nan],
            "DistillMOS": [4.25, 3.5],
        }
    ).to_csv(csv, index=False)
    # load_metadata now takes the dataset root (state-aware) rather than the CSV
    # path; in csv mode it reads <root>/balalaika.csv exactly as before.
    metadata = load_metadata(tmp_path)
    assert set(metadata) == {"chunk_001", "chunk_0010"}

    out = tmp_path / "wds"
    out.mkdir()
    processed, errors = worker_fn(
        0, [str(a), str(b)], out, metadata, 10**9, 1000
    )
    assert (processed, errors) == (2, 0)

    shards = sorted(out.glob("*.tar"))
    assert shards
    samples = {}
    for shard in shards:
        with tarfile.open(shard) as tf:
            for member in tf.getmembers():
                key, _, ext = member.name.partition(".")
                samples.setdefault(key, {})[ext] = tf.extractfile(member).read()

    s1 = json.loads(samples["chunk_001"]["json"])
    assert s1["rover.txt"] == "привет мир"
    assert s1["punct.txt"] == "Привет, мир."
    assert s1["crest_factor"] == 2.5
    assert s1["DistillMOS"] == 4.25
    # decoy sidecar must NOT leak into chunk_001... but chunk_0010's own does
    assert "0_rover.txt" not in s1
    assert samples["chunk_001"]["wav"] == b"RIFFfakewav"

    s2 = json.loads(samples["chunk_0010"]["json"])
    assert s2["rover.txt"] == "другое"
    assert s2["crest_factor"] is None  # NaN -> null
