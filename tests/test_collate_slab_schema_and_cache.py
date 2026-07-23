"""Regression tests for two slab-streaming collate bugs.

BUG A (schema lock on a null-typed first slab): a sparse metadata column that
is all-NaN across the whole first slab is inferred by pyarrow as type ``null``.
That dtype gets locked into the ParquetWriter schema, so the first later slab
that carries strings in that column raises ``ArrowNotImplementedError``
('Unsupported cast from string to null'); the ``finally`` then closes the
writer and leaves a *valid* parquet that silently contains only slab 0.

BUG B (poisoned sidecar dir cache): the per-run ``dir_names_cache`` caches the
empty set produced by a transient ``os.scandir`` ``OSError``. Every later file
in that directory then reads all its sidecars as '' -> silent data loss. A
transient error must cost at most the one file that hit it, and the cache must
not be poisoned.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import src.collate as collate

MODEL_NAMES = ["giga_ctc", "giga_rnnt", "vosk"]


def _write_sidecars(audio_dir: Path, stem: str, mapping: dict) -> None:
    """Translate old per-suffix sidecar contents into one ``<stem>.json``."""
    data: dict = {}
    for suffix, content in mapping.items():
        name = suffix[1:]
        if name.endswith(".tst"):
            data.setdefault("asr_ts", {})[name[:-4]] = content
        elif name.endswith(".txt"):
            base = name[:-4]
            if base in ("rover", "punct", "accent", "rover_phonemes"):
                data[base] = content
            else:
                data.setdefault("asr", {})[base] = content
    (audio_dir / f"{stem}.json").write_text(json.dumps(data), encoding="utf-8")


# --------------------------------------------------------------------------- #
# BUG A                                                                        #
# --------------------------------------------------------------------------- #
def test_sparse_metadata_column_survives_slab_boundary(tmp_path):
    """A metadata column that is all-NaN in slab 0 but holds strings in slab 1
    must round-trip: every row present, column dtype string, NaN preserved."""
    base = tmp_path / "data"
    d = base / "1"
    d.mkdir(parents=True)

    rows = []
    for i in range(4):
        p = d / f"f{i}.wav"
        p.write_bytes(b"x")
        _write_sidecars(d, f"f{i}", {"_giga_ctc.txt": f"text {i}"})
        # 'note' is a metadata column empty for the whole first slab (rows 0-1)
        # and populated only from row 2 on (the start of slab 1).
        rows.append(
            {
                "filepath": str(p),
                "speaker_id": i,
                "note": np.nan if i < 2 else f"note-{i}",
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(base / "balalaika.parquet", index=False)

    cfg = {
        "download": {
            "podcasts_path": str(base),
            "num_workers": 2,
            # slab_rows=2 -> slab 0 = rows 0-1 (note all-NaN), slab 1 = rows 2-3.
            "collate_slab_rows": 2,
        },
        "transcription": {"model_names": MODEL_NAMES},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    class Args:
        config_path = str(cfg_path)
        log_dir = str(tmp_path / "logs")

    collate.main(Args())

    got = pd.read_parquet(base / "balalaika.parquet")
    # All four rows present (the bug truncated the parquet to slab 0 only).
    assert len(got) == 4
    assert sorted(got["speaker_id"].tolist()) == [0, 1, 2, 3]
    got = got.set_index("speaker_id").sort_index()
    # The sparse column is string-typed with NaN/None preserved where empty.
    assert pd.isna(got.loc[0, "note"])
    assert pd.isna(got.loc[1, "note"])
    assert got.loc[2, "note"] == "note-2"
    assert got.loc[3, "note"] == "note-3"


def test_residual_schema_mismatch_raises_clear_error(tmp_path, monkeypatch):
    """A genuine (non-null) dtype disagreement between slab 0 and a later slab
    must surface, through ``collate.main``, a clear error naming the offending
    column -- not a bare ArrowInvalid -- so the writer's ``finally`` cannot
    leave a silently truncated parquet without explanation."""
    base = tmp_path / "data"
    d = base / "1"
    d.mkdir(parents=True)

    rows = []
    for i in range(4):
        p = d / f"f{i}.wav"
        p.write_bytes(b"x")
        rows.append({"filepath": str(p), "speaker_id": i})
    df = pd.DataFrame(rows)
    df.to_parquet(base / "balalaika.parquet", index=False)

    cfg = {
        "download": {
            "podcasts_path": str(base),
            "num_workers": 2,
            "collate_slab_rows": 2,
        },
        "transcription": {"model_names": MODEL_NAMES},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    class Args:
        config_path = str(cfg_path)
        log_dir = str(tmp_path / "logs")

    # Make the SECOND slab carry an incompatible (string) value in the
    # int-locked 'speaker_id' column so the per-slab cast genuinely fails.
    real_build = collate.build_slab_frame
    state = {"n": 0}

    def poison_second_slab(meta_slab, *a, **k):
        slab, errs = real_build(meta_slab, *a, **k)
        state["n"] += 1
        if state["n"] == 2:
            slab = slab.copy()
            slab["speaker_id"] = ["not-an-int"] * len(slab)
        return slab, errs

    monkeypatch.setattr(collate, "build_slab_frame", poison_second_slab)

    with pytest.raises(RuntimeError) as excinfo:
        collate.main(Args())
    assert "speaker_id" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# BUG B                                                                        #
# --------------------------------------------------------------------------- #
def test_transient_scandir_error_does_not_poison_cache(tmp_path, monkeypatch):
    """A one-shot ``os.scandir`` ``OSError`` for a directory must fall back to a
    direct per-file read for *that* call, must NOT cache an empty listing, and
    the next call for the same directory must use the real (good) listing."""
    base = tmp_path / "data"
    d = base / "1"
    d.mkdir(parents=True)

    file_types = collate.sidecar_specs(MODEL_NAMES)

    p0 = d / "f0.wav"
    p1 = d / "f1.wav"
    p0.write_bytes(b"x")
    p1.write_bytes(b"x")
    _write_sidecars(d, "f0", {"_giga_ctc.txt": "hello zero", "_vosk.txt": "hi zero"})
    _write_sidecars(d, "f1", {"_giga_ctc.txt": "hello one", "_vosk.txt": "hi one"})

    real_scandir = os.scandir
    calls = {"n": 0}

    def flaky_scandir(path, *a, **k):
        # Fail only the very first scandir of the run; succeed thereafter.
        if calls["n"] == 0:
            calls["n"] += 1
            raise OSError("EMFILE: too many open files")
        return real_scandir(path, *a, **k)

    # process_audio_file does a local ``import os`` that resolves to the real
    # ``os`` module, so patch ``os.scandir`` there.
    monkeypatch.setattr(os, "scandir", flaky_scandir)

    cache: dict = {}

    # First file: scandir raises -> must fall back to a direct read, NOT cache ''.
    res0 = collate.process_audio_file(str(p0), base, file_types, cache)
    assert res0["giga_ctc"] == "hello zero"
    assert res0["vosk"] == "hi zero"
    # The failed listing must not have been cached.
    assert all(v != set() for v in cache.values()) or not cache

    # Second file: scandir now succeeds; result must still be correct and the
    # cache must hold the real (non-empty) listing.
    res1 = collate.process_audio_file(str(p1), base, file_types, cache)
    assert res1["giga_ctc"] == "hello one"
    assert res1["vosk"] == "hi one"
    assert calls["n"] == 1  # scandir was retried exactly once after the failure
    # A real listing got cached for the directory.
    assert any(names for names in cache.values())
