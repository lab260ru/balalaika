"""Tests for src/to_webdataset.worker_fn.

Focus: the redundant exists() stat before read_bytes() was removed; a missing
file must still be skipped silently (not counted as an error), while a present
file is packed normally and any non-missing read error is logged + counted.
"""

import json
import tarfile

import pytest

from src.to_webdataset import has_metadata_sidecars, worker_fn


def _read_shard_keys(output_dir):
    """Return {key: {ext: bytes}} for all samples across produced shards."""
    samples = {}
    for tar_path in sorted(output_dir.glob("*.tar")):
        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                key, _, ext = member.name.partition(".")
                data = tar.extractfile(member).read()
                samples.setdefault(key, {})[ext] = data
    return samples


def test_present_file_is_packed(tmp_path):
    audio = tmp_path / "clip1.wav"
    audio.write_bytes(b"RIFFfake-audio-bytes")
    out = tmp_path / "out"
    out.mkdir()

    processed, errors = worker_fn(
        worker_id=0,
        audio_paths=[str(audio)],
        output_dir=out,
        metadata_dict={},
        max_shard_size=10 * 1024 * 1024,
        max_shard_count=1000,
    )

    assert processed == 1
    assert errors == 0
    samples = _read_shard_keys(out)
    assert "clip1" in samples
    assert samples["clip1"]["wav"] == b"RIFFfake-audio-bytes"


def test_missing_file_skipped_silently(tmp_path):
    """A path that does not exist must be skipped with NO error count and NO
    sample written (same semantics as the old exists() guard)."""
    missing = tmp_path / "ghost.wav"  # never created
    present = tmp_path / "real.wav"
    present.write_bytes(b"RIFFreal")
    out = tmp_path / "out"
    out.mkdir()

    processed, errors = worker_fn(
        worker_id=0,
        audio_paths=[str(missing), str(present)],
        output_dir=out,
        metadata_dict={},
        max_shard_size=10 * 1024 * 1024,
        max_shard_count=1000,
    )

    assert processed == 1  # only the real file
    assert errors == 0  # missing file is NOT an error
    samples = _read_shard_keys(out)
    assert "real" in samples
    assert "ghost" not in samples


def test_all_missing_produces_nothing(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    processed, errors = worker_fn(
        worker_id=0,
        audio_paths=[str(tmp_path / "a.wav"), str(tmp_path / "b.wav")],
        output_dir=out,
        metadata_dict={},
        max_shard_size=10 * 1024 * 1024,
        max_shard_count=1000,
    )
    assert processed == 0
    assert errors == 0


def test_unreadable_file_counts_as_error(tmp_path):
    """A real read error (not a missing file) must still be logged + counted."""
    import sys

    if sys.platform.startswith("win"):
        pytest.skip("permission semantics differ on Windows")

    audio = tmp_path / "blocked.wav"
    audio.write_bytes(b"data")
    audio.chmod(0o000)
    out = tmp_path / "out"
    out.mkdir()

    try:
        processed, errors = worker_fn(
            worker_id=0,
            audio_paths=[str(audio)],
            output_dir=out,
            metadata_dict={},
            max_shard_size=10 * 1024 * 1024,
            max_shard_count=1000,
        )
    finally:
        audio.chmod(0o644)

    # Running as root can bypass permission bits; only assert when it actually
    # produced a read error.
    if processed == 0:
        assert errors == 1
    else:
        pytest.skip("permission bits not enforced (likely running as root)")


def test_metadata_and_sibling_packed(tmp_path):
    audio = tmp_path / "clip.flac"
    audio.write_bytes(b"FLACdata")
    sibling = tmp_path / "clip_transcript"
    sibling.write_text("hello world", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()

    processed, errors = worker_fn(
        worker_id=0,
        audio_paths=[str(audio)],
        output_dir=out,
        metadata_dict={"clip": {"speaker": "s1", "dur": 1.5}},
        max_shard_size=10 * 1024 * 1024,
        max_shard_count=1000,
    )

    assert processed == 1
    assert errors == 0
    samples = _read_shard_keys(out)
    meta = json.loads(samples["clip"]["json"].decode("utf-8"))
    assert meta["speaker"] == "s1"
    assert meta["dur"] == 1.5
    assert meta["transcript"] == "hello world"


def test_has_metadata_sidecars_ignores_text_only_json(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFFdata")
    (tmp_path / "clip.json").write_text(
        json.dumps({"asr": {"m": "hello"}, "rover": "hello"}),
        encoding="utf-8",
    )
    assert not has_metadata_sidecars([str(audio)])

    (tmp_path / "clip.json").write_text(
        json.dumps({"asr": {"m": "hello"}, "total_duration": 1.2}),
        encoding="utf-8",
    )
    assert has_metadata_sidecars([str(audio)])


def test_chunk_json_metadata_is_packed_and_overrides_legacy_dict(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFFdata")
    (tmp_path / "clip.json").write_text(
        json.dumps({"speaker": "json", "p_tts": 0.91}, ensure_ascii=False),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()

    processed, errors = worker_fn(
        worker_id=0,
        audio_paths=[str(audio)],
        output_dir=out,
        metadata_dict={"clip": {"speaker": "legacy", "DistillMOS": 4.0}},
        max_shard_size=10 * 1024 * 1024,
        max_shard_count=1000,
    )

    assert (processed, errors) == (1, 0)
    samples = _read_shard_keys(out)
    meta = json.loads(samples["clip"]["json"].decode("utf-8"))
    assert meta["speaker"] == "json"
    assert meta["p_tts"] == 0.91
    assert meta["DistillMOS"] == 4.0
    assert "json" not in meta


def test_shard_start_index_preserves_worker_name_format(tmp_path):
    audio = []
    for index in range(2):
        path = tmp_path / f"clip{index}.wav"
        path.write_bytes(f"audio-{index}".encode())
        audio.append(str(path))

    out = tmp_path / "out"
    out.mkdir()
    processed, errors = worker_fn(
        worker_id=1,
        audio_paths=audio,
        output_dir=out,
        metadata_dict={},
        max_shard_size=10 * 1024 * 1024,
        max_shard_count=1,
        shard_start_index=50,
    )

    assert (processed, errors) == (2, 0)
    assert sorted(path.name for path in out.glob("*.tar")) == [
        "shard_001_0050.tar",
        "shard_001_0051.tar",
    ]


def test_existing_shard_is_not_overwritten(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"new audio")
    out = tmp_path / "out"
    out.mkdir()
    existing = out / "shard_000_0042.tar"
    existing.write_bytes(b"already uploaded")

    with pytest.raises(FileExistsError):
        worker_fn(
            worker_id=0,
            audio_paths=[str(audio)],
            output_dir=out,
            metadata_dict={},
            max_shard_size=10 * 1024 * 1024,
            max_shard_count=1000,
            shard_start_index=42,
        )

    assert existing.read_bytes() == b"already uploaded"
