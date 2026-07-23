"""Unit tests for src.utils.chunk_json — the one-JSON-per-chunk sidecar layer.

Pins the contract the four text stages + ROVER and collate rely on:
* atomic read-modify-write with nested deep-merge (so per-model ASR keys
  accumulate without clobbering each other);
* resume semantics via field presence (with the retry-empty rule);
* corrupt / missing JSON tolerated as "not done";
* ChunkJsonCache mirrors the per-file helpers (scandir + memoised parse).

Run: .dev_venv/bin/python -m pytest tests/test_chunk_json.py -q
"""
from __future__ import annotations

import os
from pathlib import Path

from src.utils import chunk_json as cj


def _audio(tmp_path: Path, stem: str = "chunk") -> Path:
    p = tmp_path / f"{stem}.flac"
    p.write_bytes(b"x")
    return p


def test_json_path_strips_one_extension(tmp_path):
    assert cj.chunk_json_path(tmp_path / "a.flac").name == "a.json"
    assert cj.chunk_json_path(tmp_path / "a.b.wav").name == "a.b.json"


def test_read_missing_returns_empty(tmp_path):
    assert cj.read_chunk_json(tmp_path / "nope.json") == {}


def test_read_corrupt_returns_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert cj.read_chunk_json(p) == {}


def test_update_deep_merges_nested(tmp_path):
    a = _audio(tmp_path)
    cj.update_chunk_json(a, {"asr": {"giga_ctc": "привет"}})
    cj.update_chunk_json(a, {"asr": {"vosk": "привет2"}, "rover": "привет"})
    cj.update_chunk_json(a, {"asr_ts": {"giga_ctc": "0.0"}})
    data = cj.read_chunk_json(cj.chunk_json_path(a))
    assert data == {
        "asr": {"giga_ctc": "привет", "vosk": "привет2"},
        "rover": "привет",
        "asr_ts": {"giga_ctc": "0.0"},
    }


def test_update_is_atomic_no_tmp_left(tmp_path):
    a = _audio(tmp_path)
    cj.update_chunk_json(a, {"rover": "x"})
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_get_field_dotted(tmp_path):
    data = {"asr": {"giga_ctc": "t"}, "rover": "r"}
    assert cj.get_field(data, "asr.giga_ctc") == "t"
    assert cj.get_field(data, "rover") == "r"
    assert cj.get_field(data, "asr.missing") is None
    assert cj.get_field(data, "nope.deep") is None


def test_field_complete_presence_and_retry_empty(tmp_path):
    a = _audio(tmp_path)
    cj.update_chunk_json(a, {"rover": "text", "punct": ""})
    assert cj.field_complete(a, "rover")
    assert cj.field_complete(a, "rover", retry_empty=True)
    # empty string: complete unless retry_empty, then it's "redo me"
    assert cj.field_complete(a, "punct")
    assert not cj.field_complete(a, "punct", retry_empty=True)
    # absent field
    assert not cj.field_complete(a, "accent")


class TestChunkJsonCache:
    def test_field_complete_matches_direct(self, tmp_path):
        a = _audio(tmp_path, "a")
        b = _audio(tmp_path, "b")  # no JSON written
        cj.update_chunk_json(a, {"asr": {"giga_ctc": "t"}, "punct": ""})
        cache = cj.ChunkJsonCache()
        assert cache.field_complete(a, "asr.giga_ctc")
        assert not cache.field_complete(a, "punct", retry_empty=True)
        assert cache.field_complete(a, "punct")
        assert not cache.field_complete(b, "asr.giga_ctc")  # no JSON -> not done

    def test_one_scandir_per_dir(self, tmp_path, monkeypatch):
        for i in range(5):
            cj.update_chunk_json(_audio(tmp_path, f"f{i}"), {"rover": "x"})

        real = os.scandir
        calls = {"n": 0}

        def counting(path):
            calls["n"] += 1
            return real(path)

        monkeypatch.setattr(os, "scandir", counting)
        cache = cj.ChunkJsonCache()
        for i in range(5):
            assert cache.field_complete(tmp_path / f"f{i}.flac", "rover")
        assert calls["n"] == 1  # one scandir for the whole directory

    def test_parsed_json_memoised(self, tmp_path):
        a = _audio(tmp_path, "a")
        cj.update_chunk_json(a, {"rover": "x"})
        cache = cj.ChunkJsonCache()
        first = cache.get(a)
        assert cache.get(a) is first  # same object on re-fetch


def test_pending_chunks_in_and_out_fields(tmp_path):
    # done: has rover + punct; need: has rover, no punct; upstream-missing: no rover
    done = _audio(tmp_path, "done")
    need = _audio(tmp_path, "need")
    upstream = _audio(tmp_path, "upstream")
    cj.update_chunk_json(done, {"rover": "r", "punct": "p"})
    cj.update_chunk_json(need, {"rover": "r"})
    cj.update_chunk_json(upstream, {"asr": {"giga_ctc": "t"}})  # no rover

    pending = cj.pending_chunks(tmp_path, out_field="punct", in_field="rover")
    assert [p.name for p in pending] == ["need.flac"]


def test_pending_chunks_no_in_field(tmp_path):
    a = _audio(tmp_path, "a")
    _audio(tmp_path, "b")  # no JSON at all -> rover pending
    cj.update_chunk_json(a, {"rover": "r"})
    pending = cj.pending_chunks(tmp_path, out_field="rover")
    assert sorted(p.name for p in pending) == ["b.flac"]


def test_partial_csv_writer_defers_metadata_sidecar_to_absorb(tmp_path):
    from src.utils.csv_manager import PartialCsvWriter, absorb_partial_csvs

    audio = _audio(tmp_path, "meta")
    with PartialCsvWriter(
        tmp_path,
        "meta_stage",
        0,
        fieldnames=("filepath", "DistillMOS", "p_tts"),
    ) as writer:
        writer.write({"filepath": str(audio), "DistillMOS": 4.2, "p_tts": 0.97})

    # Streaming a partial row no longer mirrors to JSON inline (that per-row RMW
    # in the worker hot loop was the dominant IO cost); the mirror happens when
    # the partials are absorbed into the state.
    assert not cj.chunk_json_path(audio).exists()

    absorb_partial_csvs(tmp_path, "meta_stage", ["DistillMOS", "p_tts"])

    data = cj.read_chunk_json(cj.chunk_json_path(audio))
    assert data["DistillMOS"] == 4.2
    assert data["p_tts"] == 0.97


def test_upsert_columns_updates_metadata_sidecar(tmp_path):
    import pandas as pd

    from src.utils.csv_manager import upsert_columns

    audio = _audio(tmp_path, "upsert")
    upsert_columns(
        tmp_path,
        pd.DataFrame({"filepath": [str(audio)], "total_duration": [12.5]}),
        ["total_duration"],
    )

    data = cj.read_chunk_json(cj.chunk_json_path(audio))
    assert data["total_duration"] == 12.5


def test_update_chunk_jsons_batch_merges_and_counts(tmp_path):
    a = _audio(tmp_path, "a")
    b = _audio(tmp_path, "b")
    cj.update_chunk_json(a, {"rover": "text"})  # pre-existing text key

    written, skipped, failed = cj.update_chunk_jsons(
        [(a, {"DistillMOS": 4.2}), (b, {"p_tts": 0.9})]
    )
    assert (written, skipped, failed) == (2, 0, 0)
    # Deep-merge preserves the existing text key alongside the new score.
    assert cj.read_chunk_json(cj.chunk_json_path(a)) == {
        "rover": "text",
        "DistillMOS": 4.2,
    }
    assert cj.read_chunk_json(cj.chunk_json_path(b)) == {"p_tts": 0.9}


def test_update_chunk_jsons_skips_noop_rewrite(tmp_path):
    a = _audio(tmp_path, "a")
    cj.update_chunk_jsons([(a, {"DistillMOS": 4.2})])
    jp = cj.chunk_json_path(a)
    mtime = jp.stat().st_mtime_ns

    # Identical update must not re-touch the file (no wasted atomic rewrite).
    written, skipped, failed = cj.update_chunk_jsons([(a, {"DistillMOS": 4.2})])
    assert (written, skipped, failed) == (0, 1, 0)
    assert jp.stat().st_mtime_ns == mtime


def test_update_chunk_jsons_skips_missing_audio(tmp_path):
    # Audio was deleted (e.g. by a filter stage); no orphan JSON is created.
    gone = tmp_path / "gone.flac"
    written, skipped, failed = cj.update_chunk_jsons([(gone, {"DistillMOS": 4.2})])
    assert (written, skipped, failed) == (0, 1, 0)
    assert not cj.chunk_json_path(gone).exists()

