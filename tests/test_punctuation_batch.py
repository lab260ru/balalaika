"""Tests for the stage-9 punctuation batch refinements (no model needed).

Covers:
* make_punct_batch sorts the batched slab by token length (padding-waste cut)
  while keeping each file's output independent of batch order.
* Oversize (>512-token) texts are routed to the per-file path and never enter
  the batched pipeline call (so one long text can't fail / 2x-rerun its slab).
* The stage-end 'produced' count via ChunkJsonCache matches a naive per-file
  scan.

Stages read the ``rover`` text from each chunk's ``<stem>.json`` and write the
``punct`` key back. The pipeline ``model`` and the ``tokenizer`` are
monkeypatched with cheap stand-ins so these run on CPU with no transformers
download.

Run: .dev_venv/bin/python -m pytest tests/test_punctuation_batch.py -q
"""
from __future__ import annotations

from pathlib import Path

import pytest

import src.punctuation.punctuation as punct
from src.utils.chunk_json import (
    ChunkJsonCache,
    chunk_json_path,
    get_field,
    read_chunk_json,
    update_chunk_json,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """token count == number of whitespace-delimited words."""

    def __call__(self, text, truncation=False):
        return {"input_ids": list(range(len(text.split())))}


class _RecordingModel:
    """Records every call. A batched call (list arg) records the batch order;
    a single-text call (str arg) records a per-file invocation."""

    def __init__(self):
        self.batch_calls: list[list[str]] = []
        self.batch_sizes: list[int] = []
        self.single_calls: list[str] = []

    def _preds_for(self, text):
        return [{"word": text, "entity_group": "LOWER_O"}]

    def __call__(self, arg, batch_size=None):
        if isinstance(arg, list):
            self.batch_calls.append(list(arg))
            self.batch_sizes.append(batch_size)
            return [self._preds_for(t) for t in arg]
        self.single_calls.append(arg)
        return self._preds_for(arg)


@pytest.fixture
def fake_model(monkeypatch):
    model = _RecordingModel()
    monkeypatch.setattr(punct, "model", model)
    monkeypatch.setattr(punct, "tokenizer", _FakeTokenizer())
    monkeypatch.setattr(punct, "MODEL_MAX_TOKENS", 512)
    return model


def _write_rover(tmp_path: Path, stem: str, words: int) -> Path:
    """Create ``<stem>.wav`` + a chunk JSON with a rover transcript; return audio."""
    audio = tmp_path / f"{stem}.wav"
    audio.write_bytes(b"x")
    update_chunk_json(audio, {"rover": " ".join(["w"] * words)})
    return audio


def _punct(audio: Path):
    return get_field(read_chunk_json(chunk_json_path(audio)), "punct")


def _rover(audio: Path):
    return get_field(read_chunk_json(chunk_json_path(audio)), "rover")


# ---------------------------------------------------------------------------
# Sorting + routing
# ---------------------------------------------------------------------------

def test_batch_is_sorted_by_token_length(tmp_path, fake_model):
    paths = [
        _write_rover(tmp_path, "a", 30),
        _write_rover(tmp_path, "b", 5),
        _write_rover(tmp_path, "c", 17),
    ]
    punct.make_punct_batch(paths)

    assert len(fake_model.batch_calls) == 1
    batch = fake_model.batch_calls[0]
    word_counts = [len(t.split()) for t in batch]
    assert word_counts == sorted(word_counts), "batch not length-sorted"
    assert fake_model.single_calls == []
    for p in paths:
        assert _punct(p) is not None


def test_batch_size_capped_to_micro_batch(tmp_path, fake_model):
    paths = [_write_rover(tmp_path, f"f{i}", 5 + i) for i in range(18)]
    punct.make_punct_batch(paths)

    assert len(fake_model.batch_calls) == 1
    assert fake_model.batch_sizes == [punct.PUNCT_PIPELINE_BATCH]
    assert punct.PUNCT_PIPELINE_BATCH == 8


def test_batch_size_is_len_pending_when_below_micro_batch(tmp_path, fake_model):
    paths = [_write_rover(tmp_path, f"g{i}", 5 + i) for i in range(3)]
    punct.make_punct_batch(paths)

    assert fake_model.batch_sizes == [3]


def test_oversize_routed_to_per_file_not_batched(tmp_path, fake_model):
    short = _write_rover(tmp_path, "short", 10)
    huge = _write_rover(tmp_path, "huge", 1000)  # > 512 tokens
    punct.make_punct_batch([short, huge])

    assert all(len(t.split()) <= 512 for call in fake_model.batch_calls for t in call)
    assert any(len(t.split()) > 512 for t in fake_model.single_calls)
    for p in (short, huge):
        assert _punct(p) is not None


def test_existing_outputs_skipped(tmp_path, fake_model):
    p = _write_rover(tmp_path, "done", 12)
    update_chunk_json(p, {"punct": "already"})
    punct.make_punct_batch([p])
    assert fake_model.batch_calls == []
    assert fake_model.single_calls == []
    assert _punct(p) == "already"


def test_output_independent_of_batch_order(tmp_path, fake_model):
    paths = [_write_rover(tmp_path, f"f{i}", w) for i, w in enumerate([7, 40, 3, 22])]
    punct.make_punct_batch(paths)
    # process_token LOWER_O echoes the word, so each output == its input text.
    for p in paths:
        assert _punct(p) == _rover(p)


# ---------------------------------------------------------------------------
# produced-count via ChunkJsonCache == naive scan
# ---------------------------------------------------------------------------

def test_produced_count_matches_naive_scan(tmp_path):
    audios = []
    for i in range(20):
        a = tmp_path / f"file{i}.wav"
        a.write_bytes(b"x")
        update_chunk_json(a, {"rover": "x"})
        if i % 2 == 0:  # produce a punct key for even indices only
            update_chunk_json(a, {"punct": "y"})
        audios.append(a)

    naive = sum(1 for a in audios if _punct(a) is not None)
    cache = ChunkJsonCache()
    cached = sum(1 for a in audios if cache.field_complete(a, "punct"))
    assert cached == naive == 10
