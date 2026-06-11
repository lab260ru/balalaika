"""Tests for the stage-8 punctuation batch refinements (no model needed).

Covers:
* make_punct_batch sorts the batched slab by token length (padding-waste cut)
  while keeping each file's output independent of batch order.
* Oversize (>512-token) texts are routed to the per-file path and never enter
  the batched pipeline call (so one long text can't fail / 2x-rerun its slab).
* The stage-end 'produced' count via DirNameCache matches a naive per-file
  Path.exists() scan.

The pipeline `model` and the `tokenizer` are monkeypatched with cheap stand-ins
so these run on CPU with no transformers download.

Run: .dev_venv/bin/python -m pytest tests/test_punctuation_batch.py -q
"""
from __future__ import annotations

from pathlib import Path

import pytest

import src.punctuation.punctuation as punct


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """token count == number of whitespace-delimited words."""

    def __call__(self, text, truncation=False):
        return {"input_ids": list(range(len(text.split())))}


class _RecordingModel:
    """Records every call. A batched call (list arg) records the batch order;
    a single-text call (str arg) records a per-file invocation.

    Returns a prediction list shaped like the HF pipeline: one list of
    token dicts per input text, where the single token's 'word' echoes the
    input so _punct_text_from_preds reproduces it (process_token LOWER_O
    passes the word through unchanged).
    """

    def __init__(self):
        self.batch_calls: list[list[str]] = []
        self.single_calls: list[str] = []

    def _preds_for(self, text):
        # One token group covering the whole text; LOWER_O => word unchanged.
        return [{"word": text, "entity_group": "LOWER_O"}]

    def __call__(self, arg, batch_size=None):
        if isinstance(arg, list):
            self.batch_calls.append(list(arg))
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
    p = tmp_path / f"{stem}_rover.txt"
    p.write_text(" ".join(["w"] * words), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Sorting + routing
# ---------------------------------------------------------------------------

def test_batch_is_sorted_by_token_length(tmp_path, fake_model):
    # Three in-budget files with descending word counts in discovery order.
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
    # Each output file written with its own content.
    for p in paths:
        out = p.with_name(p.name.replace("_rover.txt", "_punct.txt"))
        assert out.exists()


def test_oversize_routed_to_per_file_not_batched(tmp_path, fake_model):
    short = _write_rover(tmp_path, "short", 10)
    huge = _write_rover(tmp_path, "huge", 1000)  # > 512 tokens
    punct.make_punct_batch([short, huge])

    # The huge text must NOT appear in any batched call.
    assert all(len(t.split()) <= 512 for call in fake_model.batch_calls for t in call)
    # It must have been processed per-file.
    assert any(len(t.split()) > 512 for t in fake_model.single_calls)
    for p in (short, huge):
        out = p.with_name(p.name.replace("_rover.txt", "_punct.txt"))
        assert out.exists()


def test_existing_outputs_skipped(tmp_path, fake_model):
    p = _write_rover(tmp_path, "done", 12)
    out = p.with_name(p.name.replace("_rover.txt", "_punct.txt"))
    out.write_text("already", encoding="utf-8")
    punct.make_punct_batch([p])
    assert fake_model.batch_calls == []
    assert fake_model.single_calls == []
    assert out.read_text(encoding="utf-8") == "already"


def test_output_independent_of_batch_order(tmp_path, fake_model):
    paths = [_write_rover(tmp_path, f"f{i}", w) for i, w in enumerate([7, 40, 3, 22])]
    punct.make_punct_batch(paths)
    # process_token LOWER_O echoes the word, so each output == its input text.
    for p in paths:
        out = p.with_name(p.name.replace("_rover.txt", "_punct.txt"))
        assert out.read_text(encoding="utf-8") == p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# produced-count via DirNameCache == naive scan
# ---------------------------------------------------------------------------

def test_produced_count_matches_naive_scan(tmp_path):
    from src.utils.sidecars import DirNameCache

    rover_paths = []
    for i in range(20):
        p = tmp_path / f"file{i}_rover.txt"
        p.write_text("x", encoding="utf-8")
        rover_paths.append(p)
        # Produce a punct sidecar for even indices only.
        if i % 2 == 0:
            p.with_name(p.name.replace("_rover.txt", "_punct.txt")).write_text(
                "y", encoding="utf-8"
            )

    naive = sum(
        1
        for rp in rover_paths
        if rp.with_name(rp.name.replace("_rover.txt", "_punct.txt")).exists()
    )
    cache = DirNameCache()
    cached = sum(
        1
        for rp in rover_paths
        if cache.exists(rp.with_name(rp.name.replace("_rover.txt", "_punct.txt")))
    )
    assert cached == naive == 10
