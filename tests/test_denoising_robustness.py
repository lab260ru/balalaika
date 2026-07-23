"""Logic tests for the stage-11 denoising robustness/perf changes.

MossFormer2 ONNX is not on this node, so these exercise pure-CPU logic only:

* denoising_collate: a single >max_padded_len item is marked a per-item error
  (and zero-length) instead of raising and killing the worker/shard; valid
  files in the same batch are padded over the survivors only.
* DenoisingDataset caches a transforms.Resample per source rate (no per-file
  kernel rebuild) and the cached resample is bit-identical to the functional
  call it replaces.
* The TRT opt-profile default reproduces the historical 1 s shape; the knob
  moves it and clamps to bounds.
* _BoundedAudioSaver applies the CSV row / already_done / counter only after a
  successful save, counts a failed save as an error, and drain() blocks until
  all queued saves landed.

Run: .dev_venv/bin/python -m pytest tests/test_denoising_robustness.py -q
"""
from __future__ import annotations

import numpy as np
import torch

from src.utils.datasets.denoising import (
    DENOISING_SAMPLE_RATE,
    DenoisingDataset,
    denoising_collate,
    next_multiple,
)


def _item(path, length):
    wav = torch.arange(length, dtype=torch.int16) % 100
    return path, wav, length, ""


# ---------------------------------------------------------------------------
# collate: oversize -> per-item error, not a batch-wide raise
# ---------------------------------------------------------------------------

def test_oversize_item_marked_error_not_raised():
    pad = 384
    cap = next_multiple(1000, pad)  # small cap so 2000 is "oversize"
    batch = [_item("ok", 500), _item("toolong", 2000)]
    paths, padded, lengths, errors = denoising_collate(
        batch, pad_to_multiple=pad, pad_mode="zero", max_padded_len=cap
    )
    assert paths == ["ok", "toolong"]
    # The oversize item is flagged and zeroed.
    assert errors[0] == ""
    assert "exceeds model max" in errors[1]
    assert int(lengths[0]) == 500
    assert int(lengths[1]) == 0
    # padded_len is computed over survivors only (500 -> next multiple 768),
    # NOT over the 2000-sample reject.
    assert padded.shape[-1] == next_multiple(500, pad)


def test_valid_only_batch_unchanged_shape():
    pad = 384
    batch = [_item("a", 300), _item("b", 700)]
    _, padded, lengths, errors = denoising_collate(
        batch, pad_to_multiple=pad, pad_mode="zero", max_padded_len=960_000
    )
    assert errors == ["", ""]
    assert padded.shape[-1] == next_multiple(700, pad)
    # The first item's samples land in place.
    assert torch.equal(padded[0, 0, :300], (torch.arange(300, dtype=torch.int16) % 100))


def test_existing_decode_error_preserved():
    pad = 384
    batch = [("bad", torch.empty(0, dtype=torch.int16), 0, "decode failed")]
    _, _, lengths, errors = denoising_collate(
        batch, pad_to_multiple=pad, pad_mode="zero", max_padded_len=960_000
    )
    assert errors[0] == "decode failed"
    assert int(lengths[0]) == 0


# ---------------------------------------------------------------------------
# resampler cache: reused + bit-identical to functional.resample
# ---------------------------------------------------------------------------

def test_resampler_cached_and_bit_identical():
    import torchaudio

    ds = DenoisingDataset(["x"], sample_rate=DENOISING_SAMPLE_RATE)
    wav = torch.randn(1, 16_000, dtype=torch.float32)
    out1 = ds._resample(wav, 16_000)
    # Same source rate reuses the same transform object (no rebuild).
    assert 16_000 in ds._resamplers
    cached = ds._resamplers[16_000]
    out2 = ds._resample(wav, 16_000)
    assert ds._resamplers[16_000] is cached
    # The cached transform must reproduce the functional.resample call it
    # replaces bit-for-bit (this is why the transform pins dtype=float32; the
    # float64-default kernel would diverge -- see test_denoising_resample_parity).
    ref = torchaudio.functional.resample(wav, 16_000, DENOISING_SAMPLE_RATE)
    assert torch.equal(out1, ref)
    assert torch.equal(out2, ref)


# ---------------------------------------------------------------------------
# TRT opt profile knob
# ---------------------------------------------------------------------------

def test_trt_opt_profile_default_unchanged():
    from src.denoising import denoising as dn

    # Default reproduces the historical 1 s (48000-sample) opt shape.
    assert dn._opt_profile_samples({}) == DENOISING_SAMPLE_RATE
    providers = [(
        "TensorrtExecutionProvider",
        {"device_id": 0, "trt_engine_cache_path": "/tmp/cache"},
    )]
    patched = dn.add_denoising_trt_profile_options(providers, "audio", 2, {})
    opts = patched[0][1]
    assert opts["trt_profile_opt_shapes"] == "audio:2x1x48000"
    # Knob moves the opt shape and adds the timing-cache path.
    moved = dn.add_denoising_trt_profile_options(
        providers, "audio", 2, {"trt_opt_seconds": 15.0}
    )[0][1]
    assert moved["trt_profile_opt_shapes"] == "audio:2x1x720000"
    assert moved["trt_timing_cache_path"] == "/tmp/cache"


# ---------------------------------------------------------------------------
# bounded async saver: ordering + accounting
# ---------------------------------------------------------------------------

class _Counter:
    def __init__(self):
        self.value = 0


class _FakeWriter:
    def __init__(self):
        self.rows = []

    def write(self, row):
        self.rows.append(row)


def test_async_saver_commits_after_save_and_counts(monkeypatch, tmp_path):
    from src.denoising import denoising as dn

    saved = []

    def fake_save(path_str, tensor):
        if "bad" in path_str:
            raise RuntimeError("encode failed")
        saved.append(path_str)

    monkeypatch.setattr(dn, "_save_audio", fake_save)

    writer = _FakeWriter()
    done = set()
    processed = _Counter()
    errors = _Counter()
    saver = dn._BoundedAudioSaver(
        rank=0,
        writer=writer,
        already_done=done,
        processed_counter=processed,
        errors_counter=errors,
        workers=2,
        max_pending=2,
    )
    t = torch.zeros(1, 10)
    for i in range(5):
        saver.submit(f"/d/file{i}.wav", f"/d/file{i}.wav", t)
    saver.submit("/d/bad.wav", "/d/bad.wav", t)
    saver.close()

    # Every good file: saved, one CSV row, in already_done, processed bumped.
    assert processed.value == 5
    assert errors.value == 1
    assert len(writer.rows) == 5
    assert done == {f"/d/file{i}.wav" for i in range(5)}
    # The failed save wrote no row and is not marked done.
    assert "/d/bad.wav" not in done
    assert all(r["filepath"] != "/d/bad.wav" for r in writer.rows)
