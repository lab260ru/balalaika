"""Equivalence tests for AntiSpoofingDataset.ranged_decode fast path.

The ranged_decode option seeks+reads only the random window (plus one
predecessor sample for preemphasis) instead of fully decoding a long clip and
cropping. These tests assert that, for the eligible fixtures, the ranged output
is BIT-EXACTLY equal (torch.equal) to the full-decode-then-crop output, that
ineligible fixtures fall back to the full-decode path, and that the
random-number consumption stays identical so downstream RNG state matches.
"""

import random

import numpy as np
import pytest
import soundfile as sf
import torch

from src.utils.datasets.separation import AntiSpoofingDataset

SR = 16_000
# Small num_samples keeps fixtures tiny while still exercising the crop path.
NUM_SAMPLES = 4_000


def _write_wav(path, n_frames, channels=1, seed=0, subtype="PCM_16", sr=SR, fmt=None):
    rng = np.random.RandomState(seed)
    if subtype in ("FLOAT", "DOUBLE"):
        if channels == 1:
            data = (rng.rand(n_frames) * 2 - 1).astype(np.float32)
        else:
            data = (rng.rand(n_frames, channels) * 2 - 1).astype(np.float32)
    else:
        low, high = -32768, 32768
        if channels == 1:
            data = rng.randint(low, high, size=n_frames).astype(np.int32)
            # Force the extreme PCM16 values to stress float32 scaling.
            data[0], data[1], data[2] = -32768, 32767, 0
        else:
            data = rng.randint(low, high, size=(n_frames, channels)).astype(np.int32)
    sf.write(str(path), data, sr, subtype=subtype, format=fmt)
    return path


def _full_decode_then_crop(path, seed, sample_rate=SR, num_samples=NUM_SAMPLES):
    """Reference implementation: exactly the current full-decode path."""
    ds = AntiSpoofingDataset(
        [str(path)], sample_rate=sample_rate, num_samples=num_samples, ranged_decode=False
    )
    random.seed(seed)
    _path, waveform, original_length, error = ds[0]
    return waveform, original_length, error


def _ranged(path, seed, sample_rate=SR, num_samples=NUM_SAMPLES):
    ds = AntiSpoofingDataset(
        [str(path)], sample_rate=sample_rate, num_samples=num_samples, ranged_decode=True
    )
    random.seed(seed)
    _path, waveform, original_length, error = ds[0]
    return waveform, original_length, error


# ---------------------------------------------------------------------------
# Eligible fixtures: ranged output must equal full-decode-then-crop EXACTLY.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_16k_mono_wav_bit_exact(tmp_path, seed):
    path = _write_wav(tmp_path / "mono.wav", n_frames=SR * 2, channels=1, seed=3)
    full, full_len, full_err = _full_decode_then_crop(path, seed)
    ranged, ranged_len, ranged_err = _ranged(path, seed)
    assert full_err == "" and ranged_err == ""
    assert ranged.shape == full.shape == (NUM_SAMPLES,)
    assert torch.equal(ranged, full)
    assert ranged_len == full_len


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 99])
def test_16k_stereo_wav_bit_exact(tmp_path, seed):
    path = _write_wav(tmp_path / "stereo.wav", n_frames=SR * 2, channels=2, seed=7)
    full, full_len, full_err = _full_decode_then_crop(path, seed)
    ranged, ranged_len, ranged_err = _ranged(path, seed)
    assert full_err == "" and ranged_err == ""
    assert torch.equal(ranged, full)
    assert ranged_len == full_len


def test_flac_bit_exact(tmp_path):
    path = _write_wav(tmp_path / "mono.flac", n_frames=SR * 2, channels=1, seed=5, fmt="FLAC")
    for seed in (0, 11, 22, 33):
        full, full_len, _ = _full_decode_then_crop(path, seed)
        ranged, ranged_len, _ = _ranged(path, seed)
        assert torch.equal(ranged, full)
        assert ranged_len == full_len


def test_start_zero_edge_case(tmp_path):
    """When start==0 the window has no predecessor; preemphasis leaves y[0]=x[0].
    The ranged path must reproduce this exactly (read from offset 0, no drop)."""
    path = _write_wav(tmp_path / "mono.wav", n_frames=SR * 2, channels=1, seed=3)
    ds_full = AntiSpoofingDataset([str(path)], sample_rate=SR, num_samples=NUM_SAMPLES)
    ds_ranged = AntiSpoofingDataset(
        [str(path)], sample_rate=SR, num_samples=NUM_SAMPLES, ranged_decode=True
    )

    import unittest.mock as mock

    with mock.patch("random.randint", return_value=0):
        _, full, _, _ = ds_full[0]
    with mock.patch("random.randint", return_value=0):
        _, ranged, _, _ = ds_ranged[0]
    assert torch.equal(ranged, full)


def test_start_max_edge_case(tmp_path):
    """Largest valid start (window touches the end of the clip)."""
    n = SR * 2
    path = _write_wav(tmp_path / "mono.wav", n_frames=n, channels=1, seed=9)
    max_start = n - NUM_SAMPLES
    ds_full = AntiSpoofingDataset([str(path)], sample_rate=SR, num_samples=NUM_SAMPLES)
    ds_ranged = AntiSpoofingDataset(
        [str(path)], sample_rate=SR, num_samples=NUM_SAMPLES, ranged_decode=True
    )

    import unittest.mock as mock

    with mock.patch("random.randint", return_value=max_start):
        _, full, _, _ = ds_full[0]
    with mock.patch("random.randint", return_value=max_start):
        _, ranged, _, _ = ds_ranged[0]
    assert torch.equal(ranged, full)


# ---------------------------------------------------------------------------
# Ineligible fixtures: ranged_decode must take the full-decode fallback path
# and produce the SAME output as ranged_decode=False.
# ---------------------------------------------------------------------------

def test_short_clip_falls_back(tmp_path):
    """Clip shorter than num_samples is repeat-padded; ranged path must defer
    to the full-decode path and match it exactly."""
    path = _write_wav(tmp_path / "short.wav", n_frames=NUM_SAMPLES // 2, channels=1, seed=4)
    full, full_len, _ = _full_decode_then_crop(path, seed=0)
    ranged, ranged_len, _ = _ranged(path, seed=0)
    assert torch.equal(ranged, full)
    assert ranged_len == full_len


def test_48k_wav_falls_back(tmp_path):
    """48 kHz source must resample (unbounded context); ranged read is NOT
    equivalent there, so it must fall back to full-decode and match it."""
    path = _write_wav(tmp_path / "hi.wav", n_frames=48_000 * 2, channels=1, seed=6, sr=48_000)
    full, full_len, full_err = _full_decode_then_crop(path, seed=2)
    ranged, ranged_len, ranged_err = _ranged(path, seed=2)
    assert full_err == "" and ranged_err == ""
    assert torch.equal(ranged, full)
    assert ranged_len == full_len


def test_non_wav_falls_back(tmp_path):
    """A non-WAV/FLAC container (OGG) is ineligible; must fall back."""
    path = tmp_path / "a.ogg"
    rng = np.random.RandomState(8)
    data = (rng.rand(SR * 2) * 2 - 1).astype(np.float32)
    try:
        sf.write(str(path), data, SR, format="OGG", subtype="VORBIS")
    except Exception:
        pytest.skip("OGG/Vorbis not supported by libsndfile build")
    full, _, full_err = _full_decode_then_crop(path, seed=0)
    ranged, _, ranged_err = _ranged(path, seed=0)
    assert full_err == "" and ranged_err == ""
    # OGG/Vorbis is lossy, so both go through full-decode; outputs are identical
    # because the ranged path declines OGG and uses the same code as full.
    assert torch.equal(ranged, full)


# ---------------------------------------------------------------------------
# RNG consumption parity: a long-clip crop consumes exactly one randint() in
# BOTH paths, so the RNG state after __getitem__ must be identical.
# ---------------------------------------------------------------------------

def test_rng_consumption_matches_long_clip(tmp_path):
    path = _write_wav(tmp_path / "mono.wav", n_frames=SR * 2, channels=1, seed=3)

    random.seed(12345)
    AntiSpoofingDataset([str(path)], sample_rate=SR, num_samples=NUM_SAMPLES)[0]
    state_full = random.getstate()

    random.seed(12345)
    AntiSpoofingDataset(
        [str(path)], sample_rate=SR, num_samples=NUM_SAMPLES, ranged_decode=True
    )[0]
    state_ranged = random.getstate()

    assert state_full == state_ranged, "ranged path must consume RNG identically"


def test_randint_called_once_with_same_args(tmp_path, monkeypatch):
    """Both paths must call random.randint(0, wave_len - num_samples) once."""
    n = SR * 2
    path = _write_wav(tmp_path / "mono.wav", n_frames=n, channels=1, seed=3)
    expected_hi = n - NUM_SAMPLES

    calls = []
    real_randint = random.randint

    def spy(a, b):
        calls.append((a, b))
        return real_randint(a, b)

    monkeypatch.setattr(random, "randint", spy)

    random.seed(1)
    AntiSpoofingDataset([str(path)], sample_rate=SR, num_samples=NUM_SAMPLES)[0]
    full_calls = list(calls)

    calls.clear()
    random.seed(1)
    AntiSpoofingDataset(
        [str(path)], sample_rate=SR, num_samples=NUM_SAMPLES, ranged_decode=True
    )[0]
    ranged_calls = list(calls)

    assert full_calls == [(0, expected_hi)]
    assert ranged_calls == [(0, expected_hi)]


def test_default_behavior_is_full_decode(tmp_path):
    """ranged_decode defaults to False == exact current behavior."""
    path = _write_wav(tmp_path / "mono.wav", n_frames=SR * 2, channels=1, seed=3)
    ds = AntiSpoofingDataset([str(path)], sample_rate=SR, num_samples=NUM_SAMPLES)
    assert ds.ranged_decode is False
