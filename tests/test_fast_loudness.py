"""_integrated_loudness_fast must be bit-exact vs pyloudnorm's Meter.

The loudness stage's outputs were pinned byte-identical in report.md §5;
this suite keeps that bar for the vectorized gating path: identical LUFS
floats (not approximately — identical) and identical normalized arrays,
on real pipeline audio and on synthetic edge cases.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf

from src.preprocess.audio_postprocessing import _integrated_loudness_fast

BENCH_AUDIO = Path("cache/bench_sample/audio")


def lufs_equal(a: float, b: float) -> bool:
    if np.isnan(a) and np.isnan(b):
        return True
    return a == b


def assert_bit_exact(audio: np.ndarray, rate: int) -> None:
    meter = pyln.Meter(rate, block_size=0.400)
    stock = meter.integrated_loudness(audio.copy())
    fast = _integrated_loudness_fast(meter, audio.copy())
    assert lufs_equal(stock, fast), f"LUFS differ: stock={stock!r} fast={fast!r}"

    normalized_stock = pyln.normalize.loudness(audio, stock, -23.0)
    normalized_fast = pyln.normalize.loudness(audio, fast, -23.0)
    assert normalized_stock.dtype == normalized_fast.dtype
    assert np.array_equal(normalized_stock, normalized_fast, equal_nan=True)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("rate", [16_000, 44_100, 48_000])
def test_synthetic_noise(seed: int, channels: int, rate: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rate * 2.7)
    audio = (0.1 * rng.standard_normal((n, channels))).astype(np.float32)
    if channels == 1:
        audio = audio[:, 0]
    assert_bit_exact(audio, rate)


def test_float64_input() -> None:
    rng = np.random.default_rng(3)
    audio = 0.05 * rng.standard_normal(48_000 * 3)
    assert_bit_exact(audio, 48_000)


def test_near_silence_hits_gating_edge_cases() -> None:
    rate = 16_000
    audio = np.full(rate * 2, 1e-7, dtype=np.float32)
    assert_bit_exact(audio, rate)


def test_quiet_with_loud_burst() -> None:
    rate = 16_000
    rng = np.random.default_rng(5)
    audio = (1e-4 * rng.standard_normal(rate * 5)).astype(np.float32)
    audio[rate : rate + 4000] += 0.5  # only some blocks pass the gates
    assert_bit_exact(audio, rate)


def test_too_short_raises_like_stock() -> None:
    rate = 16_000
    meter = pyln.Meter(rate, block_size=0.400)
    audio = np.zeros(int(rate * 0.2), dtype=np.float32)
    with pytest.raises(ValueError):
        meter.integrated_loudness(audio.copy())
    with pytest.raises(ValueError):
        _integrated_loudness_fast(meter, audio.copy())


@pytest.mark.skipif(not BENCH_AUDIO.exists(), reason="bench sample not on this node")
def test_real_audio_sample() -> None:
    files = sorted(BENCH_AUDIO.rglob("*.wav"))[:40]
    assert files
    for p in files:
        audio, rate = sf.read(p)
        assert_bit_exact(np.asarray(audio), int(rate))
