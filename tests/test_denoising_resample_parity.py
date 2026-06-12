"""Parity test for the cached DenoisingDataset resampler.

The dataset caches a ``torchaudio.transforms.Resample`` per source rate to
avoid rebuilding the polyphase kernel on every file. That cache is only a
legitimate perf win if it stays bit-for-bit identical to the
``torchaudio.functional.resample`` call it replaced.

``transforms.Resample`` builds its sinc kernel in ``torch.float64`` unless an
explicit ``dtype`` is passed, then casts to the input dtype at forward time;
``functional.resample`` on a float32 waveform builds a float32 kernel. Those
two kernels are NOT equal, so without ``dtype=torch.float32`` the cached path
diverges from the functional call (measured: max|diff| ~2e-5 on the float
output, up to ~2.5% of int16 samples flip +-1 after the dataset's int16
normalization). These tests pin that parity at the source rates the pipeline
actually sees.

Pure CPU; no model / network.

Run: .dev_venv/bin/python -m pytest tests/test_denoising_resample_parity.py -q
"""
from __future__ import annotations

import pytest
import torch
import torchaudio

from src.utils.datasets.denoising import (
    DENOISING_SAMPLE_RATE,
    DenoisingDataset,
    normalize_to_int16,
)


@pytest.mark.parametrize("source_sample_rate", [44100, 22050, 32000])
def test_cached_resample_matches_functional(source_sample_rate):
    torch.manual_seed(source_sample_rate)
    waveform = torch.randn(1, source_sample_rate // 2, dtype=torch.float32)

    ds = DenoisingDataset(["x"], sample_rate=DENOISING_SAMPLE_RATE)
    cached = ds._resample(waveform, source_sample_rate)
    functional = torchaudio.functional.resample(
        waveform, source_sample_rate, DENOISING_SAMPLE_RATE
    )

    # The cached transform must reproduce the functional call bit-for-bit on
    # the float output ...
    assert torch.equal(cached, functional)
    # ... and after the dataset's actual int16 normalization (what is written
    # to disk / fed to the model). Even sub-1e-5 float drift flips int16 LSBs.
    assert np_equal(
        normalize_to_int16(cached.squeeze(0).numpy()),
        normalize_to_int16(functional.squeeze(0).numpy()),
    )


def np_equal(a, b) -> bool:
    return bool((a == b).all())
