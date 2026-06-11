"""Shared in-memory crest filtering and loudness normalization."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pyloudnorm as pyln
import torch

_METER_CACHE: dict[tuple[int, float], pyln.Meter] = {}


def _integrated_loudness_fast(meter: pyln.Meter, data: np.ndarray) -> float:
    """Bit-exact fast reimplementation of ``Meter.integrated_loudness``.

    pyloudnorm's block loop calls ``np.sum(np.square(block))`` on 75%%
    overlapping slices, squaring every sample ~4x and allocating a fresh
    block-sized temp each time; the gating passes are per-block Python list
    comprehensions with scalar ``np.log10``. Here each channel is squared
    ONCE and every reduction keeps the same element order, length and dtype
    as the original (numpy's pairwise summation depends only on those), so
    the returned LUFS — and therefore the normalized audio bytes — are
    identical. Pinned by tests/test_fast_loudness.py on real audio.

    Block boundaries replicate the original's per-block ``int()`` truncation
    (a fixed stride would round differently on some sample rates).
    """
    from pyloudnorm import util

    input_data = data.copy()
    util.valid_audio(input_data, meter.rate, meter.block_size)

    if input_data.ndim == 1:
        input_data = np.reshape(input_data, (input_data.shape[0], 1))

    num_channels = input_data.shape[1]
    num_samples = input_data.shape[0]

    for _filter_class, filter_stage in meter._filters.items():
        for ch in range(num_channels):
            input_data[:, ch] = filter_stage.apply_filter(input_data[:, ch])

    G = np.array([1.0, 1.0, 1.0, 1.41, 1.41])[:num_channels]
    T_g = meter.block_size
    Gamma_a = -70.0
    step = 1.0 - meter.overlap

    T = num_samples / meter.rate
    num_blocks = int(np.round(((T - T_g) / (T_g * step))) + 1)
    j_range = np.arange(0, num_blocks)
    z = np.zeros(shape=(num_channels, num_blocks))

    # Same truncated bounds as the original per-block loop.
    lower = (T_g * (j_range * step) * meter.rate).astype(np.int64)
    upper = (T_g * (j_range * step + 1) * meter.rate).astype(np.int64)
    scale = 1.0 / (T_g * meter.rate)
    for i in range(num_channels):
        squared = np.square(input_data[:, i])
        zi = z[i]
        for j in j_range:
            zi[j] = scale * np.sum(squared[lower[j]:upper[j]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # weighted per-block power; channel sum order matches np.sum(list)
        weighted = np.sum(G[:, np.newaxis] * z, axis=0)
        l = -0.691 + 10.0 * np.log10(weighted)

    J_g = np.flatnonzero(l >= Gamma_a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        z_first = z[:, J_g]
        z_avg_gated = np.array([np.mean(z_first[i]) for i in range(num_channels)])
        Gamma_r = -0.691 + 10.0 * np.log10(np.sum(G * z_avg_gated)) - 10.0

    J_g = np.flatnonzero((l > Gamma_r) & (l > Gamma_a))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        z_second = z[:, J_g]
        z_avg_gated = np.nan_to_num(
            np.array([np.mean(z_second[i]) for i in range(num_channels)])
        )

    with np.errstate(divide="ignore"):
        LUFS = -0.691 + 10.0 * np.log10(np.sum(G * z_avg_gated))

    return LUFS


@dataclass
class AudioPostprocessResult:
    samples: torch.Tensor
    crest_factor: float
    keep: bool
    loudness_normalized: bool
    loudness_error: Optional[str] = None


def config_flag_enabled(config: Mapping[str, object], key: str) -> bool:
    value = config.get(key, False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def fused_audio_preprocessing_enabled(config: Mapping[str, object]) -> bool:
    return config_flag_enabled(config, "fuse_audio_preprocessing")


def calculate_crest_factor(samples: torch.Tensor) -> float:
    waveform = samples.detach().to(dtype=torch.float32, device="cpu")
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)

    if waveform.numel() == 0:
        return float("inf")

    peak = waveform.abs().amax()
    rms = waveform.square().mean().sqrt()
    if float(rms) <= 0.0:
        return float("inf")
    return float((peak / rms).item())


def _get_meter(rate: int, block_size: float) -> pyln.Meter:
    key = (int(rate), float(block_size))
    meter = _METER_CACHE.get(key)
    if meter is None:
        meter = pyln.Meter(rate, block_size=block_size)
        _METER_CACHE[key] = meter
    return meter


def normalize_audio_loudness(
    audio: np.ndarray,
    rate: int,
    peak: float = -1.0,
    loudness: float = -23.0,
    block_size: float = 0.400,
) -> np.ndarray:
    """Normalize mono or frames-first multichannel audio to target LUFS."""
    _ = peak
    meter = _get_meter(rate, block_size)
    try:
        measured = _integrated_loudness_fast(meter, audio)
    except Exception:
        # Any surprise (e.g. >5 channels) falls back to the stock meter,
        # which raises/behaves exactly as before this optimization.
        measured = meter.integrated_loudness(audio)
    return pyln.normalize.loudness(audio, measured, loudness)


def postprocess_audio_tensor(
    samples: torch.Tensor,
    sample_rate: int,
    *,
    crest_threshold: float,
    peak: float,
    loudness: float,
    block_size: float,
) -> AudioPostprocessResult:
    """Apply crest filtering and LUFS normalization without intermediate I/O."""
    audio = samples.detach().to(dtype=torch.float32, device="cpu")
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    audio = audio.contiguous()

    crest_factor = calculate_crest_factor(audio)
    if crest_factor > float(crest_threshold):
        return AudioPostprocessResult(
            samples=audio,
            crest_factor=crest_factor,
            keep=False,
            loudness_normalized=False,
        )

    try:
        audio_np = audio.numpy()
        if audio_np.shape[0] == 1:
            normalized = normalize_audio_loudness(
                audio_np.squeeze(0),
                sample_rate,
                peak=peak,
                loudness=loudness,
                block_size=block_size,
            )[np.newaxis, :]
        else:
            normalized = normalize_audio_loudness(
                audio_np.T,
                sample_rate,
                peak=peak,
                loudness=loudness,
                block_size=block_size,
            ).T
        normalized_tensor = torch.as_tensor(
            np.ascontiguousarray(normalized),
            dtype=torch.float32,
        )
        return AudioPostprocessResult(
            samples=normalized_tensor,
            crest_factor=crest_factor,
            keep=True,
            loudness_normalized=True,
        )
    except Exception as exc:
        return AudioPostprocessResult(
            samples=audio,
            crest_factor=crest_factor,
            keep=True,
            loudness_normalized=False,
            loudness_error=str(exc),
        )
