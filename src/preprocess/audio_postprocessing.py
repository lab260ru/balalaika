"""Shared in-memory crest filtering and loudness normalization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pyloudnorm as pyln
import torch

_METER_CACHE: dict[tuple[int, float], pyln.Meter] = {}


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
