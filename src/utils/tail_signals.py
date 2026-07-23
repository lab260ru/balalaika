"""Tail-clipping signals per SPEC_drop_criteria.md (§2-§5).

Stage-1 chunking historically cut chunk ends with no trailing-silence margin,
so a fraction of chunks end *on loud speech* — the final word is acoustically
truncated. On a balanced GigaAM-v3 check such chunks showed 14.3% last-word
error vs 3.1% for natural tails (×4.6 relative risk). Two per-chunk signals
quantify this:

* ``tail_db`` — loudest 20 ms RMS frame inside the last 80 ms, in dB relative
  to the utterance's robust peak (95th-percentile frame RMS). ``≈ 0`` means the
  audio is still at full level at the cut (clipped); ``≪ −20`` means it faded
  to silence (natural ending). The relative scale makes the signal invariant
  to global gain, so stage-3 BS.1770 loudness normalization can neither
  manufacture nor hide it.
* ``trailing_silence_ms`` — time from the end of the audio back to the last
  frame above −20 dB-relative. Below ~40 ms the model gets no acoustic
  end-of-sentence cue even when the tail is not loud.

Spec thresholds (applied downstream, not here): ``severe`` = tail_db > −6,
``clipped`` = tail_db > −12, ``abrupt`` = clipped OR trailing_silence < 40 ms.

Consumers: stage 3.5 (``src.preprocess.tail_score``) writes both signals to
``balalaika.parquet``; stage 1 reuses :func:`frame_rms` for energy-aware split
points. NumPy-only on purpose so fast CPU tests cover it.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

# Spec §3/§8: single source of truth for the analysis grid and thresholds.
HOP_S = 0.01
WIN_S = 0.02
PEAK_PERCENTILE = 95.0
TAIL_FRAMES = 8  # last 80 ms on the 10 ms hop grid
SILENCE_DB_REL_PEAK = -20.0
MIN_AUDIO_S = 0.1


class TailSignals(NamedTuple):
    tail_db: float
    trailing_silence_ms: float


def frame_rms(x: np.ndarray, sr: int) -> np.ndarray:
    """Sliding-window RMS on the spec's 10 ms hop / 20 ms window grid.

    Equivalent to the spec reference loop
    ``sqrt(mean(x[i*hop:i*hop+win]**2) + 1e-12)`` but vectorised via a float64
    cumulative sum; every frame holds exactly ``win`` samples, so the mean is
    the frame sum divided by ``win``.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        x = x.reshape(-1)
    hop = max(1, round(sr * HOP_S))
    win = max(2, round(sr * WIN_S))
    n = 1 + max(0, (len(x) - win) // hop)
    if len(x) < win:
        return np.empty(0, dtype=np.float64)
    cs = np.concatenate(([0.0], np.cumsum(np.square(x, dtype=np.float64))))
    starts = np.arange(n) * hop
    sums = cs[starts + win] - cs[starts]
    return np.sqrt(sums / win + 1e-12)


def tail_signals(x: np.ndarray, sr: int) -> Optional[TailSignals]:
    """Compute ``(tail_db, trailing_silence_ms)`` for one mono utterance.

    ``x`` is the mono waveform (any dtype castable to float); multi-channel
    input must be downmixed by the caller (spec §2: mean across channels).
    Returns ``None`` for audio shorter than 100 ms (too short to measure).
    """
    x = np.asarray(x)
    if x.ndim != 1:
        x = x.reshape(-1)
    if len(x) < MIN_AUDIO_S * sr:
        return None

    rms = frame_rms(x, sr)
    if len(rms) == 0:
        return None
    hop = max(1, round(sr * HOP_S))
    dur = len(x) / sr

    peak = np.percentile(rms, PEAK_PERCENTILE) + 1e-9
    tail_db = float(20.0 * np.log10((rms[-TAIL_FRAMES:].max() + 1e-9) / peak))

    thr = peak * (10.0 ** (SILENCE_DB_REL_PEAK / 20.0))
    above = np.flatnonzero(rms > thr)
    if len(above):
        last_voiced_end_s = (above[-1] + 1) * (hop / sr)
        trailing_ms = max(0.0, dur - last_voiced_end_s) * 1000.0
    else:
        trailing_ms = dur * 1000.0
    return TailSignals(tail_db, float(trailing_ms))
