"""Per-stage filter summary log shared across pipeline stages.

Filtering stages (preprocess chunking, crest filter, music detection, …) call
:func:`record_stage_summary` once with their before/after counts and total
duration in hours. Rows accumulate in ``<podcasts_path>/filter_summary.csv``.

The :mod:`src.report` script reads that CSV and renders a Markdown report so
operators can see how much audio was removed at each stage of the pipeline.

Use :func:`safe_audio_duration` to probe a single file's duration cheaply via
``torchaudio.info`` (falling back to ``soundfile``).
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional

from loguru import logger

AUDIT_FILENAME = "filter_summary.csv"
AUDIT_HEADERS = (
    "timestamp",
    "stage",
    "files_in",
    "files_out",
    "hours_in",
    "hours_out",
    "hours_removed",
    "params",
)


def audit_path(podcasts_path: os.PathLike | str) -> Path:
    """Return the canonical filter summary CSV path for a dataset root."""
    return Path(podcasts_path) / AUDIT_FILENAME


def record_stage_summary(
    podcasts_path: os.PathLike | str,
    stage: str,
    files_in: int,
    files_out: int,
    hours_in: float,
    hours_out: float,
    params: Optional[Mapping[str, object]] = None,
) -> Path:
    """Append a single filter-summary row for *stage* and return the CSV path.

    The CSV is created lazily; the header is written exactly once. ``params``
    is stored as a stable JSON blob so the report can render thresholds.
    """
    path = audit_path(podcasts_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()

    hours_in_f = float(hours_in or 0.0)
    hours_out_f = float(hours_out or 0.0)
    hours_removed = max(0.0, hours_in_f - hours_out_f)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage,
        "files_in": int(files_in),
        "files_out": int(files_out),
        "hours_in": round(hours_in_f, 4),
        "hours_out": round(hours_out_f, 4),
        "hours_removed": round(hours_removed, 4),
        "params": json.dumps(dict(params or {}), ensure_ascii=False, sort_keys=True),
    }

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    logger.info(
        f"[audit] {stage}: {files_in}->{files_out} files, "
        f"{hours_in_f:.2f}h -> {hours_out_f:.2f}h "
        f"(removed {hours_removed:.2f}h)"
    )
    return path


# Containers whose frame counts libsndfile reads exactly from the header —
# for these soundfile.info (~90 us) replaces torchaudio.info (~2.4 ms, spins
# up an ffmpeg StreamReader per call). mp3 and other estimated-length
# containers keep the torchaudio-first order (VBR mp3 frame counts from
# libsndfile can disagree with ffmpeg's).
# .ogg is intentionally excluded: libsndfile omits Vorbis padding frames that
# torchaudio/ffmpeg counts, producing ~28-31 ms shorter durations per file and
# drifting dataset-hours stats vs the pre-rewrite pipeline.
_SOUNDFILE_EXACT_SUFFIXES = {".wav", ".flac", ".opus", ".aiff", ".aif"}


def _soundfile_duration(p: str) -> float:
    import soundfile as sf

    with sf.SoundFile(p) as f:
        if f.samplerate > 0:
            return float(f.frames) / float(f.samplerate)
    return 0.0


def _torchaudio_duration(p: str) -> float:
    import torchaudio

    info = torchaudio.info(p)
    sr = int(getattr(info, "sample_rate", 0) or 0)
    n = int(getattr(info, "num_frames", 0) or 0)
    if sr > 0 and n > 0:
        return n / float(sr)
    return 0.0


def safe_audio_duration(path: os.PathLike | str) -> float:
    """Best-effort fast probe of an audio file duration in seconds.

    soundfile reads the header directly (~90 us); torchaudio.info goes
    through an ffmpeg StreamReader (~2.4 ms) but understands every
    container. Probe order depends on the extension; both are tried.
    Returns 0.0 if both probes fail (e.g. corrupted file).
    """
    p = str(path)
    sf_first = os.path.splitext(p)[1].lower() in _SOUNDFILE_EXACT_SUFFIXES
    probes = (
        (_soundfile_duration, _torchaudio_duration)
        if sf_first
        else (_torchaudio_duration, _soundfile_duration)
    )
    for probe in probes:
        try:
            duration = probe(p)
            if duration > 0:
                return duration
        except Exception:
            pass
    return 0.0


def total_hours(durations_seconds: Iterable[float]) -> float:
    """Sum a sequence of durations expressed in seconds and return hours."""
    return sum(float(d or 0.0) for d in durations_seconds) / 3600.0
