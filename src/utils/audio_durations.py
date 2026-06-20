"""Shared audio-duration cache backed by ``balalaika.csv``.

Long stages use durations for length-bucketed work shards. Probing every file
with ``torchaudio.info`` is expensive on large datasets, so ``total_duration``
in the main CSV is treated as the canonical cache. Missing values are probed
once and written back to ``balalaika.csv`` for downstream stages.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.utils.audit import safe_audio_duration
from src.utils.csv_manager import (
    _normalize_path_series,
    _read_state_narrow,
    _state_header,
    normalize_path_string,
    state_path,
    upsert_columns,
)
from src.utils.io_profile import effective_workers, resolve_io_profile

TOTAL_DURATION_COLUMN = "total_duration"
DEFAULT_DURATION_PROBE_WORKERS = 4
DEFAULT_BUCKET_SECONDS = 1.0
DEFAULT_MAX_BUCKET_DURATION = 15.0


def _positive_duration(value: object) -> Optional[float]:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    if duration > 0:
        return duration
    return None


def _normalise_requested_paths(paths: Iterable[str | Path]) -> list[tuple[str, str]]:
    requested: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in paths:
        original = str(raw).strip()
        if not original:
            continue
        normalized = normalize_path_string(original)
        if normalized in seen:
            continue
        seen.add(normalized)
        requested.append((original, normalized))
    return requested


def _csv_duration_cache(podcasts_path: str | Path, requested_norms: set[str]) -> dict[str, float]:
    # Read the duration column from the parquet state (projection is cheap).
    target = state_path(podcasts_path)
    if not target.exists() or not requested_norms:
        return {}

    # Sniff the header first (one line for CSV, schema for parquet) so we can
    # skip the body read entirely when the duration column does not exist yet,
    # and so the narrow read uses the multithreaded pyarrow engine / parquet
    # projection. The previous callable-`usecols` form forced the single-
    # threaded C engine and then ran a multi-million-row Python loop; the narrow
    # read + vectorized build below produces an identical
    # {normalized filepath -> positive duration} dict.
    header = _state_header(target)
    if header is None:
        return {}

    if "filepath" not in header or TOTAL_DURATION_COLUMN not in header:
        logger.info(f"{target.name} has no {TOTAL_DURATION_COLUMN} column yet.")
        return {}

    try:
        df = _read_state_narrow(target, ["filepath", TOTAL_DURATION_COLUMN])
    except Exception as exc:
        logger.warning(f"Could not read {TOTAL_DURATION_COLUMN} from {target}: {exc}")
        return {}

    if "filepath" not in df.columns or TOTAL_DURATION_COLUMN not in df.columns:
        logger.info(f"{target.name} has no {TOTAL_DURATION_COLUMN} column yet.")
        return {}

    # Vectorized equivalent of the old per-row loop: normalize the filepath
    # column once, coerce durations to numeric (non-numeric -> NaN, matching
    # _positive_duration's TypeError/ValueError branch), keep only requested,
    # positive (>0) values. dict() over the masked, last-wins ordering matches
    # the loop's "later row overwrites earlier" semantics.
    paths = _normalize_path_series(df["filepath"])
    durations_num = pd.to_numeric(df[TOTAL_DURATION_COLUMN], errors="coerce")
    mask = durations_num.gt(0) & paths.isin(requested_norms)
    durations: dict[str, float] = dict(
        zip(paths[mask].tolist(), durations_num[mask].astype(float).tolist())
    )

    logger.info(
        f"Loaded {len(durations)} cached {TOTAL_DURATION_COLUMN} values "
        f"from {target.name}."
    )
    return durations


def _probe_duration(path: str) -> tuple[str, float]:
    return path, float(safe_audio_duration(path) or 0.0)


def _probe_missing_durations(paths: list[str], num_workers: int) -> dict[str, float]:
    if not paths:
        return {}

    workers = max(1, int(num_workers or 1))
    logger.info(
        f"Probing {len(paths)} missing {TOTAL_DURATION_COLUMN} value(s) "
        f"with {workers} worker(s)."
    )

    if workers == 1:
        items = (_probe_duration(path) for path in paths)
        iterator = tqdm(items, total=len(paths), desc="probe_total_duration")
        return {path: duration for path, duration in iterator}

    durations: dict[str, float] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for path, duration in tqdm(
            executor.map(_probe_duration, paths, chunksize=128),
            total=len(paths),
            desc="probe_total_duration",
        ):
            durations[path] = duration
    return durations


def ensure_audio_durations(
    podcasts_path: str | Path,
    paths: Iterable[str | Path],
    *,
    num_workers: int = DEFAULT_DURATION_PROBE_WORKERS,
) -> dict[str, float]:
    """Return durations for ``paths`` and persist missing values to main CSV.

    The returned dict contains both the original path strings and normalized
    absolute paths as keys, so callers that keep their original path list can
    still look up durations without normalizing again.
    """
    requested = _normalise_requested_paths(paths)
    if not requested:
        return {}

    requested_norms = {normalized for _, normalized in requested}
    durations_by_norm = _csv_duration_cache(podcasts_path, requested_norms)
    missing = [
        normalized
        for _, normalized in requested
        if _positive_duration(durations_by_norm.get(normalized)) is None
    ]
    # Header probes in path order sweep the disk directory-by-directory
    # instead of seeking in CSV-row order; values are keyed by path, so the
    # result is order-independent. HDD datasets also cap probe fan-out: more
    # concurrent random readers on one spindle only multiply seeks.
    missing.sort()
    num_workers = effective_workers(
        num_workers, resolve_io_profile(str(podcasts_path)), role="probe"
    )

    probed = _probe_missing_durations(missing, num_workers=num_workers)
    positive_probed = {
        path: duration
        for path, duration in probed.items()
        if _positive_duration(duration) is not None
    }
    if positive_probed:
        updates = pd.DataFrame(
            {
                "filepath": list(positive_probed.keys()),
                TOTAL_DURATION_COLUMN: [
                    round(duration, 4) for duration in positive_probed.values()
                ],
            }
        )
        upsert_columns(
            podcasts_path,
            updates,
            value_columns=[TOTAL_DURATION_COLUMN],
            bootstrap_audio_paths=list(requested_norms),
            preserve_existing=True,
        )
        logger.info(
            f"Saved {len(positive_probed)} {TOTAL_DURATION_COLUMN} value(s) "
            "to balalaika.csv."
        )

    durations_by_norm.update(probed)

    out: dict[str, float] = {}
    for original, normalized in requested:
        duration = float(durations_by_norm.get(normalized, 0.0) or 0.0)
        out[original] = duration
        out[normalized] = duration
    return out


def _first_config_value(configs: tuple[Mapping[str, object], ...], key: str, default: object) -> object:
    for cfg in configs:
        if isinstance(cfg, Mapping) and cfg.get(key) is not None:
            return cfg[key]
    return default


def duration_probe_workers(*configs: Mapping[str, object], default: int = DEFAULT_DURATION_PROBE_WORKERS) -> int:
    value = _first_config_value(
        configs,
        "duration_probe_workers",
        default,
    )
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


def duration_bucket_settings(
    config_path: str | Path | None = None,
    *configs: Mapping[str, object],
) -> tuple[float, float]:
    """Return ``(bucket_seconds, max_bucket_duration)`` for length sharding."""
    _ = config_path
    bucket_seconds = _first_config_value(
        configs,
        "duration_bucket_seconds",
        DEFAULT_BUCKET_SECONDS,
    )
    max_duration = _first_config_value(
        configs,
        "max_bucket_duration",
        DEFAULT_MAX_BUCKET_DURATION,
    )
    try:
        bucket_seconds_f = max(0.001, float(bucket_seconds))
    except (TypeError, ValueError):
        bucket_seconds_f = DEFAULT_BUCKET_SECONDS
    try:
        max_duration_f = max(bucket_seconds_f, float(max_duration))
    except (TypeError, ValueError):
        max_duration_f = DEFAULT_MAX_BUCKET_DURATION
    return bucket_seconds_f, max_duration_f
