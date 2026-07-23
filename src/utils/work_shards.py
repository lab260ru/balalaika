"""Crash-tolerant file-list sharding for large pipeline stages.

The heavy stages may need to process tens of millions of file paths. Passing
those lists through ``multiprocessing`` forces Python to pickle multi-gigabyte
objects and can get the parent killed before children finish starting. This
module keeps the work queue on disk instead: the parent writes small shard
files, and workers atomically claim one shard at a time by renaming it.
"""
from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional

from loguru import logger
from tqdm import tqdm

DEFAULT_WORK_SHARD_SIZE = 10_000
WORK_ROOT_NAME = ".balalaika_work"

# Shard ordering knob. "path" (default) writes shard lines in lexicographic
# path order so workers read the disk in directory-clustered order — on HDD
# datasets this turns ~one random seek per file into ~one per directory.
# "legacy" preserves the exact pre-2026-06 order (input order for plain
# shards, duration order inside length buckets).
WORK_SHARD_ORDER_ENV = "BALALAIKA_SHARD_ORDER"


def _shard_order(explicit: Optional[str] = None) -> str:
    value = explicit if explicit is not None else os.environ.get(WORK_SHARD_ORDER_ENV, "path")
    value = str(value).strip().lower()
    # Config-facing aliases for the old behavior.
    value = {"duration": "legacy", "input": "legacy"}.get(value, value)
    if value not in {"path", "legacy"}:
        logger.warning(
            f"Unknown shard order {value!r} (from "
            f"{'argument' if explicit is not None else WORK_SHARD_ORDER_ENV}); using 'path'."
        )
        return "path"
    return value


@dataclass(frozen=True)
class WorkShardPlan:
    work_dir: Path
    total_items: int
    shard_count: int
    shard_size: int


def load_work_shard_size(config_path: Optional[str | Path], default: int = DEFAULT_WORK_SHARD_SIZE) -> int:
    """Read ``runtime.work_shard_size`` from YAML with a safe default."""
    if not config_path:
        return max(1, int(default))
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        runtime = raw.get("runtime", {}) if isinstance(raw, dict) else {}
        value = runtime.get("work_shard_size", default) if isinstance(runtime, dict) else default
        return max(1, int(value))
    except Exception as exc:
        logger.warning(f"Could not read runtime.work_shard_size from {config_path}: {exc}; using {default}.")
        return max(1, int(default))


def stage_work_dir(podcasts_path: str | Path, stage_name: str) -> Path:
    safe_stage = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in stage_name)
    return Path(podcasts_path) / WORK_ROOT_NAME / safe_stage


def _reset_work_dir(work_dir: Path) -> None:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)


def _write_one_shard(work_dir: Path, shard_index: int, paths: List[str]) -> None:
    final = work_dir / f"shard_{shard_index:06d}.pending"
    tmp = final.with_suffix(final.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        for path in paths:
            f.write(path)
            f.write("\n")
    tmp.replace(final)


def _write_labeled_shard(
    work_dir: Path,
    shard_index: int,
    label: str,
    paths: List[str],
    annotations: Optional[Mapping[str, str]] = None,
) -> None:
    final = work_dir / f"shard_{shard_index:06d}_{label}.pending"
    tmp = final.with_suffix(final.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        for path in paths:
            f.write(path)
            if annotations is not None:
                note = annotations.get(path, "")
                if note:
                    f.write("\t")
                    f.write(note)
            f.write("\n")
    tmp.replace(final)


def _duration_bucket_label(bucket_index: int, bucket_seconds: float, max_duration: float) -> str:
    lower = bucket_index * bucket_seconds
    upper = lower + bucket_seconds
    if lower >= max_duration:
        return f"len_ge_{int(round(max_duration)):03d}s"
    return f"len_{int(lower):03d}_{int(upper):03d}s"


def _duration_bucket_index(duration: float, bucket_seconds: float, max_duration: float) -> int:
    if duration <= 0:
        return 0
    bucket_count = max(1, int(math.ceil(max_duration / bucket_seconds)))
    if duration <= max_duration:
        return min(int(duration / bucket_seconds), bucket_count - 1)
    return bucket_count


def prepare_work_shards(
    podcasts_path: str | Path,
    stage_name: str,
    paths: Iterable[str | Path],
    *,
    shard_size: int = DEFAULT_WORK_SHARD_SIZE,
    limit: Optional[int] = None,
    annotations: Optional[Mapping[str, str]] = None,
    order: Optional[str] = None,
) -> WorkShardPlan:
    """Write work shards for ``paths`` and return a small plan object.

    Existing work shards for the stage are discarded. The stage recomputes
    pending work from ``balalaika.csv`` before calling this, so stale shard
    files from an interrupted older run are safe to replace.

    ``annotations`` optionally maps a path to a short string stored after a
    tab on the same line (read back with :func:`read_annotated_work_shard`).

    ``order`` (default: ``$BALALAIKA_SHARD_ORDER`` or ``"path"``) controls
    shard line ordering. ``"path"`` sorts lexicographically so HDD reads
    cluster by directory; it holds one list of path references in RAM
    (callers pass materialized lists anyway). ``limit`` keeps its original
    meaning either way: the first ``limit`` items *of the input order* are
    selected, then ordered. ``"legacy"`` streams in input order.
    """
    shard_size = max(1, int(shard_size or DEFAULT_WORK_SHARD_SIZE))
    order = _shard_order(order)
    work_dir = stage_work_dir(podcasts_path, stage_name)
    _reset_work_dir(work_dir)

    total = 0
    shard_count = 0
    current: List[str] = []
    collected: List[str] = []
    expected_total = (
        limit
        if limit is not None
        else (len(paths) if hasattr(paths, "__len__") else None)
    )

    def flush(paths_chunk: List[str], index: int) -> None:
        if annotations is None:
            _write_one_shard(work_dir, index, paths_chunk)
        else:
            _write_labeled_shard(work_dir, index, "plain", paths_chunk, annotations)

    for raw in tqdm(
        paths,
        total=expected_total,
        desc=f"write_{stage_name}_work_shards",
    ):
        if limit is not None and total >= limit:
            break
        path = str(raw).strip()
        if not path:
            continue
        total += 1
        if order == "path":
            collected.append(path)
            continue
        current.append(path)
        if len(current) >= shard_size:
            flush(current, shard_count)
            shard_count += 1
            current = []

    if order == "path":
        collected.sort()
        for start in range(0, len(collected), shard_size):
            flush(collected[start:start + shard_size], shard_count)
            shard_count += 1
    elif current:
        flush(current, shard_count)
        shard_count += 1

    logger.info(
        f"Prepared {shard_count} work shard(s) with {total} item(s) "
        f"for {stage_name} at {work_dir} (shard_size={shard_size})."
    )
    return WorkShardPlan(work_dir=work_dir, total_items=total, shard_count=shard_count, shard_size=shard_size)


def prepare_length_bucketed_work_shards(
    podcasts_path: str | Path,
    stage_name: str,
    paths: Iterable[str | Path],
    durations: Mapping[str, float],
    *,
    shard_size: int = DEFAULT_WORK_SHARD_SIZE,
    bucket_seconds: float = 1.0,
    max_duration: float = 15.0,
    limit: Optional[int] = None,
    annotations: Optional[Mapping[str, str]] = None,
    order: Optional[str] = None,
) -> WorkShardPlan:
    """Write work shards grouped by audio duration buckets.

    Every produced shard contains files from a single duration bucket, e.g.
    0-1s, 1-2s, ... up to ``max_duration``. Files longer than the configured
    max go into a separate overflow bucket so they do not inflate normal
    short-clip batches. Existing stage shards are discarded, same as
    :func:`prepare_work_shards`.

    ``annotations`` optionally maps a path to a short string stored after a
    tab on the same shard line (used by grouped transcription to record
    which models still need the file). Read such shards back with
    :func:`read_annotated_work_shard`; the plain :func:`read_work_shard`
    would return the raw tab-joined lines.

    ``order`` (default: ``$BALALAIKA_SHARD_ORDER`` or ``"path"``): with
    ``"path"``, lines inside each *bounded* bucket are sorted by path so HDD
    reads cluster by directory — padding stays bounded by ``bucket_seconds``
    because bucket membership is unchanged. The overflow bucket
    (``> max_duration``, unbounded width) keeps the duration sort either
    way. ``"legacy"`` keeps the duration sort in every bucket.
    """
    shard_size = max(1, int(shard_size or DEFAULT_WORK_SHARD_SIZE))
    order = _shard_order(order)
    bucket_seconds = max(0.001, float(bucket_seconds or 1.0))
    max_duration = max(bucket_seconds, float(max_duration or bucket_seconds))
    overflow_index = max(1, int(math.ceil(max_duration / bucket_seconds)))
    work_dir = stage_work_dir(podcasts_path, stage_name)
    _reset_work_dir(work_dir)

    buckets: dict[int, List[str]] = {}
    total = 0
    expected_total = (
        limit
        if limit is not None
        else (len(paths) if hasattr(paths, "__len__") else None)
    )

    for raw in tqdm(
        paths,
        total=expected_total,
        desc=f"bucket_{stage_name}_work_shards",
    ):
        if limit is not None and total >= limit:
            break
        path = str(raw).strip()
        if not path:
            continue
        duration = float(durations.get(path, 0.0) or 0.0)
        bucket_index = _duration_bucket_index(duration, bucket_seconds, max_duration)
        buckets.setdefault(bucket_index, []).append(path)
        total += 1

    shard_count = 0
    for bucket_index in sorted(buckets):
        label = _duration_bucket_label(bucket_index, bucket_seconds, max_duration)
        if order == "path" and bucket_index < overflow_index:
            bucket_paths = sorted(buckets[bucket_index])
        else:
            bucket_paths = sorted(
                buckets[bucket_index],
                key=lambda p: float(durations.get(p, 0.0) or 0.0),
            )
        for start in range(0, len(bucket_paths), shard_size):
            _write_labeled_shard(
                work_dir,
                shard_count,
                label,
                bucket_paths[start:start + shard_size],
                annotations,
            )
            shard_count += 1

    logger.info(
        f"Prepared {shard_count} length-bucketed work shard(s) with {total} item(s) "
        f"for {stage_name} at {work_dir} "
        f"(bucket_seconds={bucket_seconds}, max_duration={max_duration}, shard_size={shard_size})."
    )
    return WorkShardPlan(work_dir=work_dir, total_items=total, shard_count=shard_count, shard_size=shard_size)


def claim_work_shard(work_dir: str | Path, worker_id: int) -> Optional[Path]:
    """Atomically claim one pending shard for a worker."""
    work_dir = Path(work_dir)
    for pending in sorted(work_dir.glob("shard_*.pending")):
        running = pending.with_name(f"{pending.stem}.running.{worker_id}")
        try:
            pending.rename(running)
            return running
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.debug(f"Could not claim work shard {pending}: {exc}")
            continue
    return None


def read_work_shard(shard_path: str | Path) -> List[str]:
    with Path(shard_path).open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_annotated_work_shard(shard_path: str | Path) -> List[tuple[str, str]]:
    """Read a shard written with ``annotations`` as (path, annotation) pairs.

    Lines without a tab (plain shards) yield an empty annotation.
    """
    items: List[tuple[str, str]] = []
    with Path(shard_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, _, note = line.partition("\t")
            items.append((path, note))
    return items


def mark_work_shard_done(shard_path: str | Path) -> None:
    shard_path = Path(shard_path)
    base = shard_path.name.split(".running.", 1)[0]
    done = shard_path.with_name(f"{base}.done")
    shard_path.replace(done)
