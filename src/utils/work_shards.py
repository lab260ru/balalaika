"""Crash-tolerant file-list sharding for large pipeline stages.

The heavy stages may need to process tens of millions of file paths. Passing
those lists through ``multiprocessing`` forces Python to pickle multi-gigabyte
objects and can get the parent killed before children finish starting. This
module keeps the work queue on disk instead: the parent writes small shard
files, and workers atomically claim one shard at a time by renaming it.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from loguru import logger

DEFAULT_WORK_SHARD_SIZE = 10_000
WORK_ROOT_NAME = ".balalaika_work"


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


def prepare_work_shards(
    podcasts_path: str | Path,
    stage_name: str,
    paths: Iterable[str | Path],
    *,
    shard_size: int = DEFAULT_WORK_SHARD_SIZE,
    limit: Optional[int] = None,
) -> WorkShardPlan:
    """Write work shards for ``paths`` and return a small plan object.

    Existing work shards for the stage are discarded. The stage recomputes
    pending work from ``balalaika.csv`` before calling this, so stale shard
    files from an interrupted older run are safe to replace.
    """
    shard_size = max(1, int(shard_size or DEFAULT_WORK_SHARD_SIZE))
    work_dir = stage_work_dir(podcasts_path, stage_name)
    _reset_work_dir(work_dir)

    total = 0
    shard_count = 0
    current: List[str] = []

    for raw in paths:
        if limit is not None and total >= limit:
            break
        path = str(raw).strip()
        if not path:
            continue
        current.append(path)
        total += 1
        if len(current) >= shard_size:
            _write_one_shard(work_dir, shard_count, current)
            shard_count += 1
            current = []

    if current:
        _write_one_shard(work_dir, shard_count, current)
        shard_count += 1

    logger.info(
        f"Prepared {shard_count} work shard(s) with {total} item(s) "
        f"for {stage_name} at {work_dir} (shard_size={shard_size})."
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


def mark_work_shard_done(shard_path: str | Path) -> None:
    shard_path = Path(shard_path)
    base = shard_path.name.split(".running.", 1)[0]
    done = shard_path.with_name(f"{base}.done")
    shard_path.replace(done)
