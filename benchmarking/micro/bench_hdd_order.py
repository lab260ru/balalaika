"""Seek-model benchmark for work-shard read ordering on HDD.

This node has no spinning disk, so instead of wall-clock we replay the exact
file-open order each shard plan produces over a synthetic dataset layout and
score it with an explicit HDD cost model. Physical position is modeled as
creation order: podcast/episode directories are written sequentially as the
downloader and chunker produce them, so files of one directory sit together
on the platter.

Cost model (single read stream):
  * next file within +READAHEAD files of the previous one -> short seek
    (track-to-track / readahead hit), SHORT_MS
  * anything else -> random seek, SEEK_MS

Usage:
  python -m benchmarking.micro.bench_hdd_order [--files 200000]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.utils.work_shards import (
    prepare_length_bucketed_work_shards,
    prepare_work_shards,
    read_work_shard,
)

SEEK_MS = 8.0
SHORT_MS = 0.5
READAHEAD = 16


def build_layout(num_files: int, chunks_per_episode: int = 80, episodes_per_podcast: int = 20):
    """Return (paths_in_walk_order, physical_position_by_path, durations)."""
    rng = random.Random(20260611)
    paths: list[str] = []
    podcast = episode = 0
    while len(paths) < num_files:
        for _ in range(chunks_per_episode):
            if len(paths) >= num_files:
                break
            paths.append(
                f"/mnt/hdd/podcast_{podcast:05d}/episode_{episode:04d}/"
                f"chunk_{len(paths):08d}.wav"
            )
        episode += 1
        if episode % episodes_per_podcast == 0:
            podcast += 1
    position = {p: i for i, p in enumerate(paths)}
    durations = {p: round(rng.uniform(0.3, 15.5), 4) for p in paths}
    return paths, position, durations


def score_order(opened: list[str], position: dict[str, int]) -> dict[str, float]:
    seeks = short = 0
    prev = None
    for path in opened:
        pos = position[path]
        if prev is not None and 0 <= pos - prev <= READAHEAD:
            short += 1
        else:
            seeks += 1
        prev = pos
    est_min = (seeks * SEEK_MS + short * SHORT_MS) / 60_000.0
    return {"files": len(opened), "random_seeks": seeks, "short_seeks": short, "est_seek_minutes": est_min}


def replay_plan(work_dir: Path) -> list[str]:
    opened: list[str] = []
    for shard in sorted(work_dir.glob("shard_*.pending")):
        for line in read_work_shard(shard):
            opened.append(line.split("\t", 1)[0])
    return opened


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=int, default=200_000)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--workdir", default="cache/bench_hdd_order")
    args = parser.parse_args()

    paths, position, durations = build_layout(args.files)
    root = Path(args.workdir)
    root.mkdir(parents=True, exist_ok=True)

    scenarios = []

    for order in ("legacy", "path"):
        plan = prepare_length_bucketed_work_shards(
            root, f"bucketed_{order}", paths, durations,
            shard_size=args.shard_size, bucket_seconds=1.0, max_duration=15.0,
            order=order,
        )
        scenarios.append((f"bucketed GPU stage, order={order}", replay_plan(plan.work_dir)))

    for order in ("legacy", "path"):
        plan = prepare_work_shards(
            root, f"plain_{order}", paths, shard_size=args.shard_size, order=order,
        )
        scenarios.append((f"plain stage (walk-order input), order={order}", replay_plan(plan.work_dir)))

    # Resume case: CSV row order is no longer pure walk order after partial
    # absorbs/deletions; model a half-churned pending list.
    rng = random.Random(7)
    churned = paths[: args.files // 2]
    tail = paths[args.files // 2:]
    rng.shuffle(tail)
    churned = churned + tail
    for order in ("legacy", "path"):
        plan = prepare_work_shards(
            root, f"churned_{order}", churned, shard_size=args.shard_size, order=order,
        )
        scenarios.append((f"plain stage (half-churned input), order={order}", replay_plan(plan.work_dir)))

    print(f"\n{args.files} files, model: random seek {SEEK_MS} ms, "
          f"short seek {SHORT_MS} ms, readahead window {READAHEAD} files\n")
    print(f"{'scenario':52s} {'random seeks':>13s} {'short seeks':>12s} {'est seek time':>14s}")
    for name, opened in scenarios:
        s = score_order(opened, position)
        print(f"{name:52s} {s['random_seeks']:13d} {s['short_seeks']:12d} "
              f"{s['est_seek_minutes']:11.1f} min")


if __name__ == "__main__":
    main()
