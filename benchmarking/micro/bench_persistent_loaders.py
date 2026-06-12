"""Micro-benchmark: shard-boundary loader overhead in stage 7 (transcription).

Each GPU worker claims work shards in a loop. The old flow built a fresh
DataLoader (spawning ``num_workers`` loader processes + refilling the prefetch
buffer) for every claimed shard; the persistent loader keeps one set of workers
alive across all shards. This bench isolates that boundary cost: it splits a
pool of real wavs into many small shards and drives BOTH loader strategies
through a trivial CPU "model" (so the timing is dominated by decode + the
per-shard worker spawn/warmup, exactly what the change targets).

It also asserts both strategies emit the identical batch sequence (paths +
waveform bytes), so the speedup never comes at the cost of changed batches.

    python -m benchmarking.micro.bench_persistent_loaders --label check
    python -m benchmarking.micro.bench_persistent_loaders --label check \
        --shards 10 --shard-size 8 --workers 4
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

from src.utils.datasets.transcription import (  # noqa: E402
    PersistentTranscriptionLoader,
    create_transcription_dataloader,
)

SAMPLE_RATE = 16_000


def _gather_shards(n_shards: int, shard_size: int) -> list[list[str]]:
    pool = sorted((REPO_ROOT / "cache/bench_sample/audio").rglob("*.wav"))
    if not pool:
        raise SystemExit("no bench wavs under cache/bench_sample/audio")
    need = n_shards * shard_size
    files = [str(pool[i % len(pool)]) for i in range(need)]
    return [files[k * shard_size:(k + 1) * shard_size] for k in range(n_shards)]


def _consume_per_shard(shards, *, batch_size, workers) -> list:
    """Old flow: a fresh DataLoader per shard."""
    seq = []
    for files in shards:
        dl = create_transcription_dataloader(
            files, sample_rate=SAMPLE_RATE, batch_size=batch_size,
            num_workers=workers, prefetch_factor=2,
        )
        for paths, waveforms, lengths, _errs in dl:
            # Trivial "model": touch the data so decode is actually realized.
            _ = float(waveforms.sum()) if waveforms.numel() else 0.0
            seq.append((tuple(paths), int(lengths.sum())))
        del dl
    return seq


def _consume_persistent(shards, *, batch_size, workers) -> list:
    """New flow: one persistent loader across all shards."""
    seq = []
    with PersistentTranscriptionLoader(
        sample_rate=SAMPLE_RATE, batch_size=batch_size,
        num_workers=workers, prefetch_factor=2,
    ) as loader:
        for files in shards:
            for paths, waveforms, lengths, _errs in loader.iter_shard(files):
                _ = float(waveforms.sum()) if waveforms.numel() else 0.0
                seq.append((tuple(paths), int(lengths.sum())))
    return seq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--shards", type=int, default=10)
    ap.add_argument("--shard-size", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    shards = _gather_shards(args.shards, args.shard_size)
    n_files = sum(len(s) for s in shards)
    print(f"{args.shards} shards x {args.shard_size} files = {n_files}; "
          f"batch={args.batch_size} workers={args.workers}")

    # Equivalence: identical batch sequence (paths + summed lengths).
    seq_old = _consume_per_shard(shards, batch_size=args.batch_size, workers=args.workers)
    seq_new = _consume_persistent(shards, batch_size=args.batch_size, workers=args.workers)
    identical = seq_old == seq_new
    print(f"batch-sequence identical: {identical} ({len(seq_old)} batches)")

    timings = {"per_shard": [], "persistent": []}
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        _consume_per_shard(shards, batch_size=args.batch_size, workers=args.workers)
        t1 = time.perf_counter()
        _consume_persistent(shards, batch_size=args.batch_size, workers=args.workers)
        t2 = time.perf_counter()
        timings["per_shard"].append(t1 - t0)
        timings["persistent"].append(t2 - t1)
        print(f"per_shard {t1 - t0:.3f}s   persistent {t2 - t1:.3f}s")

    old_best = min(timings["per_shard"])
    new_best = min(timings["persistent"])
    speedup = old_best / new_best if new_best else float("nan")
    per_boundary_ms = (old_best - new_best) / max(1, args.shards) * 1000
    print(f"best: per_shard {old_best:.3f}s  persistent {new_best:.3f}s  "
          f"speedup {speedup:.2f}x  (~{per_boundary_ms:.0f} ms saved / shard boundary)")

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "shards": args.shards,
        "shard_size": args.shard_size,
        "files": n_files,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "batch_sequence_identical": identical,
        "per_shard_best_s": old_best,
        "persistent_best_s": new_best,
        "per_shard_mean_s": statistics.mean(timings["per_shard"]),
        "persistent_mean_s": statistics.mean(timings["persistent"]),
        "speedup": speedup,
        "ms_saved_per_boundary": per_boundary_ms,
    }
    out_path = REPO_ROOT / "benchmarking" / "reports" / "micro" / "persistent_loaders.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
