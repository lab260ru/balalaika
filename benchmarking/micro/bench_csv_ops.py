"""Micro-benchmarks for csv_manager / path-discovery hot paths.

Run BEFORE and AFTER an optimization with identical fixtures:

    python -m benchmarking.micro.make_fixtures            # once
    python -m benchmarking.micro.bench_csv_ops --label before
    ... apply patch ...
    python -m benchmarking.micro.bench_csv_ops --label after

Each benchmark works on a throwaway copy of the fixture state so runs are
independent and repeatable. Results are appended to
``benchmarking/reports/micro/csv_ops.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")


def timed(fn, repeats: int):
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return times, result


def fresh_state(fixture_root: Path, work_root: Path) -> Path:
    """Copy the fixture state dir to a scratch dir (fast, same fs)."""
    dst = work_root / "state"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(fixture_root / "state", dst)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path, default=Path("cache/bench_fixtures"))
    ap.add_argument("--label", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="comma-separated benchmark names to run (default: all)",
    )
    args = ap.parse_args()

    from src.utils import csv_manager as cm
    from src.utils.utils import get_audio_paths

    fixture_root = args.fixtures.resolve()
    work_root = fixture_root / "scratch"
    work_root.mkdir(parents=True, exist_ok=True)
    out_path = REPO_ROOT / "benchmarking" / "reports" / "micro" / "csv_ops.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    def record(name: str, times, extra=None):
        results[name] = {
            "avg_s": statistics.mean(times),
            "min_s": min(times),
            "max_s": max(times),
            "times": times,
            **(extra or {}),
        }
        print(f"{name:34s} avg={statistics.mean(times):8.3f}s min={min(times):8.3f}s")

    def want(name: str) -> bool:
        return not only or name in only

    # --- 1. load_main_csv: full read + filepath normalization ---------------
    if want("load_main_csv"):
        state = fresh_state(fixture_root, work_root)
        times, df = timed(lambda: cm.load_main_csv(state), args.repeats)
        record("load_main_csv", times, {"rows": len(df)})

    # --- 2. atomic_write_csv: backup + chunked write + rename ---------------
    if want("atomic_write_csv"):
        state = fresh_state(fixture_root, work_root)
        df = cm.load_main_csv(state)
        # First write creates .bak from existing file: that's the steady state.
        times, _ = timed(lambda: cm.atomic_write_csv(df, state / "balalaika.csv"), args.repeats)
        record("atomic_write_csv", times, {"rows": len(df)})

    # --- 3. read_partial_csvs ------------------------------------------------
    if want("read_partial_csvs"):
        state = fresh_state(fixture_root, work_root)
        times, parts = timed(lambda: cm.read_partial_csvs(state, "crest"), args.repeats)
        record("read_partial_csvs", times, {"rows": len(parts)})

    # --- 4. periodic-flush path: read partials + upsert into main CSV -------
    if want("flush_cycle"):
        state = fresh_state(fixture_root, work_root)

        def flush_once():
            parts = cm.read_partial_csvs(state, "crest")
            cm.upsert_columns(
                state,
                parts,
                value_columns=["crest_factor", "total_duration"],
                preserve_existing=True,
            )

        times, _ = timed(flush_once, args.repeats)
        record("flush_cycle", times)

    # --- 5. upsert with drop_missing_files (stat storm) ---------------------
    if want("upsert_drop_missing"):
        state = fresh_state(fixture_root, work_root)
        parts = cm.read_partial_csvs(state, "crest")

        def upsert_drop():
            cm.upsert_columns(
                state,
                parts,
                value_columns=["crest_factor", "total_duration"],
                drop_missing_files=True,
                preserve_existing=True,
            )

        times, _ = timed(upsert_drop, 1)  # destructive (drops rows); once
        record("upsert_drop_missing", times)

    # --- 6. unprocessed_paths ------------------------------------------------
    if want("unprocessed_paths"):
        state = fresh_state(fixture_root, work_root)
        audio_paths = cm._audio_paths_from_csv(state)

        def unproc():
            return cm.unprocessed_paths(state, "crest_factor", audio_paths)

        times, pending = timed(unproc, args.repeats)
        record("unprocessed_paths", times, {"pending": len(pending), "universe": len(audio_paths)})

    # --- 7. _count_partial_rows ----------------------------------------------
    if want("count_partial_rows"):
        state = fresh_state(fixture_root, work_root)
        times, n = timed(lambda: cm._count_partial_rows(state, "crest"), max(args.repeats, 5))
        record("count_partial_rows", times, {"rows": n})

    # --- 8. get_audio_paths (rglob) on the fixture tree ----------------------
    if want("get_audio_paths"):
        tree = fixture_root / "tree"
        times, paths = timed(lambda: get_audio_paths(str(tree)), args.repeats)
        record("get_audio_paths", times, {"files": len(paths)})

    # --- 9. discover_audio_paths from CSV ------------------------------------
    if want("discover_from_csv"):
        state = fresh_state(fixture_root, work_root)
        times, paths = timed(lambda: cm._audio_paths_from_csv(state), args.repeats)
        record("discover_from_csv", times, {"files": len(paths)})

    entry = {
        "label": args.label,
        "at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
