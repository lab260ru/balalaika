"""Micro-benchmarks for csv_manager / path-discovery hot paths.

Run BEFORE and AFTER an optimization with identical fixtures:

    python -m benchmarking.micro.make_fixtures            # once
    python -m benchmarking.micro.bench_csv_ops --label before
    ... apply patch ...
    python -m benchmarking.micro.bench_csv_ops --label after

Each benchmark works on a throwaway copy of the fixture state so runs are
independent and repeatable. Results are appended to
``benchmarking/reports/micro/csv_ops.jsonl``.

Peak-RSS columns (load_main_csv / upsert_rss / unprocessed_paths /
duration_cache) sample ``/proc/self/statm`` in a background thread, so the
peak is the whole-process resident set during the op — directly comparable
across runs on the same node.

State-format comparison (csv vs parquet pipeline state, report §9.5):

    python -m benchmarking.micro.bench_csv_ops --label csv     --state-format csv
    python -m benchmarking.micro.bench_csv_ops --label parquet --state-format parquet

``--state-format parquet`` migrates the fixture CSV to ``balalaika.parquet``
once up front (so per-op timings measure the parquet steady state, not the
one-time migration), then runs every state op against parquet.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import threading
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


def _rss_mb() -> float:
    """Current resident set size in MB (Linux /proc; 0.0 if unavailable)."""
    try:
        with open(f"/proc/{os.getpid()}/statm") as f:
            pages = int(f.read().split()[1])
        return pages * (os.sysconf("SC_PAGE_SIZE") / (1024 * 1024))
    except Exception:
        return 0.0


class _RssSampler:
    """Background sampler tracking peak RSS (MB) over a code block."""

    def __init__(self, interval: float = 0.01) -> None:
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, _rss_mb())

    def __enter__(self) -> "_RssSampler":
        self.peak = _rss_mb()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak = max(self.peak, _rss_mb())


def timed_rss(fn, repeats: int):
    """Like ``timed`` but also returns the max peak-RSS (MB) seen across runs."""
    times = []
    result = None
    peak = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        with _RssSampler() as s:
            result = fn()
        times.append(time.perf_counter() - t0)
        peak = max(peak, s.peak)
    return times, result, peak


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
    ap.add_argument(
        "--state-format",
        choices=["csv", "parquet"],
        default="csv",
        help="state format for the per-op benchmarks (BALALAIKA_STATE_FORMAT).",
    )
    args = ap.parse_args()

    if args.state_format == "parquet":
        os.environ["BALALAIKA_STATE_FORMAT"] = "parquet"
    else:
        os.environ.pop("BALALAIKA_STATE_FORMAT", None)

    from src.utils import audio_durations as ad
    from src.utils import csv_manager as cm
    from src.utils.utils import get_audio_paths

    def _materialize_state(state: Path) -> None:
        """In parquet mode, migrate the fixture CSV to parquet up front so the
        per-op timings measure the steady state (parquet reads/writes), not the
        one-time migration."""
        if args.state_format == "parquet":
            cm.load_main_csv(state)  # triggers CSV->parquet migration

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
            "state_format": args.state_format,
            **(extra or {}),
        }
        peak = (extra or {}).get("peak_rss_mb")
        peak_str = f" peak_rss={peak:7.0f}MB" if peak else ""
        print(
            f"{name:30s}[{args.state_format:7s}] "
            f"avg={statistics.mean(times):8.3f}s min={min(times):8.3f}s{peak_str}"
        )

    def want(name: str) -> bool:
        return not only or name in only

    # --- 1. load_main_csv: full read + filepath normalization ---------------
    if want("load_main_csv"):
        state = fresh_state(fixture_root, work_root)
        _materialize_state(state)
        times, df, peak = timed_rss(lambda: cm.load_main_csv(state), args.repeats)
        record("load_main_csv", times, {"rows": len(df), "peak_rss_mb": peak})

    # --- 2. atomic_write_csv: backup + chunked write + rename ---------------
    if want("atomic_write_csv"):
        state = fresh_state(fixture_root, work_root)
        _materialize_state(state)
        df = cm.load_main_csv(state)
        # First write creates .bak from existing file: that's the steady state.
        target = cm.state_path(state)
        times, _ = timed(lambda: cm.atomic_write_csv(df, target), args.repeats)
        record("atomic_write_csv", times, {"rows": len(df)})

    # --- 3. read_partial_csvs ------------------------------------------------
    if want("read_partial_csvs"):
        state = fresh_state(fixture_root, work_root)
        times, parts = timed(lambda: cm.read_partial_csvs(state, "crest"), args.repeats)
        record("read_partial_csvs", times, {"rows": len(parts)})

    # --- 4. periodic-flush path: read partials + upsert into main CSV -------
    if want("flush_cycle"):
        state = fresh_state(fixture_root, work_root)
        _materialize_state(state)

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

    # --- 4b. upsert peak RAM (item 2: copy elimination) ---------------------
    if want("upsert_rss"):
        state = fresh_state(fixture_root, work_root)
        _materialize_state(state)
        parts = cm.read_partial_csvs(state, "crest")

        def upsert_once():
            cm.upsert_columns(
                state,
                parts,
                value_columns=["crest_factor", "total_duration"],
                preserve_existing=True,
            )

        times, _, peak = timed_rss(upsert_once, args.repeats)
        record("upsert_rss", times, {"peak_rss_mb": peak})

    # --- 5. upsert with drop_missing_files (stat storm) ---------------------
    if want("upsert_drop_missing"):
        state = fresh_state(fixture_root, work_root)
        _materialize_state(state)
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
        _materialize_state(state)
        audio_paths = cm._audio_paths_from_csv(state)

        def unproc():
            return cm.unprocessed_paths(state, "crest_factor", audio_paths)

        times, pending, peak = timed_rss(unproc, args.repeats)
        record(
            "unprocessed_paths",
            times,
            {"pending": len(pending), "universe": len(audio_paths), "peak_rss_mb": peak},
        )

    # --- 6b. duration cache (item 3: narrow read + vectorized build) --------
    if want("duration_cache"):
        state = fresh_state(fixture_root, work_root)
        _materialize_state(state)
        audio_paths = cm._audio_paths_from_csv(state)
        requested = set(audio_paths)

        times, cache, peak = timed_rss(
            lambda: ad._csv_duration_cache(state, requested), args.repeats
        )
        record(
            "duration_cache",
            times,
            {"cached": len(cache), "universe": len(requested), "peak_rss_mb": peak},
        )

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
        _materialize_state(state)
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
