"""DistillMOS-based quality filter for the balalaika pipeline.

Two-phase stage:
  1. Statistics — read balalaika.csv, show DistillMOS distribution, determine threshold.
  2. Deletion — parallel workers delete files below threshold, update CSV, record audit.

Modes:
  - Auto: threshold from config.yaml (separation.distillmos_filter.threshold).
  - Manual: interactive prompt (default when threshold is null, or forced via --manual).
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

from src.utils.audit import record_stage_summary, safe_audio_duration
from src.utils.csv_manager import (
    PartialCsvWriter,
    absorb_partial_csvs,
    audit_from_filter_partials,
    discover_audio_paths,
    ensure_main_csv,
    load_main_csv,
    resolve_path,
)
from src.utils.logging_setup import setup_logging
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config

PARTIAL_PREFIX = "distillmos_filter"
PARTIAL_FIELDS = ("filepath", "DistillMOS", "duration_s", "deleted")
COLUMN = "DistillMOS"


def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute distribution statistics for the DistillMOS column.

    Returns a dict with count, min, max, mean, median, std, and percentiles.
    """
    vals = df[COLUMN].dropna()
    if vals.empty:
        return {"count": 0}

    pct = [5, 10, 25, 50, 75, 90, 95]
    return {
        "count": int(len(vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "std": float(vals.std()),
        "percentiles": {p: float(vals.quantile(p / 100.0)) for p in pct},
    }


def print_distribution(stats: dict) -> None:
    """Print MOS distribution statistics and a 10-bin histogram."""
    if stats["count"] == 0:
        logger.warning("No DistillMOS values found in balalaika.csv")
        return

    print("\n" + "=" * 60)
    print("  DistillMOS Distribution")
    print("=" * 60)
    print(f"  Count:  {stats['count']:>10,}")
    print(f"  Min:    {stats['min']:>10.4f}")
    print(f"  Max:    {stats['max']:>10.4f}")
    print(f"  Mean:   {stats['mean']:>10.4f}")
    print(f"  Median: {stats['median']:>10.4f}")
    print(f"  Std:    {stats['std']:>10.4f}")
    print("-" * 60)
    print("  Percentiles:")
    for p, v in stats["percentiles"].items():
        print(f"    {p:>3}%:  {v:.4f}")
    print("=" * 60)


def print_histogram(df: pd.DataFrame, bins: int = 10) -> None:
    """Print a text-based histogram of DistillMOS values."""
    vals = df[COLUMN].dropna()
    if vals.empty:
        return

    lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        print(f"\n  All {len(vals)} values = {lo:.4f}")
        return

    bin_width = (hi - lo) / bins
    if "total_duration" in df.columns:
        durations = df["total_duration"].fillna(0)
    else:
        logger.warning("'total_duration' column not found in CSV — hour estimates will be 0")
        durations = pd.Series(0, index=df.index)

    print(f"\n  Histogram ({bins} bins, width={bin_width:.4f}):")
    print(f"  {'Range':>20}  {'Files':>10}  {'Hours':>10}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}")

    for i in range(bins):
        low = lo + i * bin_width
        high = lo + (i + 1) * bin_width
        if i == bins - 1:
            high = hi + 1e-9  # include max in last bin
        mask = (vals >= low) & (vals < high)
        n = int(mask.sum())
        h = float(durations[mask].sum() / 3600.0)
        bar_len = min(40, int(n / max(1, len(vals)) * 40))
        bar = "#" * bar_len
        print(f"  [{low:7.4f}, {high:7.4f})  {n:>10,}  {h:>10.2f}  {bar}")


def determine_threshold(
    config: dict, args: argparse.Namespace
) -> Optional[float]:
    """Determine the DistillMOS threshold from CLI args or config.

    Returns:
        float threshold, or None if manual mode was declined.
    """
    # CLI --threshold overrides everything (auto mode)
    if args.threshold is not None:
        t = float(args.threshold)
        if t <= 0:
            logger.error("Threshold must be positive.")
            return None
        logger.info(f"Auto mode: using CLI threshold = {t}")
        return t

    cfg = config.get("distillmos_filter", {})
    config_threshold = cfg.get("threshold")

    # Config provides a threshold and --manual not set → auto mode
    if config_threshold is not None and not args.manual:
        t = float(config_threshold)
        if t <= 0:
            logger.error("Threshold must be positive.")
            return None
        logger.info(f"Auto mode: using config threshold = {t}")
        return t

    # Manual (interactive) mode
    logger.info("Manual mode: interactive threshold selection")
    while True:
        try:
            raw = input("\nEnter DistillMOS threshold (files below this will be deleted): ").strip()
            if not raw:
                print("No threshold provided. Exiting.")
                return None
            t = float(raw)
            if t <= 0:
                print("Threshold must be positive. Exiting.")
                return None
            return t
        except ValueError:
            print("Invalid number. Try again or Ctrl+C to exit.")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return None


def print_preview(df: pd.DataFrame, threshold: float) -> tuple[int, float, int, float]:
    """Show how many files/hours would be deleted/saved at the given threshold.

    Returns (delete_count, delete_hours, save_count, save_hours).
    """
    vals = df[COLUMN].dropna()
    if "total_duration" in df.columns:
        durations = df["total_duration"].fillna(0)
    else:
        logger.warning("'total_duration' column not found in CSV — hour estimates will be 0")
        durations = pd.Series(0, index=df.index)

    mask = vals < threshold
    delete_count = int(mask.sum())
    save_count = len(vals) - delete_count
    delete_hours = float(durations[mask].sum() / 3600.0)
    save_hours = float(durations[~mask].sum() / 3600.0)

    print("\n" + "-" * 60)
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Files to DELETE (MOS < {threshold:.4f}): {delete_count:>10,}  ({delete_hours:.2f}h)")
    print(f"  Files to SAVE   (MOS >= {threshold:.4f}): {save_count:>10,}  ({save_hours:.2f}h)")
    print("-" * 60)

    return delete_count, delete_hours, save_count, save_hours


def run_worker(
    rank: int,
    my_paths: List[str],
    threshold: float,
    podcasts_path_str: str,
    processed_counter,
    deleted_counter,
    errors_counter,
) -> None:
    """Worker process: delete files with DistillMOS < threshold, write partial CSV."""
    podcasts_path = Path(podcasts_path_str)
    threshold_f = float(threshold)

    try:
        with PartialCsvWriter(
            podcasts_path, PARTIAL_PREFIX, rank, fieldnames=PARTIAL_FIELDS
        ) as writer:
            already_done = writer.already_done()
            if already_done:
                logger.info(
                    f"Worker {rank}: {len(already_done)} files already in partial; skipping."
                )

            df = load_main_csv(podcasts_path)
            df_indexed = df.set_index("filepath") if "filepath" in df.columns else None

            for path_str in my_paths:
                resolved = resolve_path(path_str)

                if resolved in already_done:
                    continue

                mos_val = None
                duration_s = 0.0

                if df_indexed is not None and resolved in df_indexed.index:
                    row = df_indexed.loc[resolved]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    mos_val = row.get(COLUMN)
                    if pd.notna(mos_val):
                        mos_val = float(mos_val)
                    dur = row.get("total_duration")
                    if pd.notna(dur):
                        duration_s = float(dur)

                if mos_val is None or pd.isna(mos_val):
                    continue

                deleted = False
                if mos_val < threshold_f:
                    try:
                        os.remove(path_str)
                        deleted = True
                        deleted_counter.value += 1
                    except OSError as exc:
                        logger.warning(f"Could not delete {path_str}: {exc}")
                        errors_counter.value += 1

                if duration_s <= 0:
                    try:
                        duration_s = safe_audio_duration(path_str)
                    except Exception:
                        duration_s = 0.0

                writer.write({
                    "filepath": resolved,
                    COLUMN: mos_val,
                    "duration_s": round(duration_s, 4),
                    "deleted": deleted,
                })
                processed_counter.value += 1

        logger.info(f"Worker {rank} done.")
    except Exception as exc:
        logger.exception(f"Worker {rank} error: {exc}")
        errors_counter.value += 1


def run_deletion_workers(
    podcasts_path: Path,
    threshold: float,
    num_workers: int,
) -> tuple:
    """Spawn workers to delete files below threshold in parallel.

    Returns (processed, deleted, errors) counts.
    """
    from multiprocessing import Process, Value

    audio_paths = discover_audio_paths(podcasts_path)
    if not audio_paths:
        logger.warning("No audio files found.")
        return 0, 0, 0

    # Split paths into shards
    shards = []
    for i in range(num_workers):
        shard = audio_paths[i::num_workers]
        if shard:
            shards.append(shard)

    if not shards:
        return 0, 0, 0

    logger.info(f"Deletion phase: {len(audio_paths)} files, {num_workers} workers, threshold={threshold}")

    processed = Value("i", 0)
    deleted = Value("i", 0)
    errors = Value("i", 0)

    procs = []
    for rank, shard in enumerate(shards):
        p = Process(
            target=run_worker,
            args=(rank, shard, threshold, str(podcasts_path), processed, deleted, errors),
        )
        p.start()
        procs.append(p)

    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        logger.warning("DistillMOS filter interrupted; waiting for workers to finish...")
        for p in procs:
            p.join()

    return processed.value, deleted.value, errors.value


def main(args):
    setup_logging("distillmos_filter", log_dir=args.log_dir)
    config = load_config(args.config_path, "separation")
    podcasts_path = config.get("podcasts_path")

    if not podcasts_path:
        logger.error("No podcasts_path in config")
        return

    podcasts_path = Path(podcasts_path)

    # --- Phase 1: Statistics & threshold ---
    df = load_main_csv(podcasts_path)

    if COLUMN not in df.columns or df[COLUMN].dropna().empty:
        logger.error(
            f"No '{COLUMN}' column in balalaika.csv. Run stage 5 (DistillMOS) first."
        )
        write_stage_status(
            stage=5.5,
            stage_name="distillmos_filter",
            log_dir=args.log_dir or "./logs",
            processed=0,
            skipped=0,
            errors=1,
        )
        return

    stats = compute_statistics(df)
    print_distribution(stats)
    print_histogram(df)

    threshold = determine_threshold(config, args)
    if threshold is None:
        write_stage_status(
            stage=5.5,
            stage_name="distillmos_filter",
            log_dir=args.log_dir or "./logs",
            processed=0,
            skipped=0,
            errors=0,
        )
        return

    delete_count, delete_hours, save_count, save_hours = print_preview(df, threshold)

    if delete_count == 0:
        logger.info("0 files would be deleted at this threshold. Nothing to do.")
        write_stage_status(
            stage=5.5,
            stage_name="distillmos_filter",
            log_dir=args.log_dir or "./logs",
            processed=0,
            skipped=0,
            errors=0,
        )
        return

    # Manual mode: ask for confirmation
    cfg = config.get("distillmos_filter", {})
    config_threshold = cfg.get("threshold")
    is_auto = (args.threshold is not None) or (config_threshold is not None and not args.manual)

    if not is_auto:
        try:
            resp = input("\nProceed with deletion? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return
        if resp not in ("y", "yes"):
            print("Deletion cancelled.")
            return

    # --- Phase 2: Deletion ---
    num_workers = cfg.get("num_workers", 4)

    # Bootstrap CSV + absorb leftover partials
    audio_paths = discover_audio_paths(podcasts_path)
    ensure_main_csv(podcasts_path, audio_paths=audio_paths)

    leftover_partials, absorbed = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=[COLUMN],
        drop_missing_files=True,
        bootstrap_audio_paths=audio_paths,
    )
    if absorbed:
        logger.info(
            f"Absorbed {absorbed} rows from leftover {PARTIAL_PREFIX}_part_*.csv"
        )

    processed, deleted, errors = run_deletion_workers(
        podcasts_path, threshold, num_workers
    )

    # Merge partials + audit
    new_partials_df, _ = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=[COLUMN],
        drop_missing_files=True,
    )

    combined = pd.concat(
        [df for df in (leftover_partials, new_partials_df) if df is not None and not df.empty],
        ignore_index=True,
    ) if (leftover_partials is not None or new_partials_df is not None) else pd.DataFrame()

    audit = audit_from_filter_partials(combined)

    record_stage_summary(
        podcasts_path=podcasts_path,
        stage="distillmos_filter",
        files_in=audit["files_in"],
        files_out=audit["files_out"],
        hours_in=audit["hours_in"],
        hours_out=audit["hours_out"],
        params={"threshold": threshold, "deleted": audit["files_deleted"]},
    )

    write_stage_status(
        stage=5.5,
        stage_name="distillmos_filter",
        log_dir=args.log_dir or "./logs",
        processed=processed,
        skipped=0,
        errors=errors,
    )

    logger.success(
        f"DistillMOS filter complete. Processed: {processed}, "
        f"Deleted: {deleted}, Errors: {errors}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DistillMOS-based quality filter for balalaika pipeline"
    )
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Force interactive mode even if threshold is set in config",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override config threshold (forces auto mode)",
    )
    main(parser.parse_args())
