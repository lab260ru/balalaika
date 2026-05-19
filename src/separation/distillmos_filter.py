"""DistillMOS-based quality filter for the balalaika pipeline.

Two-phase stage:
  1. Statistics — read balalaika.csv, show DistillMOS distribution, determine threshold.
  2. Deletion — parallel workers delete files below threshold, update CSV, record audit.

Modes:
  - Auto: threshold from config.yaml (separation.distillmos_filter.threshold).
  - Manual: interactive prompt (default when threshold is null, or forced via --manual).
"""

import argparse
import math
import os
import sys
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
    durations = df["total_duration"].fillna(0) if "total_duration" in df.columns else pd.Series(0, index=df.index)

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
        Raises SystemExit if user declines in manual mode.
    """
    # CLI --threshold overrides everything (auto mode)
    if args.threshold is not None:
        logger.info(f"Auto mode: using CLI threshold = {args.threshold}")
        return float(args.threshold)

    cfg = config.get("distillmos_filter", {})
    config_threshold = cfg.get("threshold")

    # Config provides a threshold and --manual not set → auto mode
    if config_threshold is not None and not args.manual:
        logger.info(f"Auto mode: using config threshold = {config_threshold}")
        return float(config_threshold)

    # Manual (interactive) mode
    logger.info("Manual mode: interactive threshold selection")
    while True:
        try:
            raw = input("\nEnter DistillMOS threshold (files below this will be deleted): ").strip()
            if not raw:
                print("No threshold provided. Exiting.")
                sys.exit(0)
            t = float(raw)
            if t <= 0:
                print("Threshold must be positive. Exiting.")
                sys.exit(0)
            return t
        except ValueError:
            print("Invalid number. Try again or Ctrl+C to exit.")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            sys.exit(0)


def print_preview(df: pd.DataFrame, threshold: float) -> tuple:
    """Show how many files/hours would be deleted/saved at the given threshold.

    Returns (delete_count, delete_hours, save_count, save_hours).
    """
    vals = df[COLUMN].dropna()
    durations = df["total_duration"].fillna(0) if "total_duration" in df.columns else pd.Series(0, index=df.index)

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
