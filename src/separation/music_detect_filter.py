"""Music-probability filter for the balalaika pipeline (stage 4.5).

The companion of :mod:`src.separation.music_detect`: stage 4 scores each chunk's
``music_prob`` into ``balalaika.parquet`` and this stage deletes the files whose
probability of being music is **above** a threshold. It is the structural twin
of :mod:`src.separation.distillmos_filter` (stage 5.5); the only difference is
the predicate direction — DistillMOS deletes *below* its threshold (low quality),
music detection deletes *above* its threshold (likely music).

Two-phase stage:
  1. Statistics — read balalaika.parquet, show the music_prob distribution,
     determine the threshold.
  2. Deletion — parallel workers delete files above the threshold, update state,
     record the audit row consumed by stage 15 (``src.report``).

Modes:
  - Auto: threshold from config.yaml (``separation.music_detect_filter.threshold``).
  - Manual: interactive prompt (default when threshold is null, or ``--manual``).

When stage 4 ran with ``music_detect.inline_filter: true`` the files were already
removed in the scoring pass, so this stage simply finds no candidates and exits.
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
    discover_audio_paths,
    ensure_main_csv,
    load_main_csv,
    resolve_path,
)
from src.utils.logging_setup import setup_logging
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config

PARTIAL_PREFIX = "music_filter"
COLUMN = "music_prob"
PARTIAL_FIELDS = ("filepath", "music_prob", "total_duration", "duration_s", "deleted")
VALUE_COLUMNS = [COLUMN, "total_duration"]


def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute distribution statistics for the music_prob column.

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
    """Print music_prob distribution statistics."""
    if stats["count"] == 0:
        logger.warning("No music_prob values found in balalaika.parquet")
        return

    print("\n" + "=" * 60)
    print("  Music Probability Distribution")
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
    """Print a text-based histogram of music_prob values."""
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
        logger.warning("'total_duration' column not found — hour estimates will be 0")
        durations = pd.Series(0, index=df.index)
    durations = durations.reindex(vals.index).fillna(0)

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


def determine_threshold(config: dict, args: argparse.Namespace) -> Optional[float]:
    """Determine the music_prob threshold from CLI args or config.

    Returns the float threshold, or ``None`` if manual mode was declined.
    """
    if args.threshold is not None:
        t = float(args.threshold)
        logger.info(f"Auto mode: using CLI music_prob threshold = {t}")
        return t

    cfg = config.get("music_detect_filter", {})
    config_threshold = cfg.get("threshold")

    if config_threshold is not None and not args.manual:
        t = float(config_threshold)
        logger.info(f"Auto mode: using config music_prob threshold = {t}")
        return t

    logger.info("Manual mode: interactive threshold selection")
    while True:
        try:
            raw = input(
                "\nEnter music_prob threshold (files ABOVE this will be deleted): "
            ).strip()
            if not raw:
                print("No threshold provided. Exiting.")
                return None
            return float(raw)
        except ValueError:
            print("Invalid number. Try again or Ctrl+C to exit.")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return None


def deletion_candidates(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Rows whose music_prob is present and strictly **above** ``threshold``.

    These are the only files the deletion phase needs to touch (the music side
    of the candidate-only sharding used by :mod:`src.separation.distillmos_filter`).
    """
    if COLUMN not in df.columns:
        return df.iloc[0:0].copy()
    prob = pd.to_numeric(df[COLUMN], errors="coerce")
    return df[prob.notna() & (prob > float(threshold))].copy()


def preview_counts(df: pd.DataFrame, threshold: float) -> tuple[int, float, int, float]:
    """Delete/save counts and hours at ``threshold`` from the dataframe alone.

    Returns ``(delete_count, delete_hours, save_count, save_hours)`` computed
    purely from ``music_prob`` and the authoritative ``total_duration`` column.
    """
    scored = df[df[COLUMN].notna()]
    if "total_duration" in df.columns:
        durations = pd.to_numeric(df["total_duration"], errors="coerce").fillna(0.0)
    else:
        logger.warning("'total_duration' column not found — hour estimates will be 0")
        durations = pd.Series(0.0, index=df.index)

    delete_mask = pd.to_numeric(scored[COLUMN], errors="coerce") > float(threshold)
    delete_idx = scored.index[delete_mask]
    save_idx = scored.index[~delete_mask]

    delete_count = int(len(delete_idx))
    save_count = int(len(save_idx))
    delete_hours = float(durations.loc[delete_idx].sum() / 3600.0)
    save_hours = float(durations.loc[save_idx].sum() / 3600.0)
    return delete_count, delete_hours, save_count, save_hours


def print_preview(df: pd.DataFrame, threshold: float) -> tuple[int, float, int, float]:
    """Show how many files/hours would be deleted/saved at the given threshold."""
    delete_count, delete_hours, save_count, save_hours = preview_counts(df, threshold)

    print("\n" + "-" * 60)
    print(f"  Threshold: {threshold:.4f}")
    print(
        f"  Files to DELETE (prob > {threshold:.4f}): {delete_count:>10,}  ({delete_hours:.2f}h)"
    )
    print(
        f"  Files to SAVE   (prob <= {threshold:.4f}): {save_count:>10,}  ({save_hours:.2f}h)"
    )
    print("-" * 60)

    return delete_count, delete_hours, save_count, save_hours


def run_worker(
    rank: int,
    my_items: List[tuple],
    threshold: float,
    podcasts_path_str: str,
    processed_counter,
    deleted_counter,
    errors_counter,
) -> None:
    """Worker process: delete deletion-candidate files, write a partial CSV.

    ``my_items`` is a list of ``(path, music_prob, duration_s)`` tuples prepared
    by the parent from ONE read of balalaika.parquet, already filtered to
    deletion candidates (``music_prob`` > threshold), so the worker only touches
    files it might delete. Mirrors :func:`src.separation.distillmos_filter.run_worker`.
    """
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

            for path_str, prob_val, duration_s in my_items:
                resolved = resolve_path(path_str)

                if resolved in already_done:
                    continue

                if prob_val is None or pd.isna(prob_val):
                    continue
                prob_val = float(prob_val)

                # Only candidates reach a worker, but keep the guard so a stale
                # shard can never delete a now-below-threshold file.
                if not (prob_val > threshold_f):
                    continue

                # Probe BEFORE deletion so a candidate that lacked a stored
                # duration still records one.
                if duration_s <= 0:
                    try:
                        duration_s = safe_audio_duration(path_str)
                    except Exception:
                        duration_s = 0.0

                deleted = False
                try:
                    os.remove(path_str)
                    deleted = True
                    deleted_counter.value += 1
                except FileNotFoundError:
                    deleted = True
                    deleted_counter.value += 1
                except OSError as exc:
                    logger.warning(f"Could not delete {path_str}: {exc}")
                    errors_counter.value += 1

                writer.write(
                    {
                        "filepath": resolved,
                        COLUMN: prob_val,
                        "total_duration": round(duration_s, 4),
                        "duration_s": round(duration_s, 4),
                        "deleted": deleted,
                    }
                )
                already_done.add(resolved)
                processed_counter.value += 1

        logger.info(f"Worker {rank} done.")
    except Exception as exc:
        logger.exception(f"Worker {rank} error: {exc}")
        errors_counter.value += 1


def run_deletion_workers(
    podcasts_path: Path,
    threshold: float,
    num_workers: int,
    config_path: Optional[str] = None,
) -> tuple:
    """Spawn workers to delete files above threshold in parallel.

    Only deletion candidates (``music_prob`` > threshold) are sharded to
    workers; kept files are accounted for by the caller straight from the
    dataframe. Returns (processed, deleted, errors) counts.
    """
    from multiprocessing import Process, Value

    df = load_main_csv(podcasts_path)
    candidates = deletion_candidates(df, threshold)
    if candidates.empty or "filepath" not in candidates.columns:
        logger.info("No deletion candidates above threshold.")
        return 0, 0, 0

    if "total_duration" in candidates.columns:
        durs = pd.to_numeric(candidates["total_duration"], errors="coerce").fillna(0.0)
    else:
        durs = pd.Series(0.0, index=candidates.index)
    prob_vals = pd.to_numeric(candidates[COLUMN], errors="coerce")
    items = list(
        zip(
            candidates["filepath"].astype(str),
            prob_vals.astype(float),
            durs.astype(float),
        )
    )
    candidate_count = len(items)
    scored_count = int(pd.to_numeric(df[COLUMN], errors="coerce").notna().sum())
    del df, candidates, prob_vals, durs

    shards = []
    for i in range(num_workers):
        shard = items[i::num_workers]
        if shard:
            shards.append(shard)

    if not shards:
        logger.info("No scored files to filter.")
        return 0, 0, 0

    logger.info(
        f"Deletion phase: {candidate_count} candidates "
        f"(of {scored_count} scored), "
        f"{num_workers} workers, threshold={threshold}"
    )

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
        logger.warning("Music filter interrupted; waiting for workers to finish...")
        for p in procs:
            p.join()

    return processed.value, deleted.value, errors.value


def main(args):
    setup_logging("music_detect_filter", log_dir=args.log_dir)
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
            f"No '{COLUMN}' column in balalaika.parquet. Run stage 4 (music detection) first."
        )
        write_stage_status(
            stage=4.5,
            stage_name="music_detect_filter",
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
            stage=4.5,
            stage_name="music_detect_filter",
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
            stage=4.5,
            stage_name="music_detect_filter",
            log_dir=args.log_dir or "./logs",
            processed=0,
            skipped=0,
            errors=0,
        )
        return

    cfg = config.get("music_detect_filter", {})
    config_threshold = cfg.get("threshold")
    is_auto = (args.threshold is not None) or (
        config_threshold is not None and not args.manual
    )

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

    audio_paths = discover_audio_paths(podcasts_path, config_path=args.config_path)
    ensure_main_csv(podcasts_path, audio_paths=audio_paths)

    leftover_partials, absorbed = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=VALUE_COLUMNS,
        drop_missing_files=True,
        bootstrap_audio_paths=audio_paths,
        preserve_existing=True,
    )
    if absorbed:
        logger.info(
            f"Absorbed {absorbed} rows from leftover {PARTIAL_PREFIX}_part_*.csv"
        )

    baseline = load_main_csv(podcasts_path)
    base_delete_count, base_delete_hours, base_save_count, base_save_hours = (
        preview_counts(baseline, threshold)
    )
    files_in = base_delete_count + base_save_count

    processed, deleted, errors = run_deletion_workers(
        podcasts_path, threshold, num_workers, args.config_path
    )

    new_partials_df, _ = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=VALUE_COLUMNS,
        drop_missing_files=True,
        preserve_existing=True,
    )

    combined = (
        pd.concat(
            [d for d in (leftover_partials, new_partials_df) if d is not None and not d.empty],
            ignore_index=True,
        )
        if (leftover_partials is not None or new_partials_df is not None)
        else pd.DataFrame()
    )

    # files_deleted / hours_deleted come straight from the candidate partials'
    # own rows (their ``deleted`` flag and probed ``duration_s``), matching the
    # distillmos_filter audit treatment exactly.
    files_deleted = 0
    hours_deleted = 0.0
    if not combined.empty and "deleted" in combined.columns:
        deleted_mask = combined["deleted"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        ) | (combined["deleted"] == True)  # noqa: E712
        files_deleted = int(deleted_mask.sum())
        if "duration_s" in combined.columns:
            deleted_durations = pd.to_numeric(
                combined.loc[deleted_mask, "duration_s"], errors="coerce"
            ).fillna(0.0)
            hours_deleted = float(deleted_durations.sum() / 3600.0)

    hours_in = base_save_hours + hours_deleted

    record_stage_summary(
        podcasts_path=podcasts_path,
        stage="music_detect_filter",
        files_in=files_in,
        files_out=files_in - files_deleted,
        hours_in=hours_in,
        hours_out=hours_in - hours_deleted,
        params={"threshold": threshold, "deleted": files_deleted},
    )

    write_stage_status(
        stage=4.5,
        stage_name="music_detect_filter",
        log_dir=args.log_dir or "./logs",
        processed=processed,
        skipped=0,
        errors=errors,
    )

    logger.success(
        f"Music filter complete. Processed: {processed}, "
        f"Deleted: {deleted}, Errors: {errors}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Music-probability filter for balalaika pipeline"
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
