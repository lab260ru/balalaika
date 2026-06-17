"""Filter audio using raw TTS-suitability class logits from balalaika.csv.

The filter computes ``not_tts_margin = not_tts_score - tts_score`` and deletes
files whose margin is greater than the selected threshold (i.e. the model leans
toward "not suitable for TTS"). A threshold of zero implements the model's raw
argmax decision (equivalently ``p_tts < 0.5``) without applying softmax.
"""

import argparse
import math
import os
from multiprocessing import Process, Value
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.utils.audit import record_stage_summary, safe_audio_duration
from src.utils.csv_manager import (
    PartialCsvWriter,
    absorb_partial_csvs,
    ensure_main_csv,
    load_main_csv,
    resolve_path,
)
from src.utils.logging_setup import setup_logging
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config
from src.utils.work_shards import (
    claim_work_shard,
    load_work_shard_size,
    mark_work_shard_done,
    prepare_work_shards,
    read_work_shard,
)

PARTIAL_PREFIX = "tts_suit_filter"
NOT_TTS_SCORE_COLUMN = "not_tts_score"
TTS_SCORE_COLUMN = "tts_score"
MARGIN_COLUMN = "not_tts_margin"
PARTIAL_FIELDS = ("filepath", "duration_s", "deleted")


def scored_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = (NOT_TTS_SCORE_COLUMN, TTS_SCORE_COLUMN)
    if any(column not in df.columns for column in required):
        return pd.DataFrame(columns=[*df.columns, MARGIN_COLUMN])

    out = df.dropna(subset=list(required)).copy()
    for column in required:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=list(required))
    out[MARGIN_COLUMN] = out[NOT_TTS_SCORE_COLUMN] - out[TTS_SCORE_COLUMN]
    return out


def compute_statistics(df: pd.DataFrame) -> dict:
    values = df[MARGIN_COLUMN].dropna()
    if values.empty:
        return {"count": 0}
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()),
        "percentiles": {
            percentile: float(values.quantile(percentile / 100.0))
            for percentile in percentiles
        },
    }


def print_distribution(stats: dict) -> None:
    if stats["count"] == 0:
        logger.warning("No complete TTS-suitability score pairs found.")
        return
    print("\n" + "=" * 60)
    print("  TTS-Suitability not_tts Margin Distribution")
    print("  margin = not_tts_score - tts_score")
    print("=" * 60)
    print(f"  Count:  {stats['count']:>10,}")
    print(f"  Min:    {stats['min']:>10.6f}")
    print(f"  Max:    {stats['max']:>10.6f}")
    print(f"  Mean:   {stats['mean']:>10.6f}")
    print(f"  Median: {stats['median']:>10.6f}")
    print(f"  Std:    {stats['std']:>10.6f}")
    print("-" * 60)
    for percentile, value in stats["percentiles"].items():
        print(f"  {percentile:>3}%: {value:.6f}")
    print("=" * 60)


def print_histogram(df: pd.DataFrame, bins: int = 10) -> None:
    values = df[MARGIN_COLUMN].dropna()
    if values.empty:
        return
    low = float(values.min())
    high = float(values.max())
    if low == high:
        print(f"\n  All {len(values)} margins = {low:.6f}")
        return

    width = (high - low) / bins
    print(f"\n  Histogram ({bins} bins):")
    print(f"  {'Margin range':>24}  {'Files':>10}")
    for index in range(bins):
        lower = low + index * width
        upper = high + 1e-9 if index == bins - 1 else low + (index + 1) * width
        count = int(((values >= lower) & (values < upper)).sum())
        print(f"  [{lower:9.5f}, {upper:9.5f})  {count:>10,}")


def determine_threshold(config: dict, args: argparse.Namespace) -> Optional[float]:
    if args.threshold is not None:
        threshold = float(args.threshold)
        logger.info(f"Auto mode: using CLI not_tts-margin threshold={threshold}")
        return threshold

    cfg = config.get("tts_suitability_filter", {})
    configured = cfg.get("threshold")
    if configured is not None and not args.manual:
        threshold = float(configured)
        logger.info(f"Auto mode: using config not_tts-margin threshold={threshold}")
        return threshold

    logger.info("Manual mode: interactive not_tts-margin threshold selection")
    while True:
        try:
            raw = input(
                "\nEnter not_tts-margin threshold "
                "(delete when not_tts_score - tts_score > threshold): "
            ).strip()
            if not raw:
                return None
            return float(raw)
        except ValueError:
            print("Invalid number. Try again or Ctrl+C to exit.")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return None


def deletion_candidates(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return df[df[MARGIN_COLUMN] > float(threshold)].copy()


def print_preview(
    df: pd.DataFrame, threshold: float
) -> tuple[int, float, int, float]:
    delete_df = deletion_candidates(df, threshold)
    delete_count = int(len(delete_df))
    save_count = int(len(df) - delete_count)
    durations = (
        pd.to_numeric(df["total_duration"], errors="coerce").fillna(0.0)
        if "total_duration" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    delete_hours = float(durations.loc[delete_df.index].sum() / 3600.0)
    save_hours = float(durations.sum() / 3600.0 - delete_hours)

    print("\n" + "-" * 72)
    print(f"  not_tts-margin threshold: {threshold:.6f}")
    print(
        f"  DELETE margin > {threshold:.6f}: "
        f"{delete_count:>10,} files ({delete_hours:.2f}h)"
    )
    print(
        f"  SAVE   margin <= {threshold:.6f}: "
        f"{save_count:>10,} files ({save_hours:.2f}h)"
    )
    print("-" * 72)
    return delete_count, delete_hours, save_count, save_hours


def run_worker(
    rank: int,
    work_dir: str,
    podcasts_path_str: str,
    processed_counter,
    deleted_counter,
    errors_counter,
) -> None:
    podcasts_path = Path(podcasts_path_str)
    try:
        with PartialCsvWriter(
            podcasts_path, PARTIAL_PREFIX, rank, fieldnames=PARTIAL_FIELDS
        ) as writer:
            already_done = writer.already_done()
            while True:
                shard_path = claim_work_shard(work_dir, rank)
                if shard_path is None:
                    break
                for path_str in read_work_shard(shard_path):
                    resolved = resolve_path(path_str)
                    if resolved in already_done:
                        continue
                    duration_s = safe_audio_duration(path_str)
                    deleted = False
                    try:
                        os.remove(path_str)
                        deleted = True
                        deleted_counter.value += 1
                    except FileNotFoundError:
                        deleted = True
                    except OSError as exc:
                        logger.warning(f"Could not delete {path_str}: {exc}")
                        errors_counter.value += 1
                    writer.write(
                        {
                            "filepath": resolved,
                            "duration_s": round(duration_s, 4),
                            "deleted": deleted,
                        }
                    )
                    already_done.add(resolved)
                    processed_counter.value += 1
                mark_work_shard_done(shard_path)
    except Exception as exc:
        logger.exception(f"TTS-suitability filter worker {rank} failed: {exc}")
        errors_counter.value += 1


def run_deletion_workers(
    podcasts_path: Path,
    work_dir: Path,
    num_workers: int,
    shard_count: int,
) -> tuple[int, int, int]:
    processed = Value("i", 0)
    deleted = Value("i", 0)
    errors = Value("i", 0)
    worker_count = min(max(1, int(num_workers)), max(1, int(shard_count)))
    processes = []

    for rank in range(worker_count):
        process = Process(
            target=run_worker,
            args=(
                rank,
                str(work_dir),
                str(podcasts_path),
                processed,
                deleted,
                errors,
            ),
        )
        process.start()
        processes.append(process)

    try:
        for process in processes:
            process.join()
            if process.exitcode not in (0, None):
                errors.value += 1
    except KeyboardInterrupt:
        logger.warning("TTS-suitability filter interrupted; terminating workers.")
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        raise

    return processed.value, deleted.value, errors.value


def write_status(
    args: argparse.Namespace, *, processed: int, skipped: int, errors: int
) -> None:
    write_stage_status(
        stage=7.5,
        stage_name="tts_suitability_filter",
        log_dir=args.log_dir or "./logs",
        processed=processed,
        skipped=skipped,
        errors=errors,
    )


def main(args):
    setup_logging("tts_suitability_filter", log_dir=args.log_dir)
    config = load_config(args.config_path, "separation")
    podcasts_path_value = config.get("podcasts_path")
    if not podcasts_path_value:
        logger.error("No podcasts_path in config.")
        write_status(args, processed=0, skipped=0, errors=1)
        return

    podcasts_path = Path(podcasts_path_value)
    ensure_main_csv(podcasts_path)
    baseline = scored_rows(load_main_csv(podcasts_path))
    if baseline.empty:
        logger.error(
            "No complete not_tts_score/tts_score pairs in balalaika.csv. "
            "Run stage 7 first."
        )
        write_status(args, processed=0, skipped=0, errors=1)
        return

    leftover_partials, absorbed = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=[],
        drop_missing_files=True,
        preserve_existing=True,
    )
    if absorbed:
        logger.info(f"Absorbed {absorbed} leftover TTS-suitability filter rows.")

    current = scored_rows(load_main_csv(podcasts_path))
    print_distribution(compute_statistics(current))
    print_histogram(current)
    threshold = determine_threshold(config, args)
    if threshold is None:
        write_status(args, processed=0, skipped=0, errors=0)
        return

    candidates = deletion_candidates(current, threshold)
    print_preview(current, threshold)
    if candidates.empty:
        logger.info("No files exceed the not_tts-margin threshold.")
        write_status(args, processed=0, skipped=len(current), errors=0)
        return

    cfg = config.get("tts_suitability_filter", {})
    configured = cfg.get("threshold")
    is_auto = args.threshold is not None or (
        configured is not None and not args.manual
    )
    if not is_auto:
        try:
            response = input("\nProceed with deletion? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            write_status(args, processed=0, skipped=0, errors=0)
            return
        if response not in {"y", "yes"}:
            print("Deletion cancelled.")
            write_status(args, processed=0, skipped=0, errors=0)
            return

    delete_workers = max(1, int(cfg.get("num_workers", 4)))
    configured_shard_size = load_work_shard_size(args.config_path)
    parallel_shard_size = max(1, math.ceil(len(candidates) / delete_workers))
    work_plan = prepare_work_shards(
        podcasts_path,
        PARTIAL_PREFIX,
        candidates["filepath"].astype(str).tolist(),
        shard_size=min(configured_shard_size, parallel_shard_size),
    )
    try:
        processed, deleted, errors = run_deletion_workers(
            podcasts_path,
            work_plan.work_dir,
            delete_workers,
            work_plan.shard_count,
        )
    except KeyboardInterrupt:
        processed, deleted, errors = 0, 0, 0

    new_partials, _ = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=[],
        drop_missing_files=True,
        preserve_existing=True,
    )
    frames = [
        frame
        for frame in (leftover_partials, new_partials)
        if frame is not None and not frame.empty
    ]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    deleted_paths = set()
    if not combined.empty and "deleted" in combined.columns:
        deleted_mask = combined["deleted"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
        deleted_paths = set(
            combined.loc[deleted_mask, "filepath"].astype(str).tolist()
        )

    baseline_durations = (
        pd.to_numeric(baseline["total_duration"], errors="coerce").fillna(0.0)
        if "total_duration" in baseline.columns
        else pd.Series(0.0, index=baseline.index)
    )
    baseline_deleted = baseline["filepath"].astype(str).isin(deleted_paths)
    files_in = int(len(baseline))
    files_deleted = int(baseline_deleted.sum())
    hours_in = float(baseline_durations.sum() / 3600.0)
    hours_deleted = float(baseline_durations[baseline_deleted].sum() / 3600.0)
    record_stage_summary(
        podcasts_path=podcasts_path,
        stage="tts_suitability_filter",
        files_in=files_in,
        files_out=files_in - files_deleted,
        hours_in=hours_in,
        hours_out=hours_in - hours_deleted,
        params={
            "score": "not_tts_score-tts_score",
            "threshold": threshold,
            "deleted": files_deleted,
        },
    )
    write_status(
        args,
        processed=processed,
        skipped=max(0, len(current) - processed),
        errors=errors,
    )
    logger.success(
        f"TTS-suitability filter complete. Processed: {processed}, "
        f"Deleted: {deleted}, Errors: {errors}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter audio using raw TTS-suitability not_tts/tts logits"
    )
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Force interactive mode even if threshold is configured",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override not_tts-margin threshold; delete when margin is greater",
    )
    main(parser.parse_args())
