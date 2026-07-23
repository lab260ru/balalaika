"""Crest factor (peak/RMS) filter with full audit accounting.

Per-file workflow:

* Compute the crest factor (linear ratio of peak to RMS amplitude).
* Append the result to ``balalaika.csv`` via the shared CSV manager.
* When ``crest_factor`` exceeds the configured threshold, the audio file is
  deleted; its row is removed from ``balalaika.csv`` on the final merge.
* The deleted file's duration is still recorded in the partial CSV so the
  rank-0 process can credit those hours to this stage in
  ``filter_summary.csv`` even after the audio is gone.

Resilience:

* Each worker streams rows to ``crest_part_<rank>.csv`` row-by-row through
  :class:`PartialCsvWriter` (``flush()`` after every row), so a forced stop
  preserves whatever rows were already produced.
* On startup any ``crest_part_*.csv`` left over from a previously interrupted
  run is **absorbed** into the main CSV before deciding what's still pending,
  i.e. re-running this stage *resumes* instead of re-scoring everything.
* Files already scored (``crest_factor`` populated in ``balalaika.csv``) are
  skipped automatically.

A per-stage rotating log file is initialised by :func:`setup_logging` so
operators can replay long batch runs offline.
"""

import argparse
import os
import time
from pathlib import Path
from typing import List, Set

import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm

from src.preprocess.audio_postprocessing import fused_audio_preprocessing_enabled
from src.utils.audit import record_stage_summary, safe_audio_duration
from src.utils.csv_manager import (
    PartialCsvWriter,
    PeriodicCsvMerger,
    absorb_partial_csvs,
    audit_from_filter_partials,
    discover_audio_paths,
    ensure_main_csv,
    load_csv_settings,
    resolve_path,
    unprocessed_paths,
)
from src.utils.datasets.preprocess import create_crest_factor_dataloader
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

PARTIAL_PREFIX = "crest"
COLUMN = "crest_factor"
PARTIAL_FIELDS = ("filepath", "crest_factor", "total_duration", "duration_s", "deleted")
VALUE_COLUMNS = [COLUMN, "total_duration"]


def calculate_crest_factors(waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = torch.arange(waveforms.shape[1], device=waveforms.device)[None, :] < lengths[:, None]
    masked = waveforms.masked_fill(~mask, 0.0)
    peak = masked.abs().amax(dim=1)
    power = masked.square().sum(dim=1) / lengths.clamp_min(1).to(dtype=waveforms.dtype)
    rms = power.sqrt()
    return torch.where(rms > 0, peak / rms, torch.full_like(rms, float("inf")))


def calculate_crest_factors_from_stats(
    peaks: torch.Tensor,
    sum_squares: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    lengths_f = lengths.clamp_min(1).to(dtype=sum_squares.dtype)
    rms = (sum_squares / lengths_f).sqrt()
    peaks = peaks.to(dtype=sum_squares.dtype)
    valid = (lengths > 0) & (rms > 0)
    return torch.where(valid, peaks / rms, torch.full_like(rms, float("inf")))


def _process_files(
    rank: int,
    files: List[str],
    writer: PartialCsvWriter,
    already_done: Set[str],
    crest_threshold: float,
    batch_size: int,
    loader_workers: int,
    prefetch_factor: int,
    processed_counter,
    skipped_counter,
    errors_counter,
) -> None:
    pending_files = []
    for path in files:
        resolved = resolve_path(path)
        if resolved in already_done:
            skipped_counter.value += 1
            continue
        pending_files.append(path)

    if not pending_files:
        return

    dataloader = create_crest_factor_dataloader(
        pending_files,
        batch_size=batch_size,
        num_workers=loader_workers,
        prefetch_factor=prefetch_factor,
    )
    prefetch_batches = loader_workers * prefetch_factor if loader_workers > 0 else 0
    logger.debug(
        f"perf dataloader_config stage=crest_factor rank={rank} "
        f"batch_size={batch_size} workers={loader_workers} "
        f"prefetch_factor={prefetch_factor} prefetch_batches={prefetch_batches} "
        f"items={len(pending_files)}"
    )

    batch_wait_started_at = time.perf_counter()
    for batch_idx, (paths, peaks, sum_squares, lengths, sample_rates, errors) in enumerate(tqdm(
        dataloader,
        desc=f"Worker-{rank}",
        position=rank,
    )):
        batch_received_at = time.perf_counter()
        logger.debug(
            f"perf dataloader_wait stage=crest_factor rank={rank} "
            f"batch={batch_idx} seconds={batch_received_at - batch_wait_started_at:.6f} "
            f"items={len(paths)}"
        )
        valid_indices = []
        for idx, (path_str, error) in enumerate(zip(paths, errors)):
            if error:
                errors_counter.value += 1
                logger.error(f"Error loading {path_str}: {error}")
            else:
                valid_indices.append(idx)
        if not valid_indices:
            batch_wait_started_at = time.perf_counter()
            continue

        valid = torch.tensor(valid_indices, dtype=torch.long)
        try:
            inference_started_at = time.perf_counter()
            batch_peaks = peaks.index_select(0, valid)
            batch_sum_squares = sum_squares.index_select(0, valid)
            batch_lengths = lengths.index_select(0, valid)
            batch_sample_rates = sample_rates.index_select(0, valid)
            crest_factors = calculate_crest_factors_from_stats(
                batch_peaks,
                batch_sum_squares,
                batch_lengths,
            ).tolist()
            durations = (
                batch_lengths.to(torch.float64) / batch_sample_rates.clamp_min(1).to(torch.float64)
            ).tolist()
            logger.debug(
                f"perf model=crest_factor event=batch_compute rank={rank} "
                f"batch={batch_idx} seconds={time.perf_counter() - inference_started_at:.6f} "
                f"items={len(valid_indices)}"
            )
        except Exception as exc:
            errors_counter.value += 1
            logger.error(f"Error processing batch on worker {rank}: {exc}")
            batch_wait_started_at = time.perf_counter()
            continue

        valid_paths = [paths[i] for i in valid_indices]
        for path_str, cf, duration_s in zip(valid_paths, crest_factors, durations):
            resolved = resolve_path(path_str)
            if resolved in already_done:
                skipped_counter.value += 1
                continue

            deleted = False
            if cf > crest_threshold:
                try:
                    os.remove(path_str)
                    deleted = True
                    logger.debug(f"Deleted {path_str} (crest_factor={cf:.2f})")
                except OSError as exc:
                    logger.error(f"Could not delete {path_str}: {exc}")

            write_started_at = time.perf_counter()
            writer.write(
                {
                    "filepath": resolved,
                    "crest_factor": round(cf, 4),
                    "total_duration": round(duration_s, 4),
                    "duration_s": round(duration_s, 4),
                    "deleted": deleted,
                }
            )
            logger.debug(
                f"perf partial_write stage=crest_factor rank={rank} "
                f"seconds={time.perf_counter() - write_started_at:.6f} path={resolved}"
            )
            already_done.add(resolved)
            processed_counter.value += 1
        batch_wait_started_at = time.perf_counter()


def run_worker(
    rank: int,
    world_size: int,
    work_dir: str,
    crest_threshold: float,
    output_dir: str,
    batch_size: int,
    loader_workers: int,
    prefetch_factor: int,
    processed_counter,
    skipped_counter,
    errors_counter,
):
    logger.info(
        f"Worker {rank}/{world_size} claiming work shards "
        f"(batch={batch_size}, loader_workers={loader_workers})"
    )

    claimed = 0
    with PartialCsvWriter(output_dir, PARTIAL_PREFIX, rank, fieldnames=PARTIAL_FIELDS) as writer:
        already_done: Set[str] = writer.already_done()
        if already_done:
            logger.info(
                f"Worker {rank}: {len(already_done)} files already scored in this partial; skipping repeats."
            )

        while True:
            shard_path = claim_work_shard(work_dir, rank)
            if shard_path is None:
                break
            shard_files = read_work_shard(shard_path)
            claimed += 1
            logger.info(f"Worker {rank}: processing {len(shard_files)} files from {shard_path.name}")
            _process_files(
                rank,
                shard_files,
                writer,
                already_done,
                crest_threshold,
                batch_size,
                loader_workers,
                prefetch_factor,
                processed_counter,
                skipped_counter,
                errors_counter,
            )
            mark_work_shard_done(shard_path)

    logger.info(f"Worker {rank} finished after {claimed} claimed shard(s).")

def main(args):
    setup_logging("crest_factor", log_dir=args.log_dir)

    config = load_config(args.config_path, "preprocess")
    fuse_audio = fused_audio_preprocessing_enabled(config)

    podcasts_path = config.get("podcasts_path")
    if not podcasts_path:
        podcasts_path = "../../../podcasts"
        logger.warning("Using default podcasts_path")
    podcasts_path = Path(podcasts_path)

    crest_threshold = config.get("crest_treshold", 10.0)
    num_workers = config.get("num_workers_crest_factor", 4)
    crest_batch_size = int(config.get("crest_factor_batch_size", 256))
    crest_loader_workers = int(config.get("crest_factor_loader_workers", 2))
    crest_prefetch_factor = int(config.get("crest_factor_prefetch_factor", 2))

    logger.info(
        f"Running crest factor removal: path={podcasts_path}, "
        f"threshold={crest_threshold}, workers={num_workers}"
    )

    audio_paths = discover_audio_paths(podcasts_path, config_path=args.config_path)
    if not audio_paths:
        logger.info("No audio files found for processing.")
        write_stage_status(
            stage=2,
            stage_name="crest_factor_remover",
            log_dir=args.log_dir or "./logs",
            processed=0,
            skipped=0,
            errors=0,
        )
        return

    logger.info(f"Found {len(audio_paths)} audio files.")

    # 1) Make sure balalaika.csv exists; bootstrap it from the audio tree if not.
    ensure_main_csv(podcasts_path, audio_paths=audio_paths)

    # 2) Pick up any leftover partials from a prior interrupted run before we
    #    decide what's still pending. Their rows (including deletions) are
    #    merged into balalaika.csv and the partials are removed.
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
            f"Absorbed {absorbed} rows from leftover {PARTIAL_PREFIX}_part_*.csv "
            "into balalaika.csv before scheduling new work."
        )

    # 3) Skip files that already have a crest_factor recorded in balalaika.csv.
    pending = unprocessed_paths(podcasts_path, COLUMN, audio_paths)
    if not pending:
        logger.success("All audio files already have a crest_factor entry. Skipping computation.")
        if not fuse_audio or absorbed:
            audit = audit_from_filter_partials(leftover_partials)
            if audit["files_in"] == 0:
                audit["files_in"] = len(audio_paths)
                audit["files_out"] = len(audio_paths)
            record_stage_summary(
                podcasts_path=podcasts_path,
                stage="crest_factor",
                files_in=audit["files_in"],
                files_out=audit["files_out"],
                hours_in=audit["hours_in"],
                hours_out=audit["hours_out"],
                params={"threshold": crest_threshold, "deleted": audit["files_deleted"]},
            )
        write_stage_status(
            stage=2,
            stage_name="crest_factor_remover",
            log_dir=args.log_dir or "./logs",
            processed=0,
            skipped=len(audio_paths),
            errors=0,
        )
        return

    shard_size = load_work_shard_size(args.config_path)
    work_plan = prepare_work_shards(
        podcasts_path,
        PARTIAL_PREFIX,
        pending,
        shard_size=shard_size,
    )
    pending_count = work_plan.total_items
    del pending

    logger.info(
        f"{pending_count} files still need a crest_factor; "
        f"starting workers over {work_plan.shard_count} shard(s)."
    )

    processed = mp.Value('i', 0)
    skipped = mp.Value('i', 0)
    errors = mp.Value('i', 0)

    csv_settings = load_csv_settings(args.config_path)

    try:
        with PeriodicCsvMerger(
            podcasts_path,
            prefix=PARTIAL_PREFIX,
            value_columns=VALUE_COLUMNS,
            drop_missing_files=True,
            preserve_existing=True,
            **csv_settings,
        ):
            if num_workers > 1:
                mp.spawn(
                    run_worker,
                    args=(
                        num_workers,
                        str(work_plan.work_dir),
                        crest_threshold,
                        str(podcasts_path),
                        crest_batch_size,
                        crest_loader_workers,
                        crest_prefetch_factor,
                        processed,
                        skipped,
                        errors,
                    ),
                    nprocs=num_workers,
                    join=True,
                )
            else:
                run_worker(
                    0,
                    1,
                    str(work_plan.work_dir),
                    crest_threshold,
                    str(podcasts_path),
                    crest_batch_size,
                    crest_loader_workers,
                    crest_prefetch_factor,
                    processed,
                    skipped,
                    errors,
                )
    except KeyboardInterrupt:
        logger.warning("Crest factor stage interrupted; merging whatever partials are on disk.")

    # 4) Merge whatever the workers managed to write (even after a Ctrl+C) into
    #    balalaika.csv, then drop the partials.
    new_partials, _ = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=VALUE_COLUMNS,
        drop_missing_files=True,
        preserve_existing=True,
    )

    combined = pd.concat(
        [df for df in (leftover_partials, new_partials) if df is not None and not df.empty],
        ignore_index=True,
    ) if (leftover_partials is not None or new_partials is not None) else pd.DataFrame()

    audit = audit_from_filter_partials(combined)

    if audit["files_in"] == 0 and audio_paths:
        # Fallback: workers wrote nothing (e.g. all read failures). Probe a
        # bounded sample and extrapolate so the report still has *some*
        # hours_in number — probing every file serially could take hours on
        # large datasets, for an audit-only estimate.
        sample_cap = 2000
        sample = audio_paths[:sample_cap]
        sampled_hours = sum(safe_audio_duration(p) for p in sample) / 3600.0
        fallback_hours = sampled_hours * (len(audio_paths) / max(1, len(sample)))
        if len(audio_paths) > sample_cap:
            logger.warning(
                f"Audit fallback: extrapolated hours_in from {sample_cap} of "
                f"{len(audio_paths)} files."
            )
        audit["files_in"] = len(audio_paths)
        audit["hours_in"] = fallback_hours
        audit["hours_out"] = fallback_hours

    record_stage_summary(
        podcasts_path=podcasts_path,
        stage="crest_factor",
        files_in=audit["files_in"],
        files_out=audit["files_out"],
        hours_in=audit["hours_in"],
        hours_out=audit["hours_out"],
        params={"threshold": crest_threshold, "deleted": audit["files_deleted"]},
    )

    write_stage_status(
        stage=2,
        stage_name="crest_factor_remover",
        log_dir=args.log_dir or "./logs",
        processed=processed.value,
        skipped=skipped.value,
        errors=errors.value,
    )

    logger.info("Crest factor check completed.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Remove audio files that exceed crest factor threshold (peak/rms > threshold)."
    )
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    args = parser.parse_args()

    main(args)
