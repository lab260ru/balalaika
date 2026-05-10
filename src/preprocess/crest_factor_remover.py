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
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm

from src.utils.audit import record_stage_summary, safe_audio_duration
from src.utils.csv_manager import (
    PartialCsvWriter,
    absorb_partial_csvs,
    audit_from_filter_partials,
    discover_audio_paths,
    ensure_main_csv,
    resolve_path,
    unprocessed_paths,
)
from src.utils.logging_setup import setup_logging
from src.utils.utils import load_config, load_audio

PARTIAL_PREFIX = "crest"
PARTIAL_FIELDS = ("filepath", "crest_factor", "duration_s", "deleted")
COLUMN = "crest_factor"


def calculate_crest_factor(audio: np.ndarray) -> float:
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return float("inf")
    return peak / rms


def run_worker(
    rank: int,
    world_size: int,
    all_file_paths: List[str],
    crest_threshold: float,
    output_dir: str,
):
    my_files = all_file_paths[rank::world_size]
    if not my_files:
        return

    logger.info(f"Worker {rank}/{world_size} processing {len(my_files)} files")

    with PartialCsvWriter(output_dir, PARTIAL_PREFIX, rank, fieldnames=PARTIAL_FIELDS) as writer:
        already_done: Set[str] = writer.already_done()
        if already_done:
            logger.info(
                f"Worker {rank}: {len(already_done)} files already scored in this partial; skipping."
            )

        for path_str in tqdm(my_files, desc=f"Worker-{rank}", position=rank):
            resolved = resolve_path(path_str)
            if resolved in already_done:
                continue

            try:
                audio_tensor, sr = load_audio(path_str)
                if audio_tensor.shape[0] > 1:
                    audio = audio_tensor.mean(dim=0).numpy()
                else:
                    audio = audio_tensor.squeeze(0).numpy()
                cf = calculate_crest_factor(audio)
                duration_s = float(audio.shape[-1]) / float(sr) if sr else 0.0
            except Exception as exc:
                logger.error(f"Error processing {path_str}: {exc}")
                continue
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            deleted = False
            if cf > crest_threshold:
                try:
                    os.remove(path_str)
                    deleted = True
                    logger.debug(f"Deleted {path_str} (crest_factor={cf:.2f})")
                except OSError as exc:
                    logger.error(f"Could not delete {path_str}: {exc}")

            writer.write(
                {
                    "filepath": resolved,
                    "crest_factor": round(cf, 4),
                    "duration_s": round(duration_s, 4),
                    "deleted": deleted,
                }
            )

    logger.info(f"Worker {rank} finished its shard.")


def main(args):
    setup_logging("crest_factor", log_dir=args.log_dir)

    config = load_config(args.config_path, "preprocess")

    podcasts_path = config.get("podcasts_path")
    if not podcasts_path:
        podcasts_path = "../../../podcasts"
        logger.warning("Using default podcasts_path")
    podcasts_path = Path(podcasts_path)

    crest_threshold = config.get("crest_treshold", 10.0)
    num_workers = config.get("num_workers_crest_factor", 4)

    logger.info(
        f"Running crest factor removal: path={podcasts_path}, "
        f"threshold={crest_threshold}, workers={num_workers}"
    )

    audio_paths = discover_audio_paths(podcasts_path)
    if not audio_paths:
        logger.info("No audio files found for processing.")
        return

    logger.info(f"Found {len(audio_paths)} audio files on disk.")

    # 1) Make sure balalaika.csv exists; bootstrap it from the audio tree if not.
    ensure_main_csv(podcasts_path, audio_paths=audio_paths)

    # 2) Pick up any leftover partials from a prior interrupted run before we
    #    decide what's still pending. Their rows (including deletions) are
    #    merged into balalaika.csv and the partials are removed.
    leftover_partials, absorbed = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=[COLUMN],
        drop_missing_files=True,
        bootstrap_audio_paths=audio_paths,
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
        # Still emit an audit row reflecting the state of the dataset.
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
        return

    logger.info(f"{len(pending)} files still need a crest_factor; starting workers.")

    try:
        if num_workers > 1:
            mp.spawn(
                run_worker,
                args=(num_workers, pending, crest_threshold, str(podcasts_path)),
                nprocs=num_workers,
                join=True,
            )
        else:
            run_worker(0, 1, pending, crest_threshold, str(podcasts_path))
    except KeyboardInterrupt:
        logger.warning("Crest factor stage interrupted; merging whatever partials are on disk.")

    # 4) Merge whatever the workers managed to write (even after a Ctrl+C) into
    #    balalaika.csv, then drop the partials.
    new_partials, _ = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=[COLUMN],
        drop_missing_files=True,
    )

    combined = pd.concat(
        [df for df in (leftover_partials, new_partials) if df is not None and not df.empty],
        ignore_index=True,
    ) if (leftover_partials is not None or new_partials is not None) else pd.DataFrame()

    audit = audit_from_filter_partials(combined)

    if audit["files_in"] == 0 and audio_paths:
        # Fallback: workers wrote nothing (e.g. all read failures).
        # Probe a few files so the report still has *some* hours_in number.
        fallback_hours = sum(safe_audio_duration(p) for p in audio_paths) / 3600.0
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
