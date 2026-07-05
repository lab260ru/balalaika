"""Stage 3.5 — tail-clipping signal scoring (score-only, no deletion).

Backfills the two SPEC_drop_criteria.md signals to ``balalaika.parquet`` for
chunks that were **not** cut by the smart stage-1 path (pre-existing datasets,
``existing_chunks`` mode, or trees produced before the snap/pad change):

* ``tail_db`` — loudest 20 ms frame in the last 80 ms, dB relative to the
  utterance's robust peak. ``> -12`` = clipped tail, ``> -6`` = severe clip.
* ``trailing_silence_ms`` — margin after the last voiced frame. ``< 40`` means
  no acoustic end-of-sentence cue.

This stage only *scores* — deciding a drop threshold needs the score
distribution first (run this, inspect the columns, then filter). Filtering by
these columns can be added later exactly like the other score/filter pairs.

Orchestration is a verbatim sibling of ``crest_factor_remover`` minus the
deletion/audit paths: disk-backed work shards, per-worker partial CSVs flushed
row-by-row (Ctrl+C safe), a periodic merger daemon, and resume by presence of
the ``tail_db`` column.
"""

import argparse
import time
from pathlib import Path
from typing import List, Set

import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm

from src.utils.csv_manager import (
    PartialCsvWriter,
    PeriodicCsvMerger,
    absorb_partial_csvs,
    discover_audio_paths,
    ensure_main_csv,
    load_csv_settings,
    resolve_path,
    unprocessed_paths,
)
from src.utils.datasets.preprocess import create_tail_signal_dataloader
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

STAGE_ID = 3.5
PARTIAL_PREFIX = "tail_score"
COLUMN = "tail_db"
PARTIAL_FIELDS = ("filepath", "tail_db", "trailing_silence_ms", "total_duration")
VALUE_COLUMNS = ["tail_db", "trailing_silence_ms", "total_duration"]


def _process_files(
    rank: int,
    files: List[str],
    writer: PartialCsvWriter,
    already_done: Set[str],
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

    dataloader = create_tail_signal_dataloader(
        pending_files,
        batch_size=batch_size,
        num_workers=loader_workers,
        prefetch_factor=prefetch_factor,
    )
    logger.debug(
        f"perf dataloader_config stage=tail_score rank={rank} "
        f"batch_size={batch_size} workers={loader_workers} "
        f"prefetch_factor={prefetch_factor} items={len(pending_files)}"
    )

    batch_wait_started_at = time.perf_counter()
    for batch_idx, (paths, tail_dbs, trailing_mss, durations, load_errors) in enumerate(
        tqdm(dataloader, desc=f"Worker-{rank}", position=rank)
    ):
        logger.debug(
            f"perf dataloader_wait stage=tail_score rank={rank} "
            f"batch={batch_idx} seconds={time.perf_counter() - batch_wait_started_at:.6f} "
            f"items={len(paths)}"
        )
        for path_str, tail_db, trailing_ms, duration_s, error in zip(
            paths, tail_dbs, trailing_mss, durations, load_errors
        ):
            if error:
                errors_counter.value += 1
                logger.error(f"Error loading {path_str}: {error}")
                continue
            resolved = resolve_path(path_str)
            if resolved in already_done:
                skipped_counter.value += 1
                continue
            writer.write(
                {
                    "filepath": resolved,
                    "tail_db": round(tail_db, 4),
                    "trailing_silence_ms": round(trailing_ms, 2),
                    "total_duration": round(duration_s, 4),
                }
            )
            already_done.add(resolved)
            processed_counter.value += 1
        batch_wait_started_at = time.perf_counter()


def run_worker(
    rank: int,
    world_size: int,
    work_dir: str,
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
    with PartialCsvWriter(
        output_dir, PARTIAL_PREFIX, rank, fieldnames=PARTIAL_FIELDS
    ) as writer:
        already_done: Set[str] = writer.already_done()
        if already_done:
            logger.info(
                f"Worker {rank}: {len(already_done)} files already scored in "
                "this partial; skipping repeats."
            )

        while True:
            shard_path = claim_work_shard(work_dir, rank)
            if shard_path is None:
                break
            shard_files = read_work_shard(shard_path)
            claimed += 1
            logger.info(
                f"Worker {rank}: processing {len(shard_files)} files from {shard_path.name}"
            )
            _process_files(
                rank,
                shard_files,
                writer,
                already_done,
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
    setup_logging("tail_score", log_dir=args.log_dir)

    config = load_config(args.config_path, "preprocess")

    podcasts_path = Path(config.get("podcasts_path", "../../../podcasts"))
    num_workers = int(config.get("tail_score_workers", 4))
    batch_size = int(config.get("tail_score_batch_size", 256))
    loader_workers = int(config.get("tail_score_loader_workers", 2))
    prefetch_factor = int(config.get("tail_score_prefetch_factor", 2))

    logger.info(
        f"Running tail-signal scoring: path={podcasts_path}, workers={num_workers}"
    )

    audio_paths = discover_audio_paths(podcasts_path, config_path=args.config_path)
    if not audio_paths:
        logger.info("No audio files found for processing.")
        write_stage_status(
            stage=STAGE_ID,
            stage_name=PARTIAL_PREFIX,
            log_dir=args.log_dir or "./logs",
            processed=0,
            skipped=0,
            errors=0,
        )
        return

    logger.info(f"Found {len(audio_paths)} audio files.")

    ensure_main_csv(podcasts_path, audio_paths=audio_paths)

    # Absorb leftovers from a previously interrupted run before deciding what
    # is still pending, so a rerun resumes instead of re-scoring.
    _, absorbed = absorb_partial_csvs(
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
            "before scheduling new work."
        )

    pending = unprocessed_paths(podcasts_path, COLUMN, audio_paths)
    if not pending:
        logger.success("All audio files already have a tail_db entry. Nothing to do.")
        write_stage_status(
            stage=STAGE_ID,
            stage_name=PARTIAL_PREFIX,
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
        f"{pending_count} files still need tail signals; "
        f"starting workers over {work_plan.shard_count} shard(s)."
    )

    processed = mp.Value("i", 0)
    skipped = mp.Value("i", 0)
    errors = mp.Value("i", 0)

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
                        str(podcasts_path),
                        batch_size,
                        loader_workers,
                        prefetch_factor,
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
                    str(podcasts_path),
                    batch_size,
                    loader_workers,
                    prefetch_factor,
                    processed,
                    skipped,
                    errors,
                )
    except KeyboardInterrupt:
        logger.warning(
            "Tail-score stage interrupted; merging whatever partials are on disk."
        )

    absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=VALUE_COLUMNS,
        drop_missing_files=True,
        preserve_existing=True,
    )

    write_stage_status(
        stage=STAGE_ID,
        stage_name=PARTIAL_PREFIX,
        log_dir=args.log_dir or "./logs",
        processed=processed.value,
        skipped=skipped.value,
        errors=errors.value,
    )

    logger.info("Tail-signal scoring completed.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Score tail-clipping signals (tail_db, trailing_silence_ms) into balalaika.parquet."
    )
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument(
        "--log_dir", type=str, default=None, help="Override log directory"
    )
    args = parser.parse_args()

    main(args)
