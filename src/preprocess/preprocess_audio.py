"""ITU-R BS.1770-4 loudness normalization.

The original implementation re-encoded everything through classic
``torchaudio.save``, which silently degraded MP3 files and made the FLAC → ``.mp3`` mismatch
operators reported. This rewrite:

* Reads samples through a torch DataLoader backed by ``torchaudio``.
* Writes samples back with ``torchaudio.save_with_torchcodec``.
* Sets up a per-stage log file via :func:`setup_logging`.

CSV resilience:

* The shared :mod:`src.utils.csv_manager` handles bootstrapping
  ``balalaika.csv`` (creating it from the audio tree when absent) and atomic
  rewrites.
* Each worker streams a single row per file to its ``loudness_part_<rank>.csv``
  via :class:`PartialCsvWriter` (``flush()`` after every row), so a forced
  stop preserves whatever rows were already produced.
* On startup any leftover partial from a previously interrupted run is
  absorbed into the main CSV before scheduling new work — re-running this
  stage simply *resumes* normalization.
"""

import argparse
import time
from pathlib import Path
from typing import List, Set

import numpy as np
import torch
import torch.multiprocessing as mp
import torchaudio
from loguru import logger
from tqdm import tqdm

from src.preprocess.audio_postprocessing import normalize_audio_loudness
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
from src.utils.datasets.preprocess import create_loudness_normalize_dataloader
from src.utils.gpu import apply_torch_perf_defaults
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

apply_torch_perf_defaults()


NORMALIZED_COLUMN = "loudness_normalized"
PARTIAL_PREFIX = "loudness"
PARTIAL_FIELDS = ("filepath", NORMALIZED_COLUMN)
def _write_audio(audio_path: str, samples: np.ndarray, sample_rate: int) -> None:
    """Write samples shaped ``(channels, frames)`` using TorchCodec."""
    array = samples if samples.ndim == 2 else samples[np.newaxis, :]
    tensor = torch.as_tensor(array, dtype=torch.float32)
    save_started_at = time.perf_counter()
    torchaudio.save_with_torchcodec(audio_path, tensor, sample_rate)
    logger.debug(
        f"perf audio_save stage=loudness path={audio_path} "
        f"seconds={time.perf_counter() - save_started_at:.6f} "
        f"sample_rate={int(sample_rate)} frames={int(tensor.shape[-1])}"
    )


def process_audio_file(
    audio_path: str,
    audio: torch.Tensor,
    sample_rate: int,
    peak: float,
    loudness: float,
    block_size: float,
) -> bool:
    """Normalize loudness for a single file in-place."""
    try:
        audio_np = audio.numpy()

        if audio_np.shape[0] == 1:
            mono = audio_np.squeeze(0)
            normalized = normalize_audio_loudness(
                mono, sample_rate, peak=peak, loudness=loudness, block_size=block_size
            )
            normalized_2d = normalized[np.newaxis, :]
        else:
            multi = audio_np.T  # (frames, channels)
            normalized = normalize_audio_loudness(
                multi, sample_rate, peak=peak, loudness=loudness, block_size=block_size
            )
            normalized_2d = normalized.T

        _write_audio(audio_path, normalized_2d, sample_rate)

        return True
    except Exception as exc:
        logger.error(f"Error processing {audio_path}: {exc}")
        return False


def _process_files(
    rank: int,
    files: List[str],
    writer: PartialCsvWriter,
    already_done: Set[str],
    peak: float,
    loudness: float,
    block_size: float,
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

    dataloader = create_loudness_normalize_dataloader(
        pending_files,
        batch_size=batch_size,
        num_workers=loader_workers,
        prefetch_factor=prefetch_factor,
    )
    prefetch_batches = loader_workers * prefetch_factor if loader_workers > 0 else 0
    logger.debug(
        f"perf dataloader_config stage=loudness rank={rank} "
        f"batch_size={batch_size} workers={loader_workers} "
        f"prefetch_factor={prefetch_factor} prefetch_batches={prefetch_batches} "
        f"items={len(pending_files)}"
    )

    batch_wait_started_at = time.perf_counter()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Worker-{rank}", position=rank)):
        batch_received_at = time.perf_counter()
        logger.debug(
            f"perf dataloader_wait stage=loudness rank={rank} "
            f"batch={batch_idx} seconds={batch_received_at - batch_wait_started_at:.6f} "
            f"items={len(batch)}"
        )
        for file_path, audio, sample_rate, error in batch:
            if error:
                logger.error(f"Error loading {file_path}: {error}")
                errors_counter.value += 1
                continue
            resolved = resolve_path(file_path)
            if resolved in already_done:
                skipped_counter.value += 1
                continue
            inference_started_at = time.perf_counter()
            ok = process_audio_file(str(file_path), audio, sample_rate, peak, loudness, block_size)
            logger.debug(
                f"perf model=loudness_normalize event=process_file rank={rank} "
                f"seconds={time.perf_counter() - inference_started_at:.6f} "
                f"path={file_path} sample_rate={int(sample_rate)} frames={int(audio.shape[-1])}"
            )
            if ok:
                write_started_at = time.perf_counter()
                writer.write({"filepath": resolved, NORMALIZED_COLUMN: True})
                logger.debug(
                    f"perf partial_write stage=loudness rank={rank} "
                    f"seconds={time.perf_counter() - write_started_at:.6f} path={resolved}"
                )
                already_done.add(resolved)
                processed_counter.value += 1
        batch_wait_started_at = time.perf_counter()


def run_worker(
    rank: int,
    world_size: int,
    work_dir: str,
    peak: float,
    loudness: float,
    block_size: float,
    output_dir: str,
    batch_size: int,
    loader_workers: int,
    prefetch_factor: int,
    processed_counter,
    skipped_counter,
    errors_counter,
):
    """Worker that claims work shards from disk."""
    logger.info(
        f"Worker {rank}/{world_size} claiming work shards "
        f"(batch={batch_size}, loader_workers={loader_workers})"
    )

    claimed = 0
    with PartialCsvWriter(output_dir, PARTIAL_PREFIX, rank, fieldnames=PARTIAL_FIELDS) as writer:
        already_done: Set[str] = writer.already_done()
        if already_done:
            logger.info(
                f"Worker {rank}: {len(already_done)} files already in this partial; skipping repeats."
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
                peak,
                loudness,
                block_size,
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
    setup_logging("preprocess_audio", log_dir=args.log_dir)

    config = load_config(args.config_path, "preprocess")

    podcasts_path = config.get("podcasts_path")
    if not podcasts_path:
        podcasts_path = config.get("podcasts_path", "../../../podcasts")
        logger.warning("Using default podcasts_path")
    podcasts_path = Path(podcasts_path)

    peak = config.get("peak", -1.0)
    loudness = config.get("loudness", -23.0)
    block_size = config.get("block_size", 0.400)
    num_workers = int(config.get("loudness_num_workers", config.get("num_workers", 4)))
    loudness_batch_size = int(config.get("loudness_batch_size", 8))
    loudness_loader_workers = int(config.get("loudness_loader_workers", 2))
    loudness_prefetch_factor = int(config.get("loudness_prefetch_factor", 2))

    num_processes = num_workers

    logger.info(
        f"""
        Running loudness normalization:
        Podcasts path: {podcasts_path}
        Peak normalization: {peak} dB
        Target loudness: {loudness} LUFS
        Block size: {block_size} seconds
        Number of processes: {num_processes}
        """
    )

    audio_paths = discover_audio_paths(podcasts_path, config_path=args.config_path)
    if not audio_paths:
        logger.info("No audio files found for processing.")
        write_stage_status(
            stage=3,
            stage_name="preprocess_audio",
            log_dir=args.log_dir or "./logs",
            processed=0,
            skipped=0,
            errors=0,
        )
        return

    # 1) Make sure balalaika.csv exists; bootstrap from the audio tree if not.
    ensure_main_csv(podcasts_path, audio_paths=audio_paths)

    # 2) Absorb leftover partials from a previous interrupted run into the
    #    main CSV before scheduling work, so resume is automatic.
    _, absorbed = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=[NORMALIZED_COLUMN],
        bootstrap_audio_paths=audio_paths,
        preserve_existing=True,
    )
    if absorbed:
        logger.info(
            f"Absorbed {absorbed} rows from leftover {PARTIAL_PREFIX}_part_*.csv "
            "into balalaika.csv before scheduling new work."
        )

    # 3) Determine work: any file whose loudness_normalized cell is empty/false.
    paths_to_process = unprocessed_paths(podcasts_path, NORMALIZED_COLUMN, audio_paths)
    skipped = len(audio_paths) - len(paths_to_process)
    logger.info(
        f"Found {len(audio_paths)} audio files; "
        f"skipping {skipped} already normalized; processing {len(paths_to_process)}."
    )

    if not paths_to_process:
        logger.info("All audio files are already loudness-normalized.")
        write_stage_status(
            stage=3,
            stage_name="preprocess_audio",
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
        paths_to_process,
        shard_size=shard_size,
    )
    del paths_to_process

    logger.info(
        f"Starting loudness workers over {work_plan.total_items} files "
        f"in {work_plan.shard_count} shard(s)."
    )

    processed = mp.Value('i', 0)
    skipped = mp.Value('i', 0)
    errors = mp.Value('i', 0)

    csv_settings = load_csv_settings(args.config_path)

    try:
        with PeriodicCsvMerger(
            podcasts_path,
            prefix=PARTIAL_PREFIX,
            value_columns=[NORMALIZED_COLUMN],
            preserve_existing=True,
            **csv_settings,
        ):
            if num_processes > 1:
                mp.spawn(
                    run_worker,
                    args=(
                        num_processes,
                        str(work_plan.work_dir),
                        peak,
                        loudness,
                        block_size,
                        str(podcasts_path),
                        loudness_batch_size,
                        loudness_loader_workers,
                        loudness_prefetch_factor,
                        processed,
                        skipped,
                        errors,
                    ),
                    nprocs=num_processes,
                    join=True,
                )
            else:
                run_worker(
                    0,
                    1,
                    str(work_plan.work_dir),
                    peak,
                    loudness,
                    block_size,
                    str(podcasts_path),
                    loudness_batch_size,
                    loudness_loader_workers,
                    loudness_prefetch_factor,
                    processed,
                    skipped,
                    errors,
                )
    except KeyboardInterrupt:
        logger.warning("Loudness normalization interrupted; merging partials before exit.")

    # 4) Merge whatever partials the workers produced (even after a Ctrl+C).
    absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=[NORMALIZED_COLUMN],
        preserve_existing=True,
    )

    write_stage_status(
        stage=3,
        stage_name="preprocess_audio",
        log_dir=args.log_dir or "./logs",
        processed=processed.value,
        skipped=skipped.value,
        errors=errors.value,
    )

    logger.info("Loudness normalization stage complete.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Normalize audio loudness (ITU-R BS.1770-4) for all audio files in the dataset."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    args = parser.parse_args()

    main(args)
