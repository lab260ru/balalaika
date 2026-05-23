"""ClearVoice MossFormer2_SE_48K denoising stage.

This stage mirrors the Numpy-to-Numpy ClearVoice demo:

    ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])

Inputs are decoded with torchaudio, converted to mono 48 kHz float32 batches,
processed by ClearVoice, and written back in place. Progress is tracked in
``balalaika.csv`` through the ``denoised`` column so interrupted runs resume.
"""

import argparse
import warnings
from pathlib import Path
from typing import List, Set

import numpy as np
import torch
import torch.multiprocessing as mp
import torchaudio
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
from src.utils.datasets.denoising import (
    DENOISING_SAMPLE_RATE,
    create_denoising_dataloader,
)
from src.utils.gpu import apply_torch_perf_defaults
from src.utils.logging_setup import setup_logging
from src.utils.parallel import run_per_gpu_processes
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config

apply_torch_perf_defaults()


PARTIAL_PREFIX = "denoising"
PROCESSED_COLUMN = "denoised"
PARTIAL_FIELDS = ("filepath", PROCESSED_COLUMN)


def _write_audio(audio_path: str, samples: np.ndarray, sample_rate: int) -> None:
    tensor = torch.from_numpy(samples.astype(np.float32, copy=False))
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*save_with_torchcodec.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*StreamingMediaEncoder has been deprecated.*",
            category=UserWarning,
        )
        torchaudio.save(audio_path, tensor, sample_rate)


def _load_clearvoice(task: str, model_name: str):
    try:
        from clearvoice import ClearVoice
    except Exception as exc:
        raise RuntimeError(
            "ClearVoice is not installed. Install the clearvoice package before "
            "running the denoising stage."
        ) from exc

    logger.info(f"Loading ClearVoice model: task={task}, model={model_name}")
    return ClearVoice(task=task, model_names=[model_name])


def run_worker(
    rank: int,
    world_size: int,
    all_file_paths: List[str],
    config: dict,
    podcasts_path: Path,
    processed_counter,
    skipped_counter,
    errors_counter,
):
    my_files = all_file_paths[rank::world_size]
    if not my_files:
        logger.info(f"Worker {rank}: no files to process.")
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    task = str(config.get("task", "speech_enhancement"))
    model_name = str(config.get("model_name", "MossFormer2_SE_48K"))
    sample_rate = int(config.get("sample_rate", DENOISING_SAMPLE_RATE))
    batch_size = int(config.get("batch_size", 1))
    loader_workers = int(config.get("num_workers", 0))
    prefetch_factor = int(config.get("prefetch_factor", 2))

    model = _load_clearvoice(task=task, model_name=model_name)
    logger.info(
        f"Worker {rank}/{world_size}: {len(my_files)} files, "
        f"batch={batch_size}, sample_rate={sample_rate}, loader_workers={loader_workers}"
    )

    with PartialCsvWriter(
        podcasts_path, PARTIAL_PREFIX, rank, fieldnames=PARTIAL_FIELDS
    ) as writer:
        already_done: Set[str] = writer.already_done()
        skipped_counter.value += len(already_done)
        if already_done:
            logger.info(
                f"Worker {rank}: {len(already_done)} files already in this partial; skipping."
            )

        pending_files = [p for p in my_files if resolve_path(p) not in already_done]
        dataloader = create_denoising_dataloader(
            pending_files,
            batch_size=batch_size,
            num_workers=loader_workers,
            prefetch_factor=prefetch_factor,
            sample_rate=sample_rate,
        )

        for paths, batch, lengths, errors in tqdm(
            dataloader, desc=f"Denoising-{rank}", position=rank
        ):
            valid_indices = []
            valid_paths = []
            valid_lengths = []
            for idx, (path_str, length, error) in enumerate(zip(paths, lengths.tolist(), errors)):
                if error:
                    logger.error(f"Error loading {path_str}: {error}")
                    errors_counter.value += 1
                    continue
                if int(length) <= 0:
                    logger.warning(f"Skipping empty audio: {path_str}")
                    skipped_counter.value += 1
                    continue
                valid_indices.append(idx)
                valid_paths.append(path_str)
                valid_lengths.append(int(length))

            if not valid_indices:
                continue

            try:
                input_np = batch[valid_indices].numpy().astype(np.float32, copy=False)
                output_np = model(input_np, False)
                output_np = np.asarray(output_np, dtype=np.float32)
                if output_np.ndim == 1:
                    output_np = output_np[np.newaxis, :]
            except Exception as exc:
                logger.error(f"ClearVoice batch failed on worker {rank}: {exc}")
                errors_counter.value += len(valid_indices)
                continue

            for path_str, length, enhanced in zip(valid_paths, valid_lengths, output_np):
                try:
                    enhanced = np.asarray(enhanced[:length], dtype=np.float32)
                    _write_audio(str(path_str), enhanced[np.newaxis, :], sample_rate)
                    writer.write(
                        {
                            "filepath": resolve_path(path_str),
                            PROCESSED_COLUMN: True,
                        }
                    )
                    processed_counter.value += 1
                except Exception as exc:
                    logger.error(f"Failed to save denoised audio {path_str}: {exc}")
                    errors_counter.value += 1


def main():
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N pending files")
    args = parser.parse_args()

    setup_logging("denoising", log_dir=args.log_dir)
    config = load_config(args.config_path, "denoising")

    podcasts_path = Path(config.get("podcasts_path", "."))
    configured_processes = int(config.get("processes", 0))
    available_gpus = torch.cuda.device_count()
    if configured_processes > 0:
        num_processes = min(configured_processes, available_gpus) if available_gpus > 0 else configured_processes
    else:
        num_processes = available_gpus if available_gpus > 0 else 1
    num_processes = max(1, num_processes)

    audio_paths = discover_audio_paths(podcasts_path)
    if not audio_paths:
        logger.warning("No audio files found.")
        return

    ensure_main_csv(podcasts_path, audio_paths=audio_paths)

    _, absorbed = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=[PROCESSED_COLUMN],
        bootstrap_audio_paths=audio_paths,
    )
    if absorbed:
        logger.info(
            f"Absorbed {absorbed} rows from leftover {PARTIAL_PREFIX}_part_*.csv."
        )

    pending = unprocessed_paths(podcasts_path, PROCESSED_COLUMN, audio_paths)
    if args.limit is not None:
        pending = pending[: args.limit]

    if not pending:
        logger.success("All audio files are already denoised. Exiting.")
        return

    logger.info(
        f"Running denoising for {len(pending)} files with {num_processes} process(es) "
        f"({available_gpus} GPU(s) visible)."
    )

    processed = mp.Value("i", 0)
    skipped = mp.Value("i", 0)
    errors = mp.Value("i", 0)

    csv_settings = load_csv_settings(args.config_path)

    try:
        with PeriodicCsvMerger(
            podcasts_path,
            prefix=PARTIAL_PREFIX,
            value_columns=[PROCESSED_COLUMN],
            progress_counter=processed,
            **csv_settings,
        ):
            worker_errors, _ = run_per_gpu_processes(
                run_worker,
                num_gpus=num_processes,
                args=(pending, config, podcasts_path, processed, skipped, errors),
            )
            if worker_errors:
                errors.value += worker_errors
    except KeyboardInterrupt:
        logger.warning("Denoising interrupted; merging partials before exit.")
    except Exception as exc:
        logger.critical(f"Denoising multiprocessing failed: {exc}")
        errors.value += 1

    absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=[PROCESSED_COLUMN],
        bootstrap_audio_paths=audio_paths,
    )

    write_stage_status(
        stage=10,
        stage_name="denoising",
        log_dir=args.log_dir or "./logs",
        processed=processed.value,
        skipped=skipped.value,
        errors=errors.value,
    )

    logger.info("Denoising stage complete.")


if __name__ == "__main__":
    main()
