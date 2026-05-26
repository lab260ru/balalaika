"""Music detection filter built on a fine-tuned WavLM head.

In addition to scoring each chunk and deleting clips above the threshold, the
worker now records per-file durations into the partial CSV. That lets the
rank-0 process emit a stage row in ``filter_summary.csv`` capturing the actual
hours of audio dropped, even though the files themselves are gone.

CSV resilience:

* The shared :mod:`src.utils.csv_manager` handles bootstrapping
  ``balalaika.csv`` (creating it from the audio tree when absent) and atomic
  rewrites.
* Each worker streams rows to ``music_part_<rank>.csv`` via
  :class:`PartialCsvWriter` (``flush()`` after every row), so a forced stop
  preserves whatever rows were already produced.
* On startup any leftover ``music_part_*.csv`` from a prior interrupted run
  is absorbed into the main CSV before scheduling new work — re-running this
  stage simply *resumes* scoring.
* Files already scored (``music_prob`` populated in ``balalaika.csv``) are
  skipped automatically.

A per-stage log file is initialised at startup for offline debugging.
"""

import argparse
import os
from pathlib import Path
from typing import List, Set

import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger
from musicdetection.audio_cache import create_audio_length_cache
from musicdetection.audio_sampler import LengthBasedBatchSampler
from musicdetection.core.model import WavLMForMusicDetection
from musicdetection.dataset import AudioCollate, MusicDetectionDataset
from safetensors import safe_open
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

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
from src.utils.gpu import apply_torch_perf_defaults
from src.utils.logging_setup import setup_logging
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config

apply_torch_perf_defaults(disable_math_sdp=False)


PARTIAL_PREFIX = "music"
PARTIAL_FIELDS = ("filepath", "music_prob", "duration_s", "deleted")
COLUMN = "music_prob"


def create_loader(paths: List[str], model_name: str, batch_size: int, num_workers: int, cache_file: Path):
    audio_lengths = create_audio_length_cache(file_paths=paths, cache_file=str(cache_file))
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    dataset = MusicDetectionDataset(file_paths=paths, target_sample_rate=processor.sampling_rate)
    sampler = LengthBasedBatchSampler(paths, audio_lengths, batch_size=batch_size, shuffle=False)
    return (
        DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=AudioCollate(processor),
            num_workers=num_workers,
            pin_memory=False,
        ),
        audio_lengths,
    )


def load_model(model_path: str, base_model: str, device: torch.device):
    model = WavLMForMusicDetection(base_model_name=base_model)
    with safe_open(model_path, framework="pt", device="cpu") as f:
        model.load_state_dict({k: f.get_tensor(k) for k in f.keys()})
    model = model.to(device).eval()
    model.device = device
    return model


def run_worker(rank: int, world_size: int, all_paths: List[str], config: dict, processed_counter, skipped_counter, errors_counter):
    my_paths = all_paths[rank::world_size]
    if not my_paths:
        return

    device = torch.device(f"cuda:{rank}")
    cfg = config.get("music_detect", {})
    podcasts_path = Path(config.get("podcasts_path", "."))

    threshold = cfg.get("threshold", 0.5)
    cache_dir = Path(cfg.get("cache_path", "./cache")) / f"nisqa_temp_worker_{rank}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{device}] Processing {len(my_paths)} files...")

    try:
        dataloader, audio_lengths = create_loader(
            my_paths,
            cfg.get("base_model", "microsoft/wavlm-base-plus"),
            cfg.get("bs", 32),
            cfg.get("num_workers", 4),
            cache_dir / "audio_lengths.json",
        )

        model = load_model(
            cfg.get("music_detect_model"),
            cfg.get("base_model", "microsoft/wavlm-base-plus"),
            device,
        )

        probs, paths = model.predict_proba(dataloader)

        deleted_count = 0
        with PartialCsvWriter(
            podcasts_path, PARTIAL_PREFIX, rank, fieldnames=PARTIAL_FIELDS
        ) as writer:
            already_done: Set[str] = writer.already_done()
            skipped_counter.value += len(already_done)
            if already_done:
                logger.info(
                    f"Worker {rank}: {len(already_done)} files already scored in this partial; skipping."
                )

            for path, prob in zip(paths, probs.detach().flatten()):
                resolved = resolve_path(path)
                if resolved in already_done:
                    continue

                prob_val = round(float(prob), 6)
                duration_s = float(audio_lengths.get(str(path), 0.0))
                if duration_s <= 0:
                    duration_s = safe_audio_duration(path)

                deleted = False
                if prob_val > threshold:
                    try:
                        os.remove(path)
                        deleted_count += 1
                        deleted = True
                    except OSError as exc:
                        logger.warning(f"Could not delete {path}: {exc}")
                        errors_counter.value += 1

                writer.write(
                    {
                        "filepath": resolved,
                        "music_prob": prob_val,
                        "duration_s": round(duration_s, 4),
                        "deleted": deleted,
                    }
                )
                processed_counter.value += 1

        logger.success(f"[{device}] Done. Deleted {deleted_count}/{len(my_paths)} files.")

    except Exception as exc:
        logger.exception(f"Worker {rank} error: {exc}")
        errors_counter.value += 1


def main(args):
    setup_logging("music_detect", log_dir=args.log_dir)
    mp.set_start_method("spawn", force=True)
    config = load_config(args.config_path, "separation")
    podcasts_path = config.get("podcasts_path")

    if not podcasts_path:
        logger.error("No podcasts_path in config")
        return

    podcasts_path = Path(podcasts_path)
    audio_paths = discover_audio_paths(podcasts_path)
    n_gpus = torch.cuda.device_count()

    if not audio_paths:
        logger.warning("No audio files found.")
        return

    if n_gpus == 0:
        logger.error("No GPU found.")
        return

    # 1) Make sure balalaika.csv exists; bootstrap from the audio tree if not.
    ensure_main_csv(podcasts_path, audio_paths=audio_paths)

    # 2) Absorb any leftover partials from a previous interrupted run into the
    #    main CSV before scheduling new work, so resume is automatic.
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

    # 3) Determine work: any file without a music_prob value yet.
    pending = unprocessed_paths(podcasts_path, COLUMN, audio_paths)
    if not pending:
        logger.success("All audio files already have a music_prob entry. Skipping computation.")
        audit = audit_from_filter_partials(leftover_partials)
        if audit["files_in"] == 0:
            audit["files_in"] = len(audio_paths)
            audit["files_out"] = len(audio_paths)
        cfg = config.get("music_detect", {})
        record_stage_summary(
            podcasts_path=podcasts_path,
            stage="music_detect",
            files_in=audit["files_in"],
            files_out=audit["files_out"],
            hours_in=audit["hours_in"],
            hours_out=audit["hours_out"],
            params={"threshold": cfg.get("threshold", 0.5), "deleted": audit["files_deleted"]},
        )
        return

    logger.info(f"{len(pending)} files still need a music_prob; starting workers on {n_gpus} GPUs.")

    processed = mp.Value('i', 0)
    skipped = mp.Value('i', 0)
    errors = mp.Value('i', 0)

    csv_settings = load_csv_settings(args.config_path)

    try:
        with PeriodicCsvMerger(
            podcasts_path,
            prefix=PARTIAL_PREFIX,
            value_columns=[COLUMN],
            drop_missing_files=True,
            **csv_settings,
        ):
            mp.spawn(run_worker, args=(n_gpus, pending, config, processed, skipped, errors), nprocs=n_gpus, join=True)
    except KeyboardInterrupt:
        logger.warning("Music detection stage interrupted; merging partials before exit.")

    # 4) Merge whatever the workers managed to produce.
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

    cfg = config.get("music_detect", {})
    record_stage_summary(
        podcasts_path=podcasts_path,
        stage="music_detect",
        files_in=audit["files_in"],
        files_out=audit["files_out"],
        hours_in=audit["hours_in"],
        hours_out=audit["hours_out"],
        params={"threshold": cfg.get("threshold", 0.5), "deleted": audit["files_deleted"]},
    )

    write_stage_status(
        stage=4,
        stage_name="music_detect",
        log_dir=args.log_dir or "./logs",
        processed=processed.value,
        skipped=skipped.value,
        errors=errors.value,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    main(parser.parse_args())
