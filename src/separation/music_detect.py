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
import time
from pathlib import Path
from typing import List, Set

import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger
from musicdetection.audio_sampler import LengthBasedBatchSampler
from musicdetection.core.model import WavLMForMusicDetection
from musicdetection.dataset import AudioCollate, MusicDetectionDataset
from safetensors import safe_open
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

from src.utils.audit import record_stage_summary, safe_audio_duration
from src.utils.audio_durations import duration_probe_workers, ensure_audio_durations
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
from src.utils.node_profile import resolve_batch_size
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config
from src.utils.work_shards import (
    claim_work_shard,
    load_work_shard_size,
    mark_work_shard_done,
    prepare_work_shards,
    read_work_shard,
)

apply_torch_perf_defaults(disable_math_sdp=False)


PARTIAL_PREFIX = "music"
COLUMN = "music_prob"
PARTIAL_FIELDS = ("filepath", "music_prob", "total_duration", "duration_s", "deleted")
VALUE_COLUMNS = [COLUMN, "total_duration"]


def create_loader(paths: List[str], model_name: str, batch_size: int, num_workers: int, audio_lengths: dict[str, float]):
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    dataset = MusicDetectionDataset(file_paths=paths, target_sample_rate=processor.sampling_rate)
    sampler = LengthBasedBatchSampler(paths, audio_lengths, batch_size=batch_size, shuffle=False)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=AudioCollate(processor),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_model(model_path: str, base_model: str, device: torch.device):
    model = WavLMForMusicDetection(base_model_name=base_model)
    with safe_open(model_path, framework="pt", device="cpu") as f:
        model.load_state_dict({k: f.get_tensor(k) for k in f.keys()})
    model = model.to(device).eval()
    model.device = device
    return model


def _process_files(
    rank: int,
    device: torch.device,
    files: List[str],
    config: dict,
    model,
    writer: PartialCsvWriter,
    already_done: Set[str],
    processed_counter,
    skipped_counter,
    errors_counter,
) -> int:
    cfg = config.get("music_detect", {})
    threshold = cfg.get("threshold", 0.5)
    podcasts_path = Path(config.get("podcasts_path", "."))

    pending_files = []
    for path in files:
        resolved = resolve_path(path)
        if resolved in already_done:
            skipped_counter.value += 1
            continue
        pending_files.append(path)

    if not pending_files:
        return 0

    audio_lengths = ensure_audio_durations(
        podcasts_path,
        pending_files,
        num_workers=duration_probe_workers(cfg, config),
    )
    dataloader = create_loader(
        pending_files,
        cfg.get("base_model", "microsoft/wavlm-base-plus"),
        resolve_batch_size("music_detect", cfg.get("bs"), 32),
        cfg.get("num_workers", 4),
        audio_lengths,
    )
    loader_workers = int(cfg.get("num_workers", 4))
    batch_size = resolve_batch_size("music_detect", cfg.get("bs"), 32)
    logger.debug(
        f"perf dataloader_config stage=music_detect rank={rank} "
        f"batch_size={batch_size} workers={loader_workers} "
        f"prefetch_factor=library_default prefetch_batches=library_default "
        f"items={len(pending_files)}"
    )

    inference_started_at = time.perf_counter()
    probs, paths = model.predict_proba(dataloader)
    logger.debug(
        f"perf model=music_detect event=predict_proba rank={rank} "
        f"seconds={time.perf_counter() - inference_started_at:.6f} "
        f"items={len(paths)}"
    )

    deleted_count = 0
    for path, prob in zip(paths, probs.detach().flatten()):
        resolved = resolve_path(path)
        if resolved in already_done:
            skipped_counter.value += 1
            continue

        prob_val = round(float(prob), 6)
        duration_s = float(audio_lengths.get(str(path), 0.0))
        if duration_s <= 0:
            duration_s = safe_audio_duration(path)

        deleted = False
        if prob_val > threshold:
            try:
                delete_started_at = time.perf_counter()
                os.remove(path)
                logger.debug(
                    f"perf audio_delete stage=music_detect rank={rank} "
                    f"seconds={time.perf_counter() - delete_started_at:.6f} path={path}"
                )
                deleted_count += 1
                deleted = True
            except OSError as exc:
                logger.warning(f"Could not delete {path}: {exc}")
                errors_counter.value += 1

        write_started_at = time.perf_counter()
        writer.write(
            {
                "filepath": resolved,
                "music_prob": prob_val,
                "total_duration": round(duration_s, 4),
                "duration_s": round(duration_s, 4),
                "deleted": deleted,
            }
        )
        logger.debug(
            f"perf partial_write stage=music_detect rank={rank} "
            f"seconds={time.perf_counter() - write_started_at:.6f} path={resolved}"
        )
        already_done.add(resolved)
        processed_counter.value += 1

    return deleted_count


def run_worker(rank: int, world_size: int, work_dir: str, config: dict, processed_counter, skipped_counter, errors_counter):
    device = torch.device(f"cuda:{rank}")
    cfg = config.get("music_detect", {})
    podcasts_path = Path(config.get("podcasts_path", "."))

    try:
        model = load_model(
            cfg.get("music_detect_model"),
            cfg.get("base_model", "microsoft/wavlm-base-plus"),
            device,
        )

        deleted_total = 0
        claimed = 0
        with PartialCsvWriter(
            podcasts_path, PARTIAL_PREFIX, rank, fieldnames=PARTIAL_FIELDS
        ) as writer:
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
                logger.info(f"[{device}] Processing {len(shard_files)} files from {shard_path.name}...")
                deleted_total += _process_files(
                    rank,
                    device,
                    shard_files,
                    config,
                    model,
                    writer,
                    already_done,
                    processed_counter,
                    skipped_counter,
                    errors_counter,
                )
                mark_work_shard_done(shard_path)

        logger.success(f"[{device}] Done. Claimed {claimed} shard(s), deleted {deleted_total} files.")

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
    audio_paths = discover_audio_paths(podcasts_path, config_path=args.config_path)
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

    shard_size = load_work_shard_size(args.config_path)
    work_plan = prepare_work_shards(
        podcasts_path,
        PARTIAL_PREFIX,
        pending,
        shard_size=shard_size,
    )
    del pending

    logger.info(
        f"{work_plan.total_items} files still need a music_prob; "
        f"starting workers on {n_gpus} GPUs over {work_plan.shard_count} shard(s)."
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
            mp.spawn(run_worker, args=(n_gpus, str(work_plan.work_dir), config, processed, skipped, errors), nprocs=n_gpus, join=True)
    except KeyboardInterrupt:
        logger.warning("Music detection stage interrupted; merging partials before exit.")

    # 4) Merge whatever the workers managed to produce.
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
