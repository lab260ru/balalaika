"""Spectra-0 anti-spoofing filter for generated speech detection.

The stage runs an ONNX classifier over fixed-size 16 kHz audio batches, writes
scores to ``balalaika.csv``, and deletes clips whose generated-speech
probability exceeds the configured threshold.
"""

import argparse
import os
import time
from pathlib import Path
from typing import List, Set

import huggingface_hub
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm

from src.utils.audit import record_stage_summary, safe_audio_duration
from src.utils.audio_durations import (
    duration_bucket_settings,
    duration_probe_workers,
    ensure_audio_durations,
)
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
from src.utils.datasets.separation import (
    ANTISPOOF_NUM_SAMPLES,
    ANTISPOOF_SAMPLE_RATE,
    create_antispoofing_dataloader,
)
from src.utils.gpu import get_onnx_providers
from src.utils.logging_setup import setup_logging
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config
from src.utils.work_shards import (
    claim_work_shard,
    load_work_shard_size,
    mark_work_shard_done,
    prepare_length_bucketed_work_shards,
    read_work_shard,
)


PARTIAL_PREFIX = "antispoof"
SCORE_COLUMN = "antispoof_score"
PROB_COLUMN = "antispoof_generated_prob"
PARTIAL_FIELDS = ("filepath", SCORE_COLUMN, PROB_COLUMN, "duration_s", "deleted")
MODEL_SAMPLE_RATE = ANTISPOOF_SAMPLE_RATE
MODEL_NUM_SAMPLES = ANTISPOOF_NUM_SAMPLES
GENERATED_CLASS_INDEX = 1
MODEL_INPUT_NAME = "waveform"
MODEL_OUTPUT_NAME = "logits"
MODEL_REPO_ID = "lab260/spectra_0"
MODEL_REPO_FILENAME = "model.onnx"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def generated_probability(logits: np.ndarray, class_index: int) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 1:
        logits = logits[:, None]
    if logits.shape[-1] == 1:
        return sigmoid(logits[:, 0])
    probs = softmax(logits)
    return probs[:, class_index]


def ensure_model(model_path: Path, cfg: dict | None = None) -> None:
    if model_path.exists():
        return

    import huggingface_hub

    logger.info(f"Downloading denoising ONNX from Hugging Face")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        huggingface_hub.hf_hub_download(
            repo_id='NikiPshg/spectra_0_onnx',
            filename='spectra_0.onnx',
            local_dir='./models',
        )
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Denoising ONNX model not found: {model_path}")


def create_session(model_path: Path, rank: int, cfg: dict, config_path: str | None) -> ort.InferenceSession:
    ensure_model(model_path, cfg)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    use_tensorrt = bool(cfg.get("use_tensorrt", False))
    providers = get_onnx_providers(rank, use_tensorrt=use_tensorrt, config_path=config_path)

    logger.info(f"[cuda:{rank}] Anti-spoofing ONNX providers: {providers}")
    return ort.InferenceSession(str(model_path), sess_options, providers=providers)


def _process_files(
    rank: int,
    files: List[str],
    session: ort.InferenceSession,
    cfg: dict,
    writer: PartialCsvWriter,
    already_done: Set[str],
    processed_counter,
    skipped_counter,
    errors_counter,
) -> None:
    threshold = float(cfg.get("threshold", 0.5))
    batch_size = int(cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 2))
    prefetch_factor = int(cfg.get("prefetch_factor", 2))

    pending_files = []
    for path in files:
        resolved = resolve_path(path)
        if resolved in already_done:
            skipped_counter.value += 1
            continue
        pending_files.append(path)

    if not pending_files:
        return

    dataloader = create_antispoofing_dataloader(
        pending_files,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sample_rate=MODEL_SAMPLE_RATE,
        num_samples=MODEL_NUM_SAMPLES,
    )
    prefetch_batches = num_workers * prefetch_factor if num_workers > 0 else 0
    logger.debug(
        f"perf dataloader_config stage=antispoofing rank={rank} "
        f"batch_size={batch_size} workers={num_workers} "
        f"prefetch_factor={prefetch_factor} prefetch_batches={prefetch_batches} "
        f"items={len(pending_files)}"
    )

    batch_wait_started_at = time.perf_counter()
    for batch_idx, (paths, batch, lengths, load_errors) in enumerate(tqdm(
        dataloader, desc=f"AntiSpoof-{rank}", position=rank
    )):
        batch_received_at = time.perf_counter()
        logger.debug(
            f"perf dataloader_wait stage=antispoofing rank={rank} "
            f"batch={batch_idx} seconds={batch_received_at - batch_wait_started_at:.6f} "
            f"items={len(paths)}"
        )
        for path_str, reason in load_errors:
            logger.error(f"Audio load failed {path_str}: {reason}")
            errors_counter.value += 1

        if not paths:
            batch_wait_started_at = time.perf_counter()
            continue

        feed = {MODEL_INPUT_NAME: batch.numpy().astype(np.float32, copy=False)}
        try:
            inference_started_at = time.perf_counter()
            logits = session.run([MODEL_OUTPUT_NAME], feed)[0]
            generated_probs = generated_probability(logits, GENERATED_CLASS_INDEX)
            logger.debug(
                f"perf model=antispoofing event=inference rank={rank} "
                f"batch={batch_idx} seconds={time.perf_counter() - inference_started_at:.6f} "
                f"items={len(paths)} frames={int(batch.shape[-1])}"
            )
        except Exception as exc:
            logger.error(f"Anti-spoofing batch failed on worker {rank}: {exc}")
            errors_counter.value += len(paths)
            batch_wait_started_at = time.perf_counter()
            continue

        for path_str, length, prob in zip(paths, lengths.tolist(), generated_probs.tolist()):
            resolved = resolve_path(path_str)
            if resolved in already_done:
                skipped_counter.value += 1
                continue

            duration_s = float(length) / float(MODEL_SAMPLE_RATE) if length else 0.0
            if duration_s <= 0:
                duration_s = safe_audio_duration(path_str)

            prob_val = round(float(prob), 6)
            deleted = False
            if prob_val > threshold:
                try:
                    delete_started_at = time.perf_counter()
                    os.remove(path_str)
                    logger.debug(
                        f"perf audio_delete stage=antispoofing rank={rank} "
                        f"seconds={time.perf_counter() - delete_started_at:.6f} path={path_str}"
                    )
                    deleted = True
                except OSError as exc:
                    logger.warning(f"Could not delete {path_str}: {exc}")
                    errors_counter.value += 1

            write_started_at = time.perf_counter()
            writer.write(
                {
                    "filepath": resolved,
                    SCORE_COLUMN: prob_val,
                    PROB_COLUMN: prob_val,
                    "duration_s": round(duration_s, 4),
                    "deleted": deleted,
                }
            )
            logger.debug(
                f"perf partial_write stage=antispoofing rank={rank} "
                f"seconds={time.perf_counter() - write_started_at:.6f} path={resolved}"
            )
            already_done.add(resolved)
            processed_counter.value += 1
        batch_wait_started_at = time.perf_counter()


def run_worker(
    rank: int,
    world_size: int,
    work_dir: str,
    config: dict,
    config_path: str | None,
    processed_counter,
    skipped_counter,
    errors_counter,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    podcasts_path = Path(config.get("podcasts_path", "."))
    cfg = config.get("antispoofing", {})

    model_path = Path(cfg.get("onnx_path", "./models/spectra_0.onnx"))
    batch_size = int(cfg.get("batch_size", 8))
    threshold = float(cfg.get("threshold", 0.5))

    logger.info(
        f"[cuda:{rank}] Anti-spoofing claiming shards, "
        f"batch={batch_size}, threshold={threshold}, samples={MODEL_NUM_SAMPLES}"
    )

    try:
        session = create_session(model_path, rank, cfg, config_path)
        claimed = 0
        with PartialCsvWriter(
            podcasts_path, PARTIAL_PREFIX, rank, fieldnames=PARTIAL_FIELDS
        ) as writer:
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
                logger.info(f"[cuda:{rank}] Processing {len(shard_files)} files from {shard_path.name}")
                _process_files(
                    rank,
                    shard_files,
                    session,
                    cfg,
                    writer,
                    already_done,
                    processed_counter,
                    skipped_counter,
                    errors_counter,
                )
                mark_work_shard_done(shard_path)

        logger.info(f"Worker {rank} done after {claimed} claimed shard(s).")
    except Exception as exc:
        logger.exception(f"Anti-spoofing worker {rank} failed: {exc}")
        errors_counter.value += 1

def main(args):
    setup_logging("antispoofing", log_dir=args.log_dir)
    mp.set_start_method("spawn", force=True)
    config = load_config(args.config_path, "separation")
    podcasts_path = Path(config.get("podcasts_path", "."))
    cfg = config.get("antispoofing", {})

    audio_paths = discover_audio_paths(podcasts_path, config_path=args.config_path)
    n_gpus = torch.cuda.device_count()

    if not audio_paths:
        logger.warning("No audio files found.")
        return
    if n_gpus == 0:
        logger.error("No GPU found.")
        return

    ensure_main_csv(podcasts_path, audio_paths=audio_paths)
    ensure_model(Path(cfg.get("onnx_path", "./models/spectra_0.onnx")))

    leftover_partials, absorbed = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=[SCORE_COLUMN, PROB_COLUMN],
        drop_missing_files=True,
        bootstrap_audio_paths=audio_paths,
    )
    if absorbed:
        logger.info(f"Absorbed {absorbed} leftover anti-spoofing rows.")

    pending = unprocessed_paths(podcasts_path, SCORE_COLUMN, audio_paths)
    if args.limit is not None:
        pending = pending[: args.limit]

    if not pending:
        logger.success("All audio files already have anti-spoofing scores.")
        audit = audit_from_filter_partials(leftover_partials)
        if audit["files_in"] == 0:
            audit["files_in"] = len(audio_paths)
            audit["files_out"] = len(audio_paths)
        record_stage_summary(
            podcasts_path=podcasts_path,
            stage="antispoofing",
            files_in=audit["files_in"],
            files_out=audit["files_out"],
            hours_in=audit["hours_in"],
            hours_out=audit["hours_out"],
            params={"threshold": cfg.get("threshold", 0.5), "deleted": audit["files_deleted"]},
        )
        return

    shard_size = load_work_shard_size(args.config_path)
    duration_workers = duration_probe_workers(cfg, config)
    durations = ensure_audio_durations(
        podcasts_path,
        pending,
        num_workers=duration_workers,
    )
    bucket_seconds, max_bucket_duration = duration_bucket_settings(
        args.config_path,
        cfg,
        config,
    )
    work_plan = prepare_length_bucketed_work_shards(
        podcasts_path,
        PARTIAL_PREFIX,
        pending,
        durations,
        shard_size=shard_size,
        bucket_seconds=bucket_seconds,
        max_duration=max_bucket_duration,
    )
    del pending
    del durations

    logger.info(
        f"{work_plan.total_items} files need anti-spoofing; "
        f"starting workers on {n_gpus} GPUs over {work_plan.shard_count} shard(s)."
    )

    processed = mp.Value("i", 0)
    skipped = mp.Value("i", 0)
    errors = mp.Value("i", 0)
    csv_settings = load_csv_settings(args.config_path)

    try:
        with PeriodicCsvMerger(
            podcasts_path,
            prefix=PARTIAL_PREFIX,
            value_columns=[SCORE_COLUMN, PROB_COLUMN],
            drop_missing_files=True,
            **csv_settings,
        ):
            mp.spawn(
                run_worker,
                args=(n_gpus, str(work_plan.work_dir), config, args.config_path, processed, skipped, errors),
                nprocs=n_gpus,
                join=True,
            )
    except KeyboardInterrupt:
        logger.warning("Anti-spoofing interrupted; merging partials before exit.")

    new_partials, _ = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=[SCORE_COLUMN, PROB_COLUMN],
        drop_missing_files=True,
    )
    combined = pd.concat(
        [df for df in (leftover_partials, new_partials) if df is not None and not df.empty],
        ignore_index=True,
    ) if (leftover_partials is not None or new_partials is not None) else pd.DataFrame()
    audit = audit_from_filter_partials(combined)

    record_stage_summary(
        podcasts_path=podcasts_path,
        stage="antispoofing",
        files_in=audit["files_in"],
        files_out=audit["files_out"],
        hours_in=audit["hours_in"],
        hours_out=audit["hours_out"],
        params={"threshold": cfg.get("threshold", 0.5), "deleted": audit["files_deleted"]},
    )

    write_stage_status(
        stage=5.6,
        stage_name="antispoofing",
        log_dir=args.log_dir or "./logs",
        processed=processed.value,
        skipped=skipped.value,
        errors=errors.value,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N pending files")
    main(parser.parse_args())
