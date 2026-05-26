"""Spectra-0 anti-spoofing filter for generated speech detection.

The stage runs an ONNX classifier over fixed-size 16 kHz audio batches, writes
scores to ``balalaika.csv``, and deletes clips whose generated-speech
probability exceeds the configured threshold.
"""

import argparse
import os
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


def ensure_model(model_path: Path) -> None:
    if not model_path.exists():
        logger.info(
            f"Downloading Spectra-0 ONNX from Hugging Face: "
            f"{MODEL_REPO_ID}/{MODEL_REPO_FILENAME}"
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = Path(
            huggingface_hub.hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=MODEL_REPO_FILENAME,
                local_dir=str(model_path.parent),
            )
        )
        if downloaded != model_path:
            downloaded.replace(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Anti-spoofing ONNX model not found: {model_path}")


def create_session(model_path: Path, rank: int, cfg: dict, config_path: str | None) -> ort.InferenceSession:
    ensure_model(model_path)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    use_tensorrt = bool(cfg.get("use_tensorrt", False))
    providers = get_onnx_providers(rank, use_tensorrt=use_tensorrt, config_path=config_path)

    logger.info(f"[cuda:{rank}] Anti-spoofing ONNX providers: {providers}")
    return ort.InferenceSession(str(model_path), sess_options, providers=providers)


def run_worker(
    rank: int,
    world_size: int,
    all_paths: List[str],
    config: dict,
    config_path: str | None,
    processed_counter,
    skipped_counter,
    errors_counter,
) -> None:
    my_paths = all_paths[rank::world_size]
    if not my_paths:
        return

    torch.cuda.set_device(rank)
    podcasts_path = Path(config.get("podcasts_path", "."))
    cfg = config.get("antispoofing", {})

    model_path = Path(cfg.get("onnx_path", "./models/spectra_0.onnx"))
    threshold = float(cfg.get("threshold", 0.5))
    batch_size = int(cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 2))
    prefetch_factor = int(cfg.get("prefetch_factor", 2))

    logger.info(
        f"[cuda:{rank}] Anti-spoofing {len(my_paths)} files, "
        f"batch={batch_size}, threshold={threshold}, samples={MODEL_NUM_SAMPLES}"
    )

    try:
        session = create_session(model_path, rank, cfg, config_path)
        dataloader = create_antispoofing_dataloader(
            my_paths,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            sample_rate=MODEL_SAMPLE_RATE,
            num_samples=MODEL_NUM_SAMPLES,
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

            for paths, batch, lengths, load_errors in tqdm(
                dataloader, desc=f"AntiSpoof-{rank}", position=rank
            ):
                for path_str, reason in load_errors:
                    logger.error(f"Audio load failed {path_str}: {reason}")
                    errors_counter.value += 1

                if not paths:
                    continue

                feed = {MODEL_INPUT_NAME: batch.numpy().astype(np.float32, copy=False)}
                try:
                    logits = session.run([MODEL_OUTPUT_NAME], feed)[0]
                    generated_probs = generated_probability(logits, GENERATED_CLASS_INDEX)
                except Exception as exc:
                    logger.error(f"Anti-spoofing batch failed on worker {rank}: {exc}")
                    errors_counter.value += len(paths)
                    continue

                for path_str, length, prob in zip(paths, lengths.tolist(), generated_probs.tolist()):
                    resolved = resolve_path(path_str)
                    if resolved in already_done:
                        continue

                    duration_s = float(length) / float(MODEL_SAMPLE_RATE) if length else 0.0
                    if duration_s <= 0:
                        duration_s = safe_audio_duration(path_str)

                    prob_val = round(float(prob), 6)
                    deleted = False
                    if prob_val > threshold:
                        try:
                            os.remove(path_str)
                            deleted = True
                        except OSError as exc:
                            logger.warning(f"Could not delete {path_str}: {exc}")
                            errors_counter.value += 1

                    writer.write(
                        {
                            "filepath": resolved,
                            SCORE_COLUMN: prob_val,
                            PROB_COLUMN: prob_val,
                            "duration_s": round(duration_s, 4),
                            "deleted": deleted,
                        }
                    )
                    processed_counter.value += 1
    except Exception as exc:
        logger.exception(f"Anti-spoofing worker {rank} failed: {exc}")
        errors_counter.value += 1


def main(args):
    setup_logging("antispoofing", log_dir=args.log_dir)
    mp.set_start_method("spawn", force=True)
    config = load_config(args.config_path, "separation")
    podcasts_path = Path(config.get("podcasts_path", "."))
    cfg = config.get("antispoofing", {})

    audio_paths = discover_audio_paths(podcasts_path)
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

    logger.info(f"{len(pending)} files need anti-spoofing; starting workers on {n_gpus} GPUs.")

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
                args=(n_gpus, pending, config, args.config_path, processed, skipped, errors),
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
