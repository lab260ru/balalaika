"""Spectra-0 anti-spoofing scoring with crash-safe CSV state.

The stage runs the ONNX model over fixed-size 16 kHz audio batches and stores
the model's two raw output scores in ``balalaika.csv``. It does not apply
sigmoid/softmax and does not delete audio; filtering is handled by
``src.separation.antispoofing_filter``.
"""

import argparse
import shutil
import time
from pathlib import Path
from typing import List, Set

import huggingface_hub
import numpy as np
import onnxruntime as ort
import torch
import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm

from src.utils.audio_durations import (
    duration_bucket_settings,
    duration_probe_workers,
    ensure_audio_durations,
)
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
from src.utils.datasets.separation import (
    ANTISPOOF_NUM_SAMPLES,
    ANTISPOOF_SAMPLE_RATE,
    create_antispoofing_dataloader,
)
from src.utils.gpu import (
    apply_ort_thread_caps,
    get_onnx_providers,
    onnx_first_input_name,
)
from src.utils.logging_setup import setup_logging
from src.utils.node_profile import resolve_batch_size
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
SCORE_BONAFIDE_COLUMN = "score_bonafide"
SCORE_SPOOF_COLUMN = "score_spoof"
PARTIAL_FIELDS = ("filepath", SCORE_BONAFIDE_COLUMN, SCORE_SPOOF_COLUMN)
VALUE_COLUMNS = [SCORE_BONAFIDE_COLUMN, SCORE_SPOOF_COLUMN]
MODEL_SAMPLE_RATE = ANTISPOOF_SAMPLE_RATE
MODEL_NUM_SAMPLES = ANTISPOOF_NUM_SAMPLES
SPOOF_CLASS_INDEX = 0
BONAFIDE_CLASS_INDEX = 1
MODEL_REPO_ID = "NikiPshg/spectra_0_onnx"
MODEL_REPO_FILENAME = "spectra_0.onnx"


def raw_class_scores(outputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(bonafide, spoof)`` raw scores from Spectra-0 outputs."""
    scores = np.asarray(outputs, dtype=np.float32)
    if scores.ndim == 1:
        scores = scores[None, :]
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError(
            f"Expected Spectra-0 output shaped (batch, 2), got {scores.shape}"
        )
    return scores[:, BONAFIDE_CLASS_INDEX], scores[:, SPOOF_CLASS_INDEX]


def ensure_model(model_path: Path, cfg: dict | None = None) -> Path:
    if model_path.exists():
        return model_path

    cfg = cfg or {}
    repo_id = str(cfg.get("repo_id", MODEL_REPO_ID))
    filename = str(cfg.get("repo_filename", MODEL_REPO_FILENAME))
    logger.info(f"Downloading Spectra-0 ONNX from {repo_id}/{filename}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(model_path.parent),
        )
    )
    if downloaded != model_path:
        shutil.copy2(downloaded, model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Spectra-0 ONNX model not found: {model_path}")
    return model_path


def create_session(
    model_path: Path,
    rank: int,
    cfg: dict,
    config_path: str | None,
) -> tuple[ort.InferenceSession, str, str]:
    model_path = ensure_model(model_path, cfg)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # No-op unless runtime.threads_per_worker is set (default keeps ORT's
    # physical-core intra-op pool, so single-worker latency is unchanged).
    apply_ort_thread_caps(options, config_path=config_path)
    use_tensorrt = bool(cfg.get("use_tensorrt", False))
    providers = get_onnx_providers(
        rank,
        use_tensorrt=use_tensorrt,
        config_path=config_path,
    )
    if use_tensorrt:
        # Pin one dynamic-batch TRT profile (1..batch_size x fixed samples).
        # Without it every distinct trailing-partial-batch size would trigger
        # a fresh multi-minute engine build.
        batch_size = resolve_batch_size("antispoofing", cfg.get("batch_size"), 8)
        graph_input = onnx_first_input_name(model_path)
        patched = []
        for provider in providers:
            name, opts = provider if isinstance(provider, tuple) else (provider, {})
            opts = dict(opts)
            if name == "TensorrtExecutionProvider":
                opts.update(
                    {
                        "trt_profile_min_shapes": f"{graph_input}:1x{MODEL_NUM_SAMPLES}",
                        "trt_profile_opt_shapes": f"{graph_input}:{batch_size}x{MODEL_NUM_SAMPLES}",
                        "trt_profile_max_shapes": f"{graph_input}:{batch_size}x{MODEL_NUM_SAMPLES}",
                        "trt_timing_cache_enable": True,
                    }
                )
            patched.append((name, opts))
        providers = patched
    logger.info(f"[cuda:{rank}] Spectra-0 ONNX providers: {providers}")
    session = ort.InferenceSession(str(model_path), options, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logger.info(f"[cuda:{rank}] Spectra-0 IO: input={input_name}, output={output_name}")
    return session, input_name, output_name


def _process_files(
    rank: int,
    files: List[str],
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    cfg: dict,
    writer: PartialCsvWriter,
    already_done: Set[str],
    processed_counter,
    skipped_counter,
    errors_counter,
) -> None:
    batch_size = resolve_batch_size("antispoofing", cfg.get("batch_size"), 8)
    num_workers = int(cfg.get("num_workers", 2))
    prefetch_factor = int(cfg.get("prefetch_factor", 2))
    pending_files = []
    for path in files:
        resolved = resolve_path(path)
        if resolved in already_done:
            skipped_counter.value += 1
        else:
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
        ranged_decode=bool(cfg.get("ranged_decode", False)),
    )
    prefetch_batches = num_workers * prefetch_factor if num_workers > 0 else 0
    logger.debug(
        f"perf dataloader_config stage=antispoofing rank={rank} "
        f"batch_size={batch_size} workers={num_workers} "
        f"prefetch_factor={prefetch_factor} prefetch_batches={prefetch_batches} "
        f"items={len(pending_files)}"
    )

    wait_started_at = time.perf_counter()
    for batch_idx, (paths, batch, _lengths, load_errors) in enumerate(
        tqdm(dataloader, desc=f"AntiSpoof-{rank}", position=rank)
    ):
        received_at = time.perf_counter()
        logger.debug(
            f"perf dataloader_wait stage=antispoofing rank={rank} "
            f"batch={batch_idx} seconds={received_at - wait_started_at:.6f} "
            f"items={len(paths)}"
        )
        for path_str, reason in load_errors:
            logger.error(f"Audio load failed {path_str}: {reason}")
            errors_counter.value += 1
        if not paths:
            wait_started_at = time.perf_counter()
            continue

        try:
            started_at = time.perf_counter()
            outputs = session.run(
                [output_name],
                {input_name: batch.numpy().astype(np.float32, copy=False)},
            )[0]
            bonafide_scores, spoof_scores = raw_class_scores(outputs)
            logger.debug(
                f"perf model=antispoofing event=inference rank={rank} "
                f"batch={batch_idx} seconds={time.perf_counter() - started_at:.6f} "
                f"items={len(paths)} frames={int(batch.shape[-1])}"
            )
        except Exception as exc:
            logger.error(f"Spectra-0 batch failed on worker {rank}: {exc}")
            errors_counter.value += len(paths)
            wait_started_at = time.perf_counter()
            continue

        for path_str, bonafide_score, spoof_score in zip(
            paths, bonafide_scores.tolist(), spoof_scores.tolist()
        ):
            resolved = resolve_path(path_str)
            if resolved in already_done:
                skipped_counter.value += 1
                continue
            writer.write(
                {
                    "filepath": resolved,
                    SCORE_BONAFIDE_COLUMN: float(bonafide_score),
                    SCORE_SPOOF_COLUMN: float(spoof_score),
                }
            )
            already_done.add(resolved)
            processed_counter.value += 1
        wait_started_at = time.perf_counter()


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
    logger.info(
        f"[cuda:{rank}] Spectra-0 claiming shards, "
        f"batch={resolve_batch_size('antispoofing', cfg.get('batch_size'), 8)}, samples={MODEL_NUM_SAMPLES}"
    )

    try:
        session, input_name, output_name = create_session(
            model_path, rank, cfg, config_path
        )
        claimed = 0
        with PartialCsvWriter(
            podcasts_path, PARTIAL_PREFIX, rank, fieldnames=PARTIAL_FIELDS
        ) as writer:
            already_done: Set[str] = writer.already_done()
            while True:
                shard_path = claim_work_shard(work_dir, rank)
                if shard_path is None:
                    break
                shard_files = read_work_shard(shard_path)
                claimed += 1
                logger.info(
                    f"[cuda:{rank}] Processing {len(shard_files)} files "
                    f"from {shard_path.name}"
                )
                _process_files(
                    rank,
                    shard_files,
                    session,
                    input_name,
                    output_name,
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
        logger.exception(f"Spectra-0 worker {rank} failed: {exc}")
        errors_counter.value += 1


def _write_status(args, processed: int, skipped: int, errors: int) -> None:
    write_stage_status(
        stage=6,
        stage_name="antispoofing",
        log_dir=args.log_dir or "./logs",
        processed=processed,
        skipped=skipped,
        errors=errors,
    )


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
        _write_status(args, 0, 0, 0)
        return
    if n_gpus == 0:
        logger.error("No GPU found.")
        _write_status(args, 0, 0, 1)
        return

    ensure_main_csv(podcasts_path, audio_paths=audio_paths)
    ensure_model(Path(cfg.get("onnx_path", "./models/spectra_0.onnx")), cfg)
    _, absorbed = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=VALUE_COLUMNS,
        bootstrap_audio_paths=audio_paths,
        preserve_existing=True,
    )
    if absorbed:
        logger.info(f"Absorbed {absorbed} leftover Spectra-0 rows.")

    pending = unprocessed_paths(podcasts_path, SCORE_SPOOF_COLUMN, audio_paths)
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        logger.success("All audio files already have Spectra-0 scores.")
        _write_status(args, 0, len(audio_paths), 0)
        return

    durations = ensure_audio_durations(
        podcasts_path,
        pending,
        num_workers=duration_probe_workers(cfg, config),
    )
    bucket_seconds, max_bucket_duration = duration_bucket_settings(
        args.config_path, cfg, config
    )
    work_plan = prepare_length_bucketed_work_shards(
        podcasts_path,
        PARTIAL_PREFIX,
        pending,
        durations,
        shard_size=load_work_shard_size(args.config_path),
        bucket_seconds=bucket_seconds,
        max_duration=max_bucket_duration,
    )
    del pending
    del durations

    logger.info(
        f"{work_plan.total_items} files need Spectra-0 scoring; "
        f"starting {n_gpus} GPU worker(s) over {work_plan.shard_count} shard(s)."
    )
    processed = mp.Value("i", 0)
    skipped = mp.Value("i", 0)
    errors = mp.Value("i", 0)
    try:
        with PeriodicCsvMerger(
            podcasts_path,
            prefix=PARTIAL_PREFIX,
            value_columns=VALUE_COLUMNS,
            preserve_existing=True,
            **load_csv_settings(args.config_path),
        ):
            mp.spawn(
                run_worker,
                args=(
                    n_gpus,
                    str(work_plan.work_dir),
                    config,
                    args.config_path,
                    processed,
                    skipped,
                    errors,
                ),
                nprocs=n_gpus,
                join=True,
            )
    except KeyboardInterrupt:
        logger.warning("Spectra-0 scoring interrupted; merging partials before exit.")
    except Exception as exc:
        logger.critical(f"Spectra-0 multiprocessing failed: {exc}")
        errors.value += 1

    absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=VALUE_COLUMNS,
        preserve_existing=True,
    )
    _write_status(args, processed.value, skipped.value, errors.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Store raw Spectra-0 bonafide/spoof scores in balalaika.csv"
    )
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    main(parser.parse_args())
