"""ONNX Runtime / TensorRT MossFormer2_SE_48K denoising stage.

The stage processes audio files in place and tracks progress in
``balalaika.csv`` through the ``denoised`` column. Inputs are decoded as mono
48 kHz int16 batches, padded for the dynamic ONNX profile, sent to ONNX Runtime,
and trimmed back to the original decoded length before saving.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import onnxruntime as ort
import torch
import torch.multiprocessing as mp
import torchaudio
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
from src.utils.datasets.denoising import (
    DENOISING_SAMPLE_RATE,
    create_denoising_dataloader,
)
from src.utils.gpu import apply_torch_perf_defaults, get_onnx_providers
from src.utils.logging_setup import setup_logging
from src.utils.node_profile import resolve_batch_size
from src.utils.parallel import run_per_gpu_processes
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config
from src.utils.work_shards import (
    claim_work_shard,
    load_work_shard_size,
    mark_work_shard_done,
    prepare_length_bucketed_work_shards,
    read_work_shard,
)

apply_torch_perf_defaults()


PARTIAL_PREFIX = "denoising"
PROCESSED_COLUMN = "denoised"
PARTIAL_FIELDS = ("filepath", PROCESSED_COLUMN)
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_SAMPLE_RATE = DENOISING_SAMPLE_RATE
MODEL_PAD_TO_MULTIPLE = 384
MODEL_PAD_MODE = "noise"
MODEL_MAX_PADDED_LEN = 960_000
MODEL_TRT_MIN_SHAPE = "1x1x8000"
MODEL_REPO_FILENAME = "MossFormer2_SE_48K_dynamic.onnx"
DEFAULT_ONNX_PATH = "./models/MossFormer2_SE_48K_dynamic.onnx"
ORT_THREADS = 4


def resolve_model_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def ensure_model(model_path: Path, cfg: Dict) -> None:
    if model_path.exists():
        return

    repo_id = str(cfg.get("hf_repo_id") or "").strip()
    filename = str(cfg.get("hf_filename") or MODEL_REPO_FILENAME).strip()
    if not repo_id:
        raise FileNotFoundError(
            f"Denoising ONNX model not found: {model_path}. "
            "Set denoising.hf_repo_id after uploading the ONNX model to Hugging Face, "
            "or place the model at denoising.onnx_path."
        )

    try:
        import huggingface_hub
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the denoising ONNX model. "
            "Install it or place the model at denoising.onnx_path."
        ) from exc

    logger.info(f"Downloading denoising ONNX from Hugging Face: {repo_id}/{filename}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(model_path.parent),
        )
    )

    if downloaded != model_path:
        downloaded.replace(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Denoising ONNX model not found: {model_path}")



def add_denoising_trt_profile_options(
    providers,
    input_name: str,
    batch_size: int,
):
    patched = []
    for provider in providers:
        if isinstance(provider, tuple):
            provider_name, options = provider
            options = dict(options)
        else:
            provider_name, options = provider, {}

        if provider_name == "TensorrtExecutionProvider":
            options.update(
                {
                    "trt_profile_min_shapes": f"{input_name}:{MODEL_TRT_MIN_SHAPE}",
                    "trt_profile_opt_shapes": f"{input_name}:{batch_size}x1x{MODEL_SAMPLE_RATE}",
                    "trt_profile_max_shapes": f"{input_name}:{batch_size}x1x{MODEL_MAX_PADDED_LEN}",
                    "trt_timing_cache_enable": True,
                    "trt_detailed_build_log": True,
                }
            )
        patched.append((provider_name, options))
    return patched


def _onnx_first_input_name(model_path: Path) -> str:
    """First graph input name, without instantiating an InferenceSession."""
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    initializers = {init.name for init in model.graph.initializer}
    for graph_input in model.graph.input:
        if graph_input.name not in initializers:
            return graph_input.name
    raise ValueError(f"No graph inputs found in {model_path}")


def create_session(
    model_path: Path,
    rank: int,
    cfg: Dict,
    config_path: str | None,
    batch_size: int,
) -> ort.InferenceSession:
    ensure_model(model_path, cfg)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.inter_op_num_threads = ORT_THREADS
    sess_options.intra_op_num_threads = ORT_THREADS
    sess_options.add_session_config_entry("session.set_denormal_as_zero", "1")

    # Read the input name from graph metadata instead of building a throwaway
    # CPU InferenceSession (which loaded and optimized the full model once per
    # worker just to learn one string).
    input_name = _onnx_first_input_name(model_path)

    use_tensorrt = bool(cfg.get("use_tensorrt", True))
    providers = get_onnx_providers(rank, use_tensorrt=use_tensorrt, config_path=config_path)
    providers = add_denoising_trt_profile_options(providers, input_name, batch_size)

    logger.info(f"[cuda:{rank}] Denoising ONNX providers: {providers}")
    return ort.InferenceSession(str(model_path), sess_options, providers=providers)


def _process_files(
    rank: int,
    files: List[str],
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    config: dict,
    writer: PartialCsvWriter,
    already_done: Set[str],
    processed_counter,
    skipped_counter,
    errors_counter,
) -> None:
    batch_size = resolve_batch_size("denoising", config.get("batch_size"), 2)
    loader_workers = int(config.get("num_workers", 0))
    prefetch_factor = int(config.get("prefetch_factor", 2))

    pending_files = []
    for path in files:
        resolved = resolve_path(path)
        if resolved in already_done:
            skipped_counter.value += 1
            continue
        pending_files.append(path)

    if not pending_files:
        return

    dataloader = create_denoising_dataloader(
        pending_files,
        batch_size=batch_size,
        num_workers=loader_workers,
        prefetch_factor=prefetch_factor,
        sample_rate=MODEL_SAMPLE_RATE,
        pad_to_multiple=MODEL_PAD_TO_MULTIPLE,
        pad_mode=MODEL_PAD_MODE,
        max_padded_len=MODEL_MAX_PADDED_LEN,
    )
    prefetch_batches = loader_workers * prefetch_factor if loader_workers > 0 else 0
    logger.debug(
        f"perf dataloader_config stage=denoising rank={rank} "
        f"batch_size={batch_size} workers={loader_workers} "
        f"prefetch_factor={prefetch_factor} prefetch_batches={prefetch_batches} "
        f"items={len(pending_files)}"
    )

    batch_wait_started_at = time.perf_counter()
    for batch_idx, (paths, batch, lengths, errors) in enumerate(tqdm(
        dataloader, desc=f"Denoising-{rank}", position=rank
    )):
        batch_received_at = time.perf_counter()
        logger.debug(
            f"perf dataloader_wait stage=denoising rank={rank} "
            f"batch={batch_idx} seconds={batch_received_at - batch_wait_started_at:.6f} "
            f"items={len(paths)}"
        )
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
            batch_wait_started_at = time.perf_counter()
            continue

        try:
            input_np = batch[valid_indices].numpy().astype(np.float32, copy=False)
            inference_started_at = time.perf_counter()
            denoised = session.run([output_name], {input_name: input_np})[0]
            logger.debug(
                f"perf model=denoising event=inference rank={rank} "
                f"batch={batch_idx} seconds={time.perf_counter() - inference_started_at:.6f} "
                f"items={len(valid_indices)} frames={int(input_np.shape[-1])}"
            )
            denoised = np.asarray(denoised)
            if denoised.ndim == 1:
                denoised = denoised[np.newaxis, :]
        except Exception as exc:
            logger.error(f"ONNX denoising batch failed on worker {rank}: {exc}")
            errors_counter.value += len(valid_indices)
            batch_wait_started_at = time.perf_counter()
            continue

        for out_index, (path_str, length) in enumerate(zip(valid_paths, valid_lengths)):
            resolved = resolve_path(path_str)
            if resolved in already_done:
                skipped_counter.value += 1
                continue
            try:
                enhanced = denoised[out_index]
                if enhanced.ndim == 2:
                    enhanced = enhanced[0]
                if enhanced.shape[-1] < length:
                    enhanced = np.pad(enhanced, (0, length - enhanced.shape[-1]))
                enhanced = np.clip(enhanced[:length], -32768.0, 32767.0)
                enhanced_tensor = torch.from_numpy(
                    enhanced.astype(np.float32, copy=False) / 32768.0
                ).unsqueeze(0)
                save_started_at = time.perf_counter()
                torchaudio.save_with_torchcodec(
                    str(path_str),
                    enhanced_tensor,
                    MODEL_SAMPLE_RATE,
                )
                logger.debug(
                    f"perf audio_save stage=denoising rank={rank} "
                    f"seconds={time.perf_counter() - save_started_at:.6f} "
                    f"path={path_str} sample_rate={MODEL_SAMPLE_RATE} frames={int(length)}"
                )
                write_started_at = time.perf_counter()
                writer.write(
                    {
                        "filepath": resolved,
                        PROCESSED_COLUMN: True,
                    }
                )
                logger.debug(
                    f"perf partial_write stage=denoising rank={rank} "
                    f"seconds={time.perf_counter() - write_started_at:.6f} path={resolved}"
                )
                already_done.add(resolved)
                processed_counter.value += 1
            except Exception as exc:
                logger.error(f"Failed to save denoised audio {path_str}: {exc}")
                errors_counter.value += 1
        batch_wait_started_at = time.perf_counter()


def run_worker(
    rank: int,
    world_size: int,
    work_dir: str,
    config: dict,
    config_path: str | None,
    podcasts_path: Path,
    processed_counter,
    skipped_counter,
    errors_counter,
):
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    batch_size = resolve_batch_size("denoising", config.get("batch_size"), 2)
    model_path = resolve_model_path(str(config.get("onnx_path", DEFAULT_ONNX_PATH)))

    session = create_session(
        model_path=model_path,
        rank=rank,
        cfg=config,
        config_path=config_path,
        batch_size=batch_size,
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logger.info(
        f"Worker {rank}/{world_size}: claiming shards, batch={batch_size}, "
        f"sample_rate={MODEL_SAMPLE_RATE}, providers={session.get_providers()}, "
        f"input={input_name}{session.get_inputs()[0].shape}, "
        f"output={output_name}{session.get_outputs()[0].shape}"
    )

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
            logger.info(f"Worker {rank}: processing {len(shard_files)} files from {shard_path.name}")
            _process_files(
                rank,
                shard_files,
                session,
                input_name,
                output_name,
                config,
                writer,
                already_done,
                processed_counter,
                skipped_counter,
                errors_counter,
            )
            mark_work_shard_done(shard_path)

    logger.info(f"Worker {rank} finished after {claimed} claimed shard(s).")

def main():
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

    model_path = resolve_model_path(str(config.get("onnx_path", DEFAULT_ONNX_PATH)))
    ensure_model(model_path, config)

    audio_paths = discover_audio_paths(podcasts_path, config_path=args.config_path)
    if not audio_paths:
        logger.warning("No audio files found.")
        return

    ensure_main_csv(podcasts_path, audio_paths=audio_paths)

    _, absorbed = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=[PROCESSED_COLUMN],
        bootstrap_audio_paths=audio_paths,
        preserve_existing=True,
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

    shard_size = load_work_shard_size(args.config_path)
    duration_workers = duration_probe_workers(config)
    durations = ensure_audio_durations(
        podcasts_path,
        pending,
        num_workers=duration_workers,
    )
    bucket_seconds, max_bucket_duration = duration_bucket_settings(
        args.config_path,
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
        f"Running ONNX denoising for {work_plan.total_items} files with {num_processes} process(es) "
        f"({available_gpus} GPU(s) visible), model={model_path}, "
        f"shards={work_plan.shard_count}."
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
            preserve_existing=True,
            **csv_settings,
        ):
            worker_errors, _ = run_per_gpu_processes(
                run_worker,
                num_gpus=num_processes,
                args=(str(work_plan.work_dir), config, args.config_path, podcasts_path, processed, skipped, errors),
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
        preserve_existing=True,
    )

    write_stage_status(
        stage=11,
        stage_name="denoising",
        log_dir=args.log_dir or "./logs",
        processed=processed.value,
        skipped=skipped.value,
        errors=errors.value,
    )

    logger.info("Denoising stage complete.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
