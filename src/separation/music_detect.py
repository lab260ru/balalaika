"""Music detection filter built on a fine-tuned WavLM head (ONNX runtime).

The WavLM music detector now runs as an ONNX model through onnxruntime
(CUDA EP, or TensorRT EP when ``music_detect.use_tensorrt`` is set), so this
stage no longer imports ``musicdetection`` or ``transformers`` at runtime. The
wavlm-base-plus feature extractor used ``do_normalize=false``, so preprocessing
is just decode + resample + pad + attention mask (see
:mod:`src.utils.datasets.separation`).

In addition to scoring each chunk and deleting clips above the threshold, the
worker records per-file durations into the partial CSV. That lets the rank-0
process emit a stage row in ``filter_summary.csv`` capturing the actual hours of
audio dropped, even though the files themselves are gone.

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
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

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
    MUSICDETECT_SAMPLE_RATE,
    create_music_detect_dataloader,
)
from src.utils.gpu import get_onnx_providers, make_session_options
from src.utils.logging_setup import setup_logging
from src.utils.node_profile import resolve_batch_size
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config
from src.utils.work_shards import (
    claim_work_shard,
    load_work_shard_size,
    mark_work_shard_done,
    prepare_length_bucketed_work_shards,
    read_annotated_work_shard,
)

PARTIAL_PREFIX = "music"
COLUMN = "music_prob"
PARTIAL_FIELDS = ("filepath", "music_prob", "total_duration", "duration_s", "deleted")
VALUE_COLUMNS = [COLUMN, "total_duration"]
MODEL_SAMPLE_RATE = MUSICDETECT_SAMPLE_RATE
MODEL_REPO_ID = "NikiPshg/music_detection_onnx"
MODEL_REPO_FILENAME = "music_detection.onnx"


def ensure_model(model_path: Path, cfg: dict | None = None) -> Path:
    """Return the local ONNX path, downloading it from HF on first use."""
    if model_path.exists():
        return model_path

    cfg = cfg or {}
    repo_id = str(cfg.get("repo_id", MODEL_REPO_ID))
    filename = str(cfg.get("repo_filename", MODEL_REPO_FILENAME))
    logger.info(f"Downloading music-detection ONNX from {repo_id}/{filename}")
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
        raise FileNotFoundError(f"Music-detection ONNX model not found: {model_path}")
    return model_path


def _graph_input_names(model_path: Path) -> List[str]:
    """Real (non-initializer) graph input names, in declaration order."""
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    initializers = {init.name for init in model.graph.initializer}
    return [i.name for i in model.graph.input if i.name not in initializers]


def create_session(
    model_path: Path,
    rank: int,
    cfg: dict,
    config_path: str | None,
):
    """Build the ORT session and resolve the waveform/mask input names.

    Returns ``(session, waveform_input, mask_input, output_name)``. The two
    inputs are matched by declared dtype (float -> waveform, int -> mask) so the
    feed dict never depends on graph input ordering.
    """
    model_path = ensure_model(model_path, cfg)
    options = make_session_options(config_path=config_path)
    use_tensorrt = bool(cfg.get("use_tensorrt", False))
    providers = get_onnx_providers(
        rank,
        use_tensorrt=use_tensorrt,
        config_path=config_path,
    )
    if use_tensorrt:
        # Force FP32 and pin one dynamic-shape profile. FP16 is unusable for
        # this model (NaNs / classification flips, verified during conversion),
        # and runtime.trt_fp16 is True for other stages, so we override it here.
        # The audio length is dynamic, so the profile spans batch 1..bs over a
        # wide sample range; shapes outside it fall back to the CUDA EP instead
        # of triggering a fresh multi-minute engine build per distinct shape.
        batch_size = resolve_batch_size("music_detect", cfg.get("bs"), 8)
        input_names = _graph_input_names(model_path)
        min_samples = int(MODEL_SAMPLE_RATE * float(cfg.get("trt_min_seconds", 1.0)))
        opt_samples = int(MODEL_SAMPLE_RATE * float(cfg.get("trt_opt_seconds", 15.0)))
        max_samples = max(
            opt_samples,
            int(MODEL_SAMPLE_RATE * float(cfg.get("trt_max_seconds", 30.0))),
        )
        min_shapes = ",".join(f"{name}:1x{min_samples}" for name in input_names)
        opt_shapes = ",".join(
            f"{name}:{batch_size}x{opt_samples}" for name in input_names
        )
        max_shapes = ",".join(
            f"{name}:{batch_size}x{max_samples}" for name in input_names
        )
        patched = []
        for provider in providers:
            name, opts = provider if isinstance(provider, tuple) else (provider, {})
            opts = dict(opts)
            if name == "TensorrtExecutionProvider":
                opts.update(
                    {
                        "trt_fp16_enable": False,
                        "trt_profile_min_shapes": min_shapes,
                        "trt_profile_opt_shapes": opt_shapes,
                        "trt_profile_max_shapes": max_shapes,
                        "trt_timing_cache_enable": True,
                    }
                )
            patched.append((name, opts))
        providers = patched
    logger.info(f"[cuda:{rank}] music-detection ONNX providers: {providers}")
    session = ort.InferenceSession(str(model_path), options, providers=providers)

    waveform_input = mask_input = None
    for graph_input in session.get_inputs():
        if "float" in graph_input.type:
            waveform_input = graph_input.name
        else:
            mask_input = graph_input.name
    if waveform_input is None or mask_input is None:
        raise RuntimeError(
            f"Could not resolve waveform/mask inputs from {model_path}: "
            f"{[(i.name, i.type) for i in session.get_inputs()]}"
        )
    output_name = session.get_outputs()[0].name
    logger.info(
        f"[cuda:{rank}] music-detection IO: waveform={waveform_input}, "
        f"mask={mask_input}, output={output_name}"
    )
    return session, waveform_input, mask_input, output_name


def _process_files(
    rank: int,
    files: List[str],
    session: ort.InferenceSession,
    waveform_input: str,
    mask_input: str,
    output_name: str,
    config: dict,
    writer: PartialCsvWriter,
    already_done: Set[str],
    processed_counter,
    skipped_counter,
    errors_counter,
    audio_lengths: Optional[Dict[str, float]] = None,
) -> int:
    cfg = config.get("music_detect", {})
    threshold = cfg.get("threshold", 0.5)
    batch_size = resolve_batch_size("music_detect", cfg.get("bs"), 8)
    num_workers = int(cfg.get("num_workers", 4))
    prefetch_factor = int(cfg.get("prefetch_factor", 2))

    pending_files = []
    for path in files:
        resolved = resolve_path(path)
        if resolved in already_done:
            skipped_counter.value += 1
            continue
        pending_files.append(path)

    if not pending_files:
        return 0

    dataloader = create_music_detect_dataloader(
        pending_files,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sample_rate=MODEL_SAMPLE_RATE,
    )
    prefetch_batches = num_workers * prefetch_factor if num_workers > 0 else 0
    logger.debug(
        f"perf dataloader_config stage=music_detect rank={rank} "
        f"batch_size={batch_size} workers={num_workers} "
        f"prefetch_factor={prefetch_factor} prefetch_batches={prefetch_batches} "
        f"items={len(pending_files)}"
    )

    deleted_count = 0
    wait_started_at = time.perf_counter()
    for batch_idx, (
        paths,
        input_values,
        attention_mask,
        _lengths,
        load_errors,
    ) in enumerate(tqdm(dataloader, desc=f"MusicDetect-{rank}", position=rank)):
        received_at = time.perf_counter()
        logger.debug(
            f"perf dataloader_wait stage=music_detect rank={rank} "
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
            probs = session.run(
                [output_name],
                {
                    waveform_input: input_values.numpy().astype(np.float32, copy=False),
                    mask_input: attention_mask.numpy().astype(np.int32, copy=False),
                },
            )[0]
            probs = np.asarray(probs, dtype=np.float32).reshape(-1)
            logger.debug(
                f"perf model=music_detect event=inference rank={rank} "
                f"batch={batch_idx} seconds={time.perf_counter() - started_at:.6f} "
                f"items={len(paths)} frames={int(input_values.shape[-1])}"
            )
        except Exception as exc:
            logger.error(f"Music-detection batch failed on worker {rank}: {exc}")
            errors_counter.value += len(paths)
            wait_started_at = time.perf_counter()
            continue

        for path_str, prob in zip(paths, probs.tolist()):
            resolved = resolve_path(path_str)
            if resolved in already_done:
                skipped_counter.value += 1
                continue

            prob_val = round(float(prob), 6)
            duration_s = (
                float(audio_lengths.get(str(path_str), 0.0)) if audio_lengths else 0.0
            )
            if duration_s <= 0:
                duration_s = safe_audio_duration(path_str)

            deleted = False
            if prob_val > threshold:
                try:
                    os.remove(path_str)
                    deleted_count += 1
                    deleted = True
                except OSError as exc:
                    logger.warning(f"Could not delete {path_str}: {exc}")
                    errors_counter.value += 1

            writer.write(
                {
                    "filepath": resolved,
                    "music_prob": prob_val,
                    "total_duration": round(duration_s, 4),
                    "duration_s": round(duration_s, 4),
                    "deleted": deleted,
                }
            )
            already_done.add(resolved)
            processed_counter.value += 1
        wait_started_at = time.perf_counter()

    return deleted_count


def run_worker(
    rank: int,
    world_size: int,
    work_dir: str,
    config: dict,
    config_path: str | None,
    processed_counter,
    skipped_counter,
    errors_counter,
):
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    cfg = config.get("music_detect", {})
    podcasts_path = Path(config.get("podcasts_path", "."))
    model_path = Path(cfg.get("onnx_path", "./models/music_detection.onnx"))

    try:
        session, waveform_input, mask_input, output_name = create_session(
            model_path, rank, cfg, config_path
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
                items = read_annotated_work_shard(shard_path)
                shard_files = [path for path, _ in items]
                shard_lengths: Optional[Dict[str, float]] = None
                if items and all(note for _, note in items):
                    try:
                        shard_lengths = {path: float(note) for path, note in items}
                    except ValueError:
                        shard_lengths = None
                claimed += 1
                logger.info(
                    f"[cuda:{rank}] Processing {len(shard_files)} files from {shard_path.name}..."
                )
                deleted_total += _process_files(
                    rank,
                    shard_files,
                    session,
                    waveform_input,
                    mask_input,
                    output_name,
                    config,
                    writer,
                    already_done,
                    processed_counter,
                    skipped_counter,
                    errors_counter,
                    audio_lengths=shard_lengths,
                )
                mark_work_shard_done(shard_path)

        logger.success(
            f"[cuda:{rank}] Done. Claimed {claimed} shard(s), deleted {deleted_total} files."
        )

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
    cfg = config.get("music_detect", {})
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

    # Fetch the ONNX once up front so worker spawns don't race on the download.
    ensure_model(Path(cfg.get("onnx_path", "./models/music_detection.onnx")), cfg)

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
        logger.success(
            "All audio files already have a music_prob entry. Skipping computation."
        )
        audit = audit_from_filter_partials(leftover_partials)
        if audit["files_in"] == 0:
            audit["files_in"] = len(audio_paths)
            audit["files_out"] = len(audio_paths)
        record_stage_summary(
            podcasts_path=podcasts_path,
            stage="music_detect",
            files_in=audit["files_in"],
            files_out=audit["files_out"],
            hours_in=audit["hours_in"],
            hours_out=audit["hours_out"],
            params={
                "threshold": cfg.get("threshold", 0.5),
                "deleted": audit["files_deleted"],
            },
        )
        return

    # One duration pass for the whole stage: durations feed both the
    # length-bucketed shards (less padding waste per batch) and the per-file
    # CSV rows (carried into each shard as a tab annotation).
    durations = ensure_audio_durations(
        podcasts_path,
        pending,
        num_workers=duration_probe_workers(cfg, config),
    )
    bucket_seconds, max_bucket_duration = duration_bucket_settings(
        args.config_path, cfg, config
    )
    annotations = {p: str(float(durations.get(p, 0.0) or 0.0)) for p in pending}
    work_plan = prepare_length_bucketed_work_shards(
        podcasts_path,
        PARTIAL_PREFIX,
        pending,
        durations,
        shard_size=load_work_shard_size(args.config_path),
        bucket_seconds=bucket_seconds,
        max_duration=max_bucket_duration,
        annotations=annotations,
    )
    del pending
    del durations
    del annotations

    logger.info(
        f"{work_plan.total_items} files still need a music_prob; "
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
            value_columns=VALUE_COLUMNS,
            drop_missing_files=True,
            preserve_existing=True,
            **csv_settings,
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
        logger.warning(
            "Music detection stage interrupted; merging partials before exit."
        )

    # 4) Merge whatever the workers managed to produce.
    new_partials, _ = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=VALUE_COLUMNS,
        drop_missing_files=True,
        preserve_existing=True,
    )

    partial_frames = [
        df
        for df in (leftover_partials, new_partials)
        if df is not None and not df.empty
    ]
    combined = (
        pd.concat(partial_frames, ignore_index=True)
        if partial_frames
        else pd.DataFrame()
    )

    audit = audit_from_filter_partials(combined)

    record_stage_summary(
        podcasts_path=podcasts_path,
        stage="music_detect",
        files_in=audit["files_in"],
        files_out=audit["files_out"],
        hours_in=audit["hours_in"],
        hours_out=audit["hours_out"],
        params={
            "threshold": cfg.get("threshold", 0.5),
            "deleted": audit["files_deleted"],
        },
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
    parser.add_argument(
        "--log_dir", type=str, default=None, help="Override log directory"
    )
    main(parser.parse_args())
