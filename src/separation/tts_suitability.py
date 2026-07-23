"""TTS-suitability scoring with crash-safe CSV state.

The stage runs the lab260/TTS-Suitability-Classifier ONNX model over 16 kHz
audio and stores simple softmax probabilities in ``balalaika.csv``. The model
is a wav2vec2-300M classifier: each file is layer-normalized, chunked into
10 s windows, and the per-chunk logits are averaged before probabilities are
computed and written. It does not delete audio; filtering is handled by
``src.separation.tts_suitability_filter``.

Inference is per file rather than batched across files: each file yields a
variable number of chunks of differing length and the ONNX graph has no
attention-mask input, so padding clips together would change their logits.
"""

import argparse
import shutil
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

from src.separation.inline_filter import (
    INLINE_PARTIAL_FIELDS,
    TTS,
    resolve_inline,
    write_score_row,
)
from src.utils.audit import record_stage_summary
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
from src.utils.datasets.separation import (
    TTS_SUITABILITY_CHUNK_FRAMES,
    TTS_SUITABILITY_SAMPLE_RATE,
    create_tts_suitability_dataloader,
)
from src.utils.gpu import (
    apply_ort_thread_caps,
    get_onnx_providers,
)
from src.utils.logging_setup import setup_logging
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config
from src.utils.work_shards import (
    claim_work_shard,
    load_work_shard_size,
    mark_work_shard_done,
    prepare_work_shards,
    read_annotated_work_shard,
)

PARTIAL_PREFIX = "tts_suit"
P_TTS_COLUMN = "p_tts"
P_NOT_TTS_COLUMN = "p_not_tts"
PARTIAL_FIELDS = ("filepath", P_NOT_TTS_COLUMN, P_TTS_COLUMN)
VALUE_COLUMNS = [P_NOT_TTS_COLUMN, P_TTS_COLUMN]


def _inline_threshold(config: dict):
    """Inline-delete threshold for stage 7, or ``None`` when score-only."""
    return resolve_inline(
        config.get("tts_suitability", {}), config.get("tts_suitability_filter", {})
    )
MODEL_SAMPLE_RATE = TTS_SUITABILITY_SAMPLE_RATE
MODEL_CHUNK_FRAMES = TTS_SUITABILITY_CHUNK_FRAMES
NOT_TTS_CLASS_INDEX = 0
TTS_CLASS_INDEX = 1
MODEL_REPO_ID = "lab260/TTS-Suitability-Classifier"
MODEL_REPO_FILENAME = "model.onnx"


def mean_class_logits(outputs: np.ndarray) -> tuple[float, float]:
    """Return ``(not_tts, tts)`` raw logits averaged over a file's chunks."""
    scores = np.asarray(outputs, dtype=np.float32)
    if scores.ndim == 1:
        scores = scores[None, :]
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError(
            f"Expected classifier output shaped (chunks, 2), got {scores.shape}"
        )
    mean_logits = scores.mean(axis=0)
    return (
        float(mean_logits[NOT_TTS_CLASS_INDEX]),
        float(mean_logits[TTS_CLASS_INDEX]),
    )


def softmax_pair(not_tts_logit: float, tts_logit: float) -> tuple[float, float]:
    logits = np.asarray([not_tts_logit, tts_logit], dtype=np.float64)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return float(probs[NOT_TTS_CLASS_INDEX]), float(probs[TTS_CLASS_INDEX])


def ensure_model(model_path: Path, cfg: dict | None = None) -> Path:
    if model_path.exists():
        return model_path

    cfg = cfg or {}
    repo_id = str(cfg.get("repo_id", MODEL_REPO_ID))
    filename = str(cfg.get("repo_filename", MODEL_REPO_FILENAME))
    logger.info(f"Downloading TTS-suitability ONNX from {repo_id}/{filename}")
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
        raise FileNotFoundError(f"TTS-suitability ONNX model not found: {model_path}")
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
    # TensorRT off by default: this model scores variable-length, variable-chunk
    # inputs, so a single dynamic-shape profile cannot cover the range and every
    # distinct shape would trigger a fresh multi-minute engine build. The CUDA
    # EP handles variable shapes without rebuilds.
    use_tensorrt = bool(cfg.get("use_tensorrt", False))
    providers = get_onnx_providers(
        rank,
        use_tensorrt=use_tensorrt,
        config_path=config_path,
    )
    logger.info(f"[cuda:{rank}] TTS-suitability ONNX providers: {providers}")
    session = ort.InferenceSession(str(model_path), options, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logger.info(
        f"[cuda:{rank}] TTS-suitability IO: input={input_name}, output={output_name}"
    )
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
    inline_threshold=None,
    audio_lengths=None,
) -> None:
    batch_size = int(cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 4))
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

    dataloader = create_tts_suitability_dataloader(
        pending_files,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sample_rate=MODEL_SAMPLE_RATE,
        chunk_frames=MODEL_CHUNK_FRAMES,
    )
    prefetch_batches = num_workers * prefetch_factor if num_workers > 0 else 0
    logger.debug(
        f"perf dataloader_config stage=tts_suitability rank={rank} "
        f"batch_size={batch_size} workers={num_workers} "
        f"prefetch_factor={prefetch_factor} prefetch_batches={prefetch_batches} "
        f"items={len(pending_files)}"
    )

    wait_started_at = time.perf_counter()
    for batch_idx, (paths, chunk_batches, load_errors) in enumerate(
        tqdm(dataloader, desc=f"TTSSuit-{rank}", position=rank)
    ):
        received_at = time.perf_counter()
        logger.debug(
            f"perf dataloader_wait stage=tts_suitability rank={rank} "
            f"batch={batch_idx} seconds={received_at - wait_started_at:.6f} "
            f"items={len(paths)}"
        )
        for path_str, reason in load_errors:
            logger.error(f"Audio load failed {path_str}: {reason}")
            errors_counter.value += 1
        if not paths:
            wait_started_at = time.perf_counter()
            continue

        for path_str, chunks in zip(paths, chunk_batches):
            resolved = resolve_path(path_str)
            if resolved in already_done:
                skipped_counter.value += 1
                continue
            try:
                started_at = time.perf_counter()
                outputs = session.run(
                    [output_name],
                    {input_name: chunks.numpy().astype(np.float32, copy=False)},
                )[0]
                not_tts_logit, tts_logit = mean_class_logits(outputs)
                p_not_tts, p_tts = softmax_pair(not_tts_logit, tts_logit)
                logger.debug(
                    f"perf model=tts_suitability event=inference rank={rank} "
                    f"batch={batch_idx} seconds={time.perf_counter() - started_at:.6f} "
                    f"chunks={int(chunks.shape[0])} frames={int(chunks.shape[-1])}"
                )
            except Exception as exc:
                logger.error(f"TTS-suitability failed on {path_str} (worker {rank}): {exc}")
                errors_counter.value += 1
                continue

            write_score_row(
                writer,
                stage=TTS,
                resolved_path=resolved,
                audio_path=path_str,
                scores={
                    P_NOT_TTS_COLUMN: float(p_not_tts),
                    P_TTS_COLUMN: float(p_tts),
                },
                inline_threshold=inline_threshold,
                audio_lengths=audio_lengths,
                errors_counter=errors_counter,
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
    cfg = config.get("tts_suitability", {})
    model_path = Path(cfg.get("onnx_path", "./models/tts_suitability.onnx"))
    logger.info(
        f"[cuda:{rank}] TTS-suitability claiming shards, "
        f"batch={int(cfg.get('batch_size', 8))}, chunk_frames={MODEL_CHUNK_FRAMES}"
    )

    try:
        session, input_name, output_name = create_session(
            model_path, rank, cfg, config_path
        )
        inline_threshold = _inline_threshold(config)
        fieldnames = (
            PARTIAL_FIELDS + INLINE_PARTIAL_FIELDS
            if inline_threshold is not None
            else PARTIAL_FIELDS
        )
        claimed = 0
        with PartialCsvWriter(
            podcasts_path, PARTIAL_PREFIX, rank, fieldnames=fieldnames
        ) as writer:
            already_done: Set[str] = writer.already_done()
            while True:
                shard_path = claim_work_shard(work_dir, rank)
                if shard_path is None:
                    break
                items = read_annotated_work_shard(shard_path)
                shard_files = [p for p, _ in items]
                audio_lengths = None
                if inline_threshold is not None and items and all(n for _, n in items):
                    try:
                        audio_lengths = {p: float(n) for p, n in items}
                    except ValueError:
                        audio_lengths = None
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
                    inline_threshold=inline_threshold,
                    audio_lengths=audio_lengths,
                )
                mark_work_shard_done(shard_path)
        logger.info(f"Worker {rank} done after {claimed} claimed shard(s).")
    except Exception as exc:
        logger.exception(f"TTS-suitability worker {rank} failed: {exc}")
        errors_counter.value += 1


def _write_status(args, processed: int, skipped: int, errors: int) -> None:
    write_stage_status(
        stage=7,
        stage_name="tts_suitability",
        log_dir=args.log_dir or "./logs",
        processed=processed,
        skipped=skipped,
        errors=errors,
    )


def main(args):
    setup_logging("tts_suitability", log_dir=args.log_dir)
    mp.set_start_method("spawn", force=True)
    config = load_config(args.config_path, "separation")
    podcasts_path = Path(config.get("podcasts_path", "."))
    cfg = config.get("tts_suitability", {})
    # Score-only unless inline_filter is on; when on, delete not-TTS-margin files
    # in this pass, prune their rows, and emit a filter row (stage 7.5 no-ops).
    inline_threshold = _inline_threshold(config)
    drop_missing = inline_threshold is not None
    value_columns = (
        VALUE_COLUMNS + ["total_duration"] if drop_missing else VALUE_COLUMNS
    )
    if inline_threshold is not None:
        logger.info(
            f"inline_filter active: deleting files with p_not_tts - p_tts > "
            f"{inline_threshold} during scoring."
        )
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
    ensure_model(Path(cfg.get("onnx_path", "./models/tts_suitability.onnx")), cfg)
    leftover_partials, absorbed = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=value_columns,
        drop_missing_files=drop_missing,
        bootstrap_audio_paths=audio_paths,
        preserve_existing=True,
    )
    if absorbed:
        logger.info(f"Absorbed {absorbed} leftover TTS-suitability rows.")

    pending = unprocessed_paths(podcasts_path, P_TTS_COLUMN, audio_paths)
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        logger.success("All audio files already have TTS-suitability scores.")
        _write_status(args, 0, len(audio_paths), 0)
        return

    # This stage does not bucket by length, so it normally needs no durations.
    # In inline mode probe them once (cached in the state) and carry them into
    # the shard so kept rows record hours without a re-probe at delete time.
    annotations = None
    if drop_missing:
        durations = ensure_audio_durations(
            podcasts_path,
            pending,
            num_workers=duration_probe_workers(cfg, config),
        )
        annotations = {p: str(float(durations.get(p, 0.0) or 0.0)) for p in pending}
        del durations
    work_plan = prepare_work_shards(
        podcasts_path,
        PARTIAL_PREFIX,
        pending,
        shard_size=load_work_shard_size(args.config_path),
        annotations=annotations,
    )
    del pending
    del annotations

    logger.info(
        f"{work_plan.total_items} files need TTS-suitability scoring; "
        f"starting {n_gpus} GPU worker(s) over {work_plan.shard_count} shard(s)."
    )
    processed = mp.Value("i", 0)
    skipped = mp.Value("i", 0)
    errors = mp.Value("i", 0)
    try:
        with PeriodicCsvMerger(
            podcasts_path,
            prefix=PARTIAL_PREFIX,
            value_columns=value_columns,
            drop_missing_files=drop_missing,
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
        logger.warning(
            "TTS-suitability scoring interrupted; merging partials before exit."
        )
    except Exception as exc:
        logger.critical(f"TTS-suitability multiprocessing failed: {exc}")
        errors.value += 1

    new_partials, _ = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=value_columns,
        drop_missing_files=drop_missing,
        preserve_existing=True,
    )

    if inline_threshold is not None:
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
            stage="tts_suitability_filter",
            files_in=audit["files_in"],
            files_out=audit["files_out"],
            hours_in=audit["hours_in"],
            hours_out=audit["hours_out"],
            params={"threshold": inline_threshold, "deleted": audit["files_deleted"]},
        )

    _write_status(args, processed.value, skipped.value, errors.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Store raw TTS-suitability not_tts/tts logits in balalaika.csv"
    )
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    main(parser.parse_args())
