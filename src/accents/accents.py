"""Stage 9 — RUAccent stress restoration on ``*_punct.txt`` sidecars.

Fast path (default): one :class:`~src.accents.fast_accent.FastRUAccent` per
worker, files submitted in slabs (chunked, not one Future per file), each slab
processed with cross-file ONNX batching.  ``accent.use_fast_accent: false``
restores the stock per-file ``RUAccent.process_all`` flow bit-for-bit.
"""
import argparse
import multiprocessing
import os
import tempfile
from pathlib import Path

from loguru import logger

from src.accents.fast_accent import FastRUAccent, capped_onnx_threads
from src.utils.gpu import apply_torch_perf_defaults, get_onnx_providers
from src.utils.logging_setup import setup_logging
from src.utils.parallel import run_per_gpu_pool_chunked
from src.utils.sidecars import pending_sidecar_chain, replace_in_stem
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config, read_file_content

apply_torch_perf_defaults()

accentizer = None

# Per-worker count of fast-path fallbacks (process_batch -> stock per-file
# path). The pool-chunked runner has no end-of-worker hook, so the running
# total is surfaced in a greppable line at each fallback rather than once at
# shutdown; a non-zero count means this worker mixed fast and stock outputs.
_FAST_PATH_FALLBACKS = 0


def init_process(
    model_name: str,
    cuda_id: int,
    use_tensorrt: bool,
    device: str,
    intra_op_threads: int,
    fast_knobs: tuple,
    config_path=None,
) -> None:
    global accentizer
    if device == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        providers = get_onnx_providers(
            cuda_id, use_tensorrt=use_tensorrt, config_path=config_path
        )
    batch_sentences, memo_accent, lazy_rule_engine = fast_knobs
    logger.info(
        f"Initializing accent worker (device={device} id={cuda_id} TRT={use_tensorrt}, "
        f"intra_op_threads={intra_op_threads or 'default'})"
    )
    accentizer = FastRUAccent(
        batch_sentences=batch_sentences,
        memo_accent=memo_accent,
        lazy_rule_engine=lazy_rule_engine,
    )
    with capped_onnx_threads(intra_op_threads):
        accentizer.load(
            omograph_model_size=model_name,
            use_dictionary=True,
            tiny_mode=False,
            providers=providers,
        )


def _write_accent(punct_path: Path, accented: str) -> None:
    accent_path = replace_in_stem(punct_path, "_punct", "_accent")
    fd, tmp = tempfile.mkstemp(dir=accent_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(accented)
        os.replace(tmp, accent_path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def process_chunk(chunk) -> list:
    """Process a slab of ``*_punct.txt`` paths with cross-file ONNX batching.

    Reads every not-yet-done file, runs them through ``process_batch`` as one
    batched group, and writes each output atomically.  Returns a list of
    ``{"item", "reason"}`` failures so one bad file never kills its slab-mates.
    """
    failures = []
    paths = []
    texts = []
    for punct_path in chunk:
        punct_path = Path(punct_path)
        try:
            accent_path = replace_in_stem(punct_path, "_punct", "_accent")
            if accent_path.exists():
                continue
            text = read_file_content(punct_path)
            if not text or not text.strip():
                continue
            paths.append(punct_path)
            texts.append(text)
        except Exception as exc:
            logger.error(f"Error reading {punct_path.name}: {exc}")
            failures.append({"item": str(punct_path), "reason": str(exc)})

    if not texts:
        return failures

    try:
        outputs = accentizer.process_batch(texts)
    except Exception as exc:
        # A batch-level failure must not silently drop the slab: fall back to
        # per-file processing so good files still complete and only the
        # offending file is reported.
        global _FAST_PATH_FALLBACKS
        _FAST_PATH_FALLBACKS += 1
        logger.warning(
            f"Accent batch failed ({exc}); falling back to per-file. "
            f"fast-path fallbacks: {_FAST_PATH_FALLBACKS}"
        )
        outputs = None

    if outputs is None:
        for punct_path, text in zip(paths, texts):
            try:
                _write_accent(punct_path, accentizer.process_all(text))
            except Exception as exc:
                logger.error(f"Error processing {punct_path.name}: {exc}")
                failures.append({"item": str(punct_path), "reason": str(exc)})
        return failures

    for punct_path, accented in zip(paths, outputs):
        try:
            _write_accent(punct_path, accented)
        except Exception as exc:
            logger.error(f"Error writing {punct_path.name}: {exc}")
            failures.append({"item": str(punct_path), "reason": str(exc)})
    return failures


def main(args):
    setup_logging("accents", log_dir=args.log_dir)
    config = load_config(args.config_path, "accent")

    num_workers_per_gpu = config.get("num_workers", 1)
    model_name = config.get("model_name", "turbo3.1")
    podcast_path = config.get("podcasts_path", "./data")
    use_tensorrt = config.get("use_tensorrt", False)
    device = str(config.get("device", "cuda")).lower()
    chunk_size = int(config.get("batch_size", 64))
    intra_op_threads = int(config.get("intra_op_threads", 4))
    use_fast = config.get("use_fast_accent", True)
    fast_knobs = (
        bool(config.get("batch_sentences", True)) and use_fast,
        bool(config.get("memo_accent", True)) and use_fast,
        bool(config.get("lazy_rule_engine", True)) and use_fast,
    )

    pending_files = pending_sidecar_chain(
        podcast_path,
        in_suffix="_punct.txt",
        out_derive=lambda p: replace_in_stem(p, "_punct", "_accent"),
        config_path=args.config_path,
    )
    if not pending_files:
        logger.success("No pending _punct.txt files; accents already up to date.")
        return
    # Path order keeps sidecar reads directory-clustered on HDD datasets.
    pending_files.sort()

    logger.info(
        f"Found {len(pending_files)} files to process "
        f"(chunk_size={chunk_size}, fast={use_fast})."
    )

    pool_kwargs = {}
    if device == "cpu":
        # One CPU pool; the id is a shard slot, never a CUDA device — keeps the
        # stage alive on CPU-only nodes (pool's default gpu_ids would refuse).
        pool_kwargs["gpu_ids"] = [0]

    error_count, error_details = run_per_gpu_pool_chunked(
        pending_files,
        work_fn=process_chunk,
        initializer=init_process,
        init_args_factory=lambda gpu_id: (
            model_name,
            gpu_id,
            use_tensorrt,
            device,
            intra_op_threads,
            fast_knobs,
            args.config_path,
        ),
        chunk_size=chunk_size,
        num_workers_per_gpu=num_workers_per_gpu,
        desc="Accents",
        **pool_kwargs,
    )
    write_stage_status(
        stage=10,
        stage_name="accents",
        log_dir=args.log_dir or "./logs",
        processed=len(pending_files) - error_count,
        skipped=0,
        errors=error_count,
        error_details=error_details,
    )

    logger.success("Accent restoration completed!")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Multi-GPU accent restoration via RUAccent.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    main(parser.parse_args())
