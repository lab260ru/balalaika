"""Stage 10 — TryIParu G2P phonemization on ``*_rover.txt`` sidecars."""
import argparse
import multiprocessing
import os
import tempfile
from pathlib import Path

from loguru import logger

from src.phonemizer.fast_g2p import FastG2P
from src.utils.gpu import apply_torch_perf_defaults
from src.utils.logging_setup import setup_logging
from src.utils.parallel import run_per_gpu_pool
from src.utils.sidecars import pending_sidecar_chain
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config, read_file_content

apply_torch_perf_defaults()

g2p_model = None


def init_process(device_str: str, batch_size: int, oov_cache_path: str | None) -> None:
    global g2p_model
    g2p_model = FastG2P(
        device=device_str,
        batch_size=batch_size,
        oov_cache_path=oov_cache_path,
    )


def process_text(text_path: Path) -> None:
    text_path = Path(text_path)
    output_path = text_path.with_name(f"{text_path.stem}_phonemes.txt")
    if output_path.exists():
        return

    text = read_file_content(text_path)
    phonemes = g2p_model(text)
    # Atomic: a worker killed mid-write must not leave a truncated sidecar
    # that the bare-existence resume check would then skip forever.
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(" ".join(phonemes))
        os.replace(tmp, output_path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(args):
    setup_logging("phonemizer", log_dir=args.log_dir)
    config = load_config(args.config_path, "phonemizer")

    num_workers_per_gpu = config.get("num_workers", 4)
    src_path = config.get("podcasts_path", "../../../podcasts")
    device = config.get("device", "cuda")
    batch_size = config.get("g2p_batch_size", 64)
    oov_cache_path = config.get("oov_cache_path", "cache/g2p_oov_cache.pkl") or None

    pending_files = pending_sidecar_chain(
        src_path,
        in_suffix="_rover.txt",
        out_derive=lambda p: p.with_name(f"{p.stem}_phonemes.txt"),
        config_path=args.config_path,
    )
    if not pending_files:
        logger.success("No pending _rover.txt files; phonemes already up to date.")
        return

    logger.info(f"Found {len(pending_files)} text files to process.")

    pool_kwargs = {}
    if device == "cpu":
        # One CPU pool of num_workers; the id is a shard slot, never a CUDA
        # device — also keeps the stage alive on CPU-only nodes where the
        # pool's default gpu_ids=range(device_count()) would refuse to run.
        pool_kwargs["gpu_ids"] = [0]

    error_count, error_details = run_per_gpu_pool(
        pending_files,
        work_fn=process_text,
        initializer=init_process,
        init_args_factory=lambda gpu_id: (
            "cpu" if device == "cpu" else f"cuda:{gpu_id}",
            batch_size,
            oov_cache_path,
        ),
        num_workers_per_gpu=num_workers_per_gpu,
        desc="Phonemizer",
        **pool_kwargs,
    )
    write_stage_status(
        stage=10,
        stage_name="phonemizer",
        log_dir=args.log_dir or "./logs",
        processed=len(pending_files) - error_count,
        skipped=0,
        errors=error_count,
        error_details=error_details,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        description="Parallel text→phoneme conversion (multi-GPU)."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    main(parser.parse_args())
