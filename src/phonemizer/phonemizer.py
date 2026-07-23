"""Stage 11 — TryIParu G2P phonemization on the ``rover`` text of each chunk's
``<stem>.json`` (writes the ``rover_phonemes`` key)."""
import argparse
import multiprocessing
from pathlib import Path

from loguru import logger

from src.phonemizer.fast_g2p import FastG2P
from src.utils.chunk_json import (
    chunk_json_path,
    get_field,
    pending_chunks,
    read_chunk_json,
    update_chunk_json,
)
from src.utils.gpu import apply_torch_perf_defaults
from src.utils.logging_setup import setup_logging
from src.utils.parallel import run_per_gpu_pool_chunked
from src.utils.stage_status import last_line, write_stage_status
from src.utils.utils import load_config

apply_torch_perf_defaults()

g2p_model = None


def init_process(device_str: str, batch_size: int, oov_cache_path: str | None) -> None:
    global g2p_model
    g2p_model = FastG2P(
        device=device_str,
        batch_size=batch_size,
        oov_cache_path=oov_cache_path,
    )


def _process_one(audio_path: Path) -> None:
    audio_path = Path(audio_path)
    data = read_chunk_json(chunk_json_path(audio_path))
    if get_field(data, "rover_phonemes") is not None:
        return

    text = get_field(data, "rover")
    if not text:
        return
    phonemes = g2p_model(text)
    # update_chunk_json is an atomic read-modify-write, so a worker killed
    # mid-write never leaves a partial JSON the resume check would skip forever.
    update_chunk_json(audio_path, {"rover_phonemes": " ".join(phonemes)})


def process_chunk(chunk) -> list:
    """Run G2P over a slab of audio paths (reads ``rover`` from each chunk JSON).

    The per-file G2P call (``FastG2P.__call__``) already batches OOV decode
    across each text's unique words and shares its dict/OOV caches across the
    whole worker, so the only change here vs the old one-file-per-Future loop is
    that files arrive in slabs — O(N) Futures and per-file IPC pickling at 2M
    files collapse to O(N/chunk_size).  Per-file fault isolation is preserved:
    one bad file is reported, its slab-mates still complete."""
    failures = []
    for audio_path in chunk:
        try:
            _process_one(audio_path)
        except Exception as exc:
            logger.error(f"Error processing {Path(audio_path).name}: {exc}")
            failures.append({"item": str(audio_path), "reason": last_line(exc)})
    return failures


def main(args):
    setup_logging("phonemizer", log_dir=args.log_dir)
    config = load_config(args.config_path, "phonemizer")

    num_workers_per_gpu = config.get("num_workers", 4)
    src_path = config.get("podcasts_path", "../../../podcasts")
    device = config.get("device", "cuda")
    batch_size = config.get("g2p_batch_size", 64)
    oov_cache_path = config.get("oov_cache_path", "cache/g2p_oov_cache.pkl") or None
    # Files per work-shard slab (submit-loop chunking; does not change G2P math).
    chunk_size = int(config.get("submit_chunk_size", 256))

    pending_files = pending_chunks(
        src_path,
        out_field="rover_phonemes",
        in_field="rover",
        config_path=args.config_path,
    )
    if not pending_files:
        logger.success("No chunks with rover text need phonemes; up to date.")
        return
    # Path order keeps chunk-JSON reads directory-clustered on HDD datasets.
    pending_files.sort()

    logger.info(f"Found {len(pending_files)} chunks to process.")

    pool_kwargs = {}
    if device == "cpu":
        # One CPU pool of num_workers; the id is a shard slot, never a CUDA
        # device — also keeps the stage alive on CPU-only nodes where the
        # pool's default gpu_ids=range(device_count()) would refuse to run.
        pool_kwargs["gpu_ids"] = [0]

    error_count, error_details = run_per_gpu_pool_chunked(
        pending_files,
        work_fn=process_chunk,
        initializer=init_process,
        init_args_factory=lambda gpu_id: (
            "cpu" if device == "cpu" else f"cuda:{gpu_id}",
            batch_size,
            oov_cache_path,
        ),
        chunk_size=chunk_size,
        num_workers_per_gpu=num_workers_per_gpu,
        desc="Phonemizer",
        **pool_kwargs,
    )
    write_stage_status(
        stage=11,
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
