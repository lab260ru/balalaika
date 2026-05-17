"""Stage 9 — TryIParu G2P phonemization on ``*_rover.txt`` sidecars."""
import argparse
import multiprocessing
from pathlib import Path

from loguru import logger
from tryiparu import G2PModel

from src.utils.gpu import apply_torch_perf_defaults
from src.utils.logging_setup import setup_logging
from src.utils.parallel import run_per_gpu_pool
from src.utils.sidecars import pending_sidecar_chain
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config, read_file_content

apply_torch_perf_defaults()

g2p_model = None


def init_process(device_str: str) -> None:
    global g2p_model
    g2p_model = G2PModel(load_dataset=True, device=device_str)


def process_text(text_path: Path) -> None:
    text_path = Path(text_path)
    output_path = text_path.with_name(f"{text_path.stem}_phonemes.txt")
    if output_path.exists():
        return

    text = read_file_content(text_path)
    phonemes = g2p_model(text)
    output_path.write_text(" ".join(phonemes), encoding="utf-8")


def main(args):
    setup_logging("phonemizer", log_dir=args.log_dir)
    config = load_config(args.config_path, "phonemizer")

    num_workers_per_gpu = config.get("num_workers", 4)
    src_path = config.get("podcasts_path", "../../../podcasts")

    pending_files = pending_sidecar_chain(
        src_path,
        in_suffix="_rover.txt",
        out_derive=lambda p: p.with_name(f"{p.stem}_phonemes.txt"),
    )
    if not pending_files:
        logger.success("No pending _rover.txt files; phonemes already up to date.")
        return

    logger.info(f"Found {len(pending_files)} text files to process.")

    error_count, error_details = run_per_gpu_pool(
        pending_files,
        work_fn=process_text,
        initializer=init_process,
        init_args_factory=lambda gpu_id: (f"cuda:{gpu_id}",),
        num_workers_per_gpu=num_workers_per_gpu,
        desc="Phonemizer",
    )
    write_stage_status(
        stage=9,
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
