"""Stage 7 — RUPunct punctuation restoration on ``*_rover.txt`` sidecars."""
import argparse
import multiprocessing
from pathlib import Path

from loguru import logger
from transformers import AutoTokenizer, pipeline

from src.utils.gpu import apply_torch_perf_defaults
from src.utils.logging_setup import setup_logging
from src.utils.parallel import run_per_gpu_pool
from src.utils.sidecars import pending_audio_to_sidecar
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config, process_token, read_file_content

apply_torch_perf_defaults()

model = None


def init_process(model_name: str, device: str) -> None:
    global model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        strip_accents=False,
        add_prefix_space=True,
    )
    model = pipeline(
        "ner",
        model=model_name,
        tokenizer=tokenizer,
        aggregation_strategy="first",
        device=device,
    )


def make_punct_txt(rover_path: Path) -> None:
    rover_path = Path(rover_path)
    punct_path = rover_path.with_name(rover_path.name.replace("_rover.txt", "_punct.txt"))
    if punct_path.exists():
        return

    src_text = read_file_content(rover_path)
    if not src_text:
        return

    preds = model(src_text)
    output = " ".join(
        process_token(item["word"].strip(), item["entity_group"]) for item in preds
    ).strip()
    punct_path.write_text(output, encoding="utf-8")


def main(args):
    setup_logging("punctuation", log_dir=args.log_dir)
    config = load_config(args.config_path, "punctuation")

    num_workers_per_gpu = config.get("num_workers", 4)
    model_name = config.get("model_name", "RUPunct/RUPunct_big")
    podcasts_path = config.get("podcasts_path", "../../../balalaika")

    pending_files = pending_audio_to_sidecar(
        podcasts_path,
        in_suffix="_rover.txt",
        out_suffix="_punct.txt",
    )
    if not pending_files:
        logger.success("No pending _rover.txt files; punctuation already up to date.")
        return

    logger.info(f"Found {len(pending_files)} _rover.txt files needing punctuation.")

    error_count, error_details = run_per_gpu_pool(
        pending_files,
        work_fn=make_punct_txt,
        initializer=init_process,
        init_args_factory=lambda gpu_id: (model_name, f"cuda:{gpu_id}"),
        num_workers_per_gpu=num_workers_per_gpu,
        desc="Punctuation",
    )
    write_stage_status(
        stage=7,
        stage_name="punctuation",
        log_dir=args.log_dir or "./logs",
        processed=len(pending_files) - error_count,
        skipped=0,
        errors=error_count,
        error_details=error_details,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Multi-GPU punctuation restoration via RUPunct.")
    parser.add_argument("--config_path", type=str, help="Path to the configuration file")
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    main(parser.parse_args())
