"""Stage 8 — RUAccent stress restoration on ``*_punct.txt`` sidecars."""
import argparse
import multiprocessing
from pathlib import Path

from loguru import logger
from ruaccent import RUAccent

from src.utils.gpu import apply_torch_perf_defaults, get_onnx_providers
from src.utils.logging_setup import setup_logging
from src.utils.parallel import run_per_gpu_pool
from src.utils.sidecars import pending_sidecar_chain, replace_in_stem
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config, read_file_content

apply_torch_perf_defaults()

accentizer = None


def init_process(model_name: str, cuda_id: int, use_tensorrt: bool, config_path=None) -> None:
    global accentizer
    providers = get_onnx_providers(cuda_id, use_tensorrt=use_tensorrt, config_path=config_path)
    logger.info(f"Initializing accent worker on GPU:{cuda_id} (TRT={use_tensorrt})")
    accentizer = RUAccent()
    accentizer.load(
        omograph_model_size=model_name,
        use_dictionary=True,
        tiny_mode=False,
        providers=providers,
    )


def process_file(punct_path: Path) -> None:
    try:
        accent_path = replace_in_stem(punct_path, "_punct", "_accent")
        if accent_path.exists():
            return

        text = read_file_content(punct_path)
        if not text or not text.strip():
            return

        accent_path.write_text(accentizer.process_all(text), encoding="utf-8")
    except Exception as exc:
        logger.error(f"Error processing {punct_path.name}: {exc}")


def main(args):
    setup_logging("accents", log_dir=args.log_dir)
    config = load_config(args.config_path, "accent")

    num_workers_per_gpu = config.get("num_workers", 1)
    model_name = config.get("model_name", "turbo3.1")
    podcast_path = config.get("podcasts_path", "./data")
    use_tensorrt = config.get("use_tensorrt", False)

    pending_files = pending_sidecar_chain(
        podcast_path,
        in_suffix="_punct.txt",
        out_derive=lambda p: replace_in_stem(p, "_punct", "_accent"),
        config_path=args.config_path,
    )
    if not pending_files:
        logger.success("No pending _punct.txt files; accents already up to date.")
        return

    logger.info(f"Found {len(pending_files)} files to process.")

    error_count, error_details = run_per_gpu_pool(
        pending_files,
        work_fn=process_file,
        initializer=init_process,
        init_args_factory=lambda gpu_id: (model_name, gpu_id, use_tensorrt, args.config_path),
        num_workers_per_gpu=num_workers_per_gpu,
        desc="Accents",
    )
    write_stage_status(
        stage=8,
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
