import argparse
import multiprocessing
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import torch
from loguru import logger
from ruaccent import RUAccent
from tqdm import tqdm

from src.utils.logging_setup import setup_logging
from src.utils.runtime_env import runtime_cfg
from src.utils.utils import get_txt_paths, load_config, read_file_content

torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

accentizer = None

def get_providers(cuda_id: int, use_tensorrt: bool = False, config_path=None) -> list:
    if use_tensorrt:
        rt = runtime_cfg(config_path)
        cache_path = os.path.join(str(rt["trt_cache_path"]), f"trt_cache_{cuda_id}")
        os.makedirs(cache_path, exist_ok=True)
        return [
            ("TensorrtExecutionProvider", {
                "device_id": cuda_id,
                "trt_max_workspace_size": int(rt["trt_workspace_bytes"]),
                "trt_fp16_enable": bool(rt["trt_fp16"]),
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cache_path,
            }),
            ("CUDAExecutionProvider", {"device_id": cuda_id}),
        ]
    return [("CUDAExecutionProvider", {"device_id": cuda_id})]


def init_process(model_name: str, cuda_id: int, use_tensorrt: bool, config_path=None) -> None:
    global accentizer

    providers = get_providers(cuda_id, use_tensorrt, config_path)
    
    logger.info(f"Initializing worker on GPU:{cuda_id} (TRT={use_tensorrt})")
    
    accentizer = RUAccent()
    accentizer.load(
        omograph_model_size=model_name, 
        use_dictionary=True, 
        tiny_mode=False, 
        providers=providers
    )


def process_file(path: Path):    
    try:
        new_path = path.with_name(path.stem.replace("_punct", "_accent") + ".txt")

        if new_path.exists():
            return

        text = read_file_content(path)
        if not text or len(text.strip()) == 0:
            return
        
        processed_text = accentizer.process_all(text)
        
        with open(new_path, "w", encoding="utf-8") as f:
            f.write(processed_text)

    except Exception as e:
        logger.error(f"Error processing {path.name}: {e}")

def get_valid_txt_paths(path: str) -> List[Path]:
    all_punct_paths = get_txt_paths(path, "_punct.txt")
    valid_paths = []

    for punct_path in all_punct_paths:
        p = Path(punct_path)
        accent_path = p.with_name(p.stem.replace("_punct", "_accent") + ".txt")
        if not accent_path.exists():
            valid_paths.append(p)
    return valid_paths

def main(args):
    setup_logging("accents", log_dir=args.log_dir)
    config = load_config(args.config_path, 'accent')
    
    num_workers_per_gpu = config.get('num_workers', 1)
    model_name = config.get('model_name', 'turbo3.1')
    podcast_path = config.get('podcasts_path', './data')
    use_tensorrt = config.get('use_tensorrt', False)

    available_gpu_ids = list(range(torch.cuda.device_count()))
    num_gpus = len(available_gpu_ids)

    if num_gpus == 0:
        logger.error("No GPUs found via torch.cuda.device_count().")
        return
    
    logger.info(f"Config loaded. GPUs: {available_gpu_ids}, Workers per GPU: {num_workers_per_gpu}, TRT: {use_tensorrt}")

    valid_text_files = get_valid_txt_paths(podcast_path)
    if not valid_text_files:
        logger.success("All files are already processed (no new _punct.txt found).")
        return

    logger.info(f"Found {len(valid_text_files)} files to process.")
    
    files_per_gpu = [[] for _ in range(num_gpus)]
    for i, file_path in enumerate(valid_text_files):
        files_per_gpu[i % num_gpus].append(file_path)

    all_futures = []
    executors = []

    for i, gpu_id in enumerate(available_gpu_ids):
        gpu_files = files_per_gpu[i]
        if not gpu_files:
            continue

        logger.info(f"Starting {num_workers_per_gpu} workers for GPU:{gpu_id} ({len(gpu_files)} files)")

        executor = ProcessPoolExecutor(
            max_workers=num_workers_per_gpu,
            initializer=init_process,
            initargs=(model_name, gpu_id, use_tensorrt, args.config_path),
        )
        executors.append(executor)

        for path in gpu_files:
            future = executor.submit(process_file, path)
            all_futures.append(future)

    try:
        with tqdm(total=len(all_futures), desc="Accents Restoration") as pbar:
            for future in as_completed(all_futures):
                future.result()
                pbar.update(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Shutting down...")
    finally:
        for e in executors:
            e.shutdown(wait=True)

    logger.success("Accent restoration completed!")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Multi-GPU Accent Restoration")
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")

    args = parser.parse_args()
    main(args)