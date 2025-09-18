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

from src.utils import load_config, get_txt_paths, read_file_content

accentizer = None

def init_process(
    model_name: str,
    device: str
    ) -> None:
    global accentizer

    accentizer = RUAccent()
    accentizer.load(
        omograph_model_size=model_name,
        use_dictionary=True,
        tiny_mode=False,
        device=device
    )


def process_file(path: Path):    
    try:
        new_path = path.with_name(path.stem.replace("_punct", "_accent") + ".txt")

        if new_path.exists():
            return

        text = read_file_content(path)
        
        processed_text = accentizer.process_all(text)
        
        with open(new_path, "w", encoding="utf-8") as f:
            f.write(processed_text)

    except Exception as e:
        logger.error(f"Error processing {path}: {e}")
        raise

def get_valid_txt_paths(path: str) -> List[str]:
    all_punct_paths = get_txt_paths(path, "_punct.txt")
    valid_paths = []

    for punct_path in all_punct_paths:
        accent_path = punct_path.with_name(punct_path.stem.replace("_punct", "_accent") + ".txt")
        if not os.path.exists(accent_path):
            valid_paths.append(punct_path)
    return valid_paths

def main(args):
    config = load_config(args.config_path, 'accent')
    num_workers = args.num_workers if args.num_workers else config.get('num_workers', 4)
    model_name = args.model_name if args.model_name else config.get('model_name', 'turbo3.1')
    podcast_path = args.podcasts_path if args.podcasts_path else config.get('podcasts_path', '../../../balalaika')

    available_gpu_ids = list(range(torch.cuda.device_count()))
    num_gpus = len(available_gpu_ids)

    if num_gpus == 0:
        logger.error("No GPUs available. Exiting.")
        return
    
    logger.info(
        f"""
        Using parms 
        podcast_path:{podcast_path} 
        num_workers:{num_workers} 
        model_name:{model_name} 
        devices:{available_gpu_ids}
        """)

    valid_text_files = get_valid_txt_paths(podcast_path)

    files_for_each_gpu = [[] for _ in range(num_gpus)]
    for i, path in enumerate(valid_text_files):
        gpu_assignment_index = i % num_gpus
        files_for_each_gpu[gpu_assignment_index].append(path)

    logger.info(f"Found {len(valid_text_files)} files to process")

    all_futures = []
    executors = []

    for i, gpu_id in enumerate(available_gpu_ids):
        device_str = f'cuda:{gpu_id}'
        files_for_this_gpu = files_for_each_gpu[i]

        if not files_for_this_gpu:
            continue

        logger.info(f"Creating ProcessPoolExecutor for {device_str} with {num_workers} workers for {len(files_for_this_gpu)} files.")
        
        executor = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_process,
            initargs=(model_name, device_str)
        )
        executors.append(executor)

        for path in files_for_this_gpu:
            future = executor.submit(process_file, path)
            all_futures.append(future)

    logger.info(f"Submitted all {len(all_futures)} tasks across {len(executors)} GPU(s). Waiting for completion...")

    for future in tqdm(as_completed(all_futures), total=len(all_futures), desc="Overall Punctuation Progress"):
        try:
            future.result()
        except Exception as e:
            logger.error(f"A task processing encountered an error (already logged by worker): {e}")

    logger.info("Processing completed")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Accent restoration script.")
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to config"
        )
    parser.add_argument(
        "--podcasts_path",
        type=str,
        help="Path to dataset directory"
        )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of worker processes"
        )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model version"
        )
    
    args = parser.parse_args()
    main(args)