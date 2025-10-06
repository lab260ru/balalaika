import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import torch
import yaml
from loguru import logger
from tqdm import tqdm
from safetensors import safe_open

from src.separation.musicdetector import WavLMForMusicDetection
from src.utils import get_audio_paths, load_config


def process_and_delete_chunk(
    files: List[Path],
    device_str: str,
    model_path: str,
    threshold: float,
    batch_size: int
) -> int:
    """
    Initializes a model on a specific device, processes a chunk of files in batches,
    and deletes those with music. Designed to run in a separate process.
    """
    deleted_count = 0
    try:
        model = WavLMForMusicDetection(device=device_str, batch_size=batch_size)
        
        with safe_open(model_path, framework="pt") as f:
            model.load_state_dict({k: f.get_tensor(k) for k in f.keys()})

        logger.info(f"Worker (PID: {os.getpid()}) initialized on {device_str} for {len(files)} files.")
    except Exception as e:
        logger.error(f"Failed to initialize worker on {device_str} with model {model_path}: {e}")
        return 0

    for i in tqdm(range(0, len(files), batch_size)):
        batch_paths = files[i:i + batch_size]
        if not batch_paths:
            continue

        try:
            audio_paths_str = [str(p) for p in batch_paths]
            probabilities = model.predict_proba(audio_paths_str)

            for path, prob in zip(batch_paths, probabilities):
                prob_item = prob.item()
                if prob_item > threshold:
                    try:
                        os.remove(path)
                        deleted_count += 1
                    except OSError as e:
                        logger.error(f"Error deleting file {path} on {device_str}: {e}")

        except Exception as e:
            batch_start = batch_paths[0].name if batch_paths else "N/A"
            logger.error(f"Error processing batch starting with {batch_start} on {device_str}: {e}")

    return deleted_count

def main(args):
    """
    Main function to distribute audio processing tasks across multiple GPUs and workers.
    """
    config = load_config(args.config_path, 'separation')
    music_detect_config = config.get('music_detect', {})
    
    src_path = config.get('podcasts_path')
    n_workers_per_gpu = music_detect_config.get('num_workers', 1)
    threshold = music_detect_config.get('threshold', 0.5)
    model_path = music_detect_config.get('music_detect_model')
    batch_size = music_detect_config.get('bs', 16)

    if not all([src_path, model_path]):
        logger.error("Config missing required parameters: 'podcasts_path' or 'music_detect_model'.")
        return

    available_gpu_ids = list(range(torch.cuda.device_count()))
    if not available_gpu_ids:
        logger.error("No CUDA-enabled GPUs found. Exiting.")
        return

    all_audio_paths = get_audio_paths(src_path)
    if not all_audio_paths:
        logger.warning(f"No audio files found in '{src_path}'.")
        return
    logger.info(f"Found {len(all_audio_paths)} audio files to process.")

    num_devices = len(available_gpu_ids)
    total_workers = n_workers_per_gpu * num_devices
    logger.info(f"Found {num_devices} GPUs. Launching {n_workers_per_gpu} workers per GPU for a total of {total_workers} workers.")

    files_for_each_worker = [[] for _ in range(total_workers)]
    for i, path in enumerate(all_audio_paths):
        files_for_each_worker[i % total_workers].append(path)

    tasks = []
    for i in range(total_workers):
        worker_files = files_for_each_worker[i]
        if not worker_files:
            continue
        
        device_str = f'cuda:{available_gpu_ids[i % num_devices]}'
        
        tasks.append(
            (worker_files, device_str, model_path, threshold, batch_size)
        )

    total_deleted_count = 0
    with ProcessPoolExecutor(max_workers=total_workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = [executor.submit(process_and_delete_chunk, *task) for task in tasks]
        
        for future in as_completed(futures):
            try:
                total_deleted_count += future.result()
            except Exception as e:
                logger.error(f"A worker process task failed: {e}")

    logger.success(f"Total files deleted: {total_deleted_count}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Parallel music detection and deletion based on a config file.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML configuration file.")
    parsed_args = parser.parse_args()
    main(parsed_args)