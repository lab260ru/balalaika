import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import torch
from loguru import logger
from safetensors import safe_open
from tqdm import tqdm

from src.separation.model import WavLMForMusicDetection
from src.utils import get_audio_paths, load_config


def process_files_on_device(
    audio_paths: List[Path],
    checkpoint_path: str,
    device: str,
    threshold: float,
    batch_size: int
) -> int:
    """
    Processes a list of audio files on a single device (GPU).

    This function initializes the model, gets predictions for all files,
    and deletes those that exceed the threshold.

    Returns:
        The number of deleted files.
    """
    logger.info(f"Starting process on device {device} for {len(audio_paths)} files...")
    deleted_count = 0
    
    try:
        # 1. Initialize the model, specifying the device and batch size
        model = WavLMForMusicDetection(batch_size=batch_size, device=device)

        # 2. Load the weights
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
        model.load_state_dict(state_dict)

        # 3. Get probabilities for all files in a single call
        # Convert Path objects to strings, as the model expects
        string_paths = [str(p) for p in audio_paths]
        probs = model.predict_proba(string_paths)

        # 4. Check and delete files
        for path, prob in zip(audio_paths, probs):
            score = prob.item()
            if score > threshold:
                logger.info(f"Deleting file: {path} (score: {score:.4f})")
                os.remove(path)
                deleted_count += 1

        logger.info(f"Process on {device} finished. Deleted {deleted_count} files.")
        return deleted_count

    except Exception as e:
        logger.error(f"Error in process on device {device}: {e}")
        return 0  # Return 0 deleted files in case of an error


def main(args):
    """
    Main function to orchestrate the multiprocessing of audio files.
    """
    config = load_config(args.config_path, 'separation')

    # Get parameters from the config
    batch_size = config.get('batch_size', 32)
    checkpoint_path = config.get('checkpoint_path', '/path/to/your/model.safetensors')
    audio_data_path = config.get('podcasts_path', '/path/to/your/audio')
    threshold = config.get('threshold', 0.5)

    all_audio_files = get_audio_paths(audio_data_path)
    if not all_audio_files:
        logger.warning(f"Audio files not found in {audio_data_path}")
        return

    if not torch.cuda.is_available():
        logger.error("No available GPUs. Exiting.")
        return

    available_gpu_ids = list(range(torch.cuda.device_count()))
    num_gpus = len(available_gpu_ids)
    
    # We now use 1 worker per GPU
    total_workers = num_gpus

    logger.info(
        f"""
        Starting audio processing:
        - Data path: {audio_data_path}
        - Checkpoint: {checkpoint_path}
        - Number of GPUs: {num_gpus} (IDs: {available_gpu_ids})
        - Total workers: {total_workers}
        - Total files to process: {len(all_audio_files)}
        - Deletion threshold: {threshold}
        - Batch size per GPU: {batch_size}
        """
    )

    # Distribute the files among the available GPUs
    files_per_gpu = {gpu_id: [] for gpu_id in available_gpu_ids}
    for i, path in enumerate(all_audio_files):
        gpu_id = available_gpu_ids[i % num_gpus]
        files_per_gpu[gpu_id].append(path)

    futures = []
    # Create a single process pool for all GPUs
    with ProcessPoolExecutor(max_workers=total_workers) as executor:
        for gpu_id, files_for_this_gpu in files_per_gpu.items():
            if not files_for_this_gpu:
                continue

            device_str = f'cuda:{gpu_id}'
            logger.info(f"Submitting task for {device_str} with {len(files_for_this_gpu)} files.")
            
            # Submit one large task for each GPU
            future = executor.submit(
                process_files_on_device,
                files_for_this_gpu,
                checkpoint_path,
                device_str,
                threshold,
                batch_size
            )
            futures.append(future)

    logger.info(f"All {len(futures)} tasks have been submitted for processing. Awaiting completion...")

    total_deleted_count = 0
    # The progress bar will show the completion of tasks per GPU
    with tqdm(total=len(futures), desc="Processing on GPUs") as pbar:
        for future in as_completed(futures):
            # Collect the number of deleted files from each process
            total_deleted_count += future.result()
            pbar.update(1)
    
    logger.info(f"Processing complete. Total files deleted: {total_deleted_count}.")


if __name__ == "__main__":
    # It's recommended to set the start method to 'spawn' for CUDA compatibility
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Music detection and removal script.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()
    main(args)