import argparse
import os
import torch
import torch.multiprocessing as mp
import torchaudio
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger
from tqdm import tqdm

from src.utils.utils import load_config, get_audio_paths


def calculate_crest_factor(audio: np.ndarray) -> float:
    """
    Calculate crest factor (peak / rms) for audio signal.
    
    Args:
        audio: audio data array (1D numpy array)
    
    Returns:
        crest factor value (peak / rms)
    """
    # Calculate peak (maximum absolute value)
    peak = np.max(np.abs(audio))
    
    # Calculate RMS (root mean square)
    rms = np.sqrt(np.mean(audio ** 2))
    
    # Avoid division by zero
    if rms == 0:
        return float('inf')
    
    # Crest factor = peak / rms
    crest_factor = peak / rms
    
    return crest_factor


def process_audio_file(
    audio_path: str,
    crest_threshold: float
) -> bool:
    """
    Process a single audio file: check crest factor and delete if it exceeds threshold.
    
    Args:
        audio_path: Path to the audio file to process
        crest_threshold: Maximum allowed crest factor (peak / rms)
    
    Returns:
        True if file was deleted, False otherwise
    """
    try:
        # Read audio file using torchaudio
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        
        # Convert to numpy array (handle mono/stereo)
        if audio_tensor.shape[0] > 1:
            # For stereo, use the channel with higher energy or average
            audio = audio_tensor.mean(dim=0).numpy()
        else:
            audio = audio_tensor.squeeze(0).numpy()
        
        # Calculate crest factor
        crest_factor = calculate_crest_factor(audio)
        
        # Check if crest factor exceeds threshold
        if crest_factor > crest_threshold:
            # Delete the file
            os.remove(audio_path)
            logger.debug(f"Deleted {audio_path} (crest factor: {crest_factor:.2f} > {crest_threshold})")
            return True
        else:
            logger.debug(f"Kept {audio_path} (crest factor: {crest_factor:.2f} <= {crest_threshold})")
            return False
        
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        return False
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_worker(
    rank: int,
    world_size: int,
    all_file_paths: List[Path],
    crest_threshold: float
):
    """
    Worker function for processing files on a specific process.
    
    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        all_file_paths: List of all audio file paths to process
        crest_threshold: Maximum allowed crest factor
    """
    if not all_file_paths:
        return
    
    # Distribute files across workers
    my_files = all_file_paths[rank::world_size]
    
    if not my_files:
        return
    
    deleted_count = 0
    logger.info(f"Worker {rank}/{world_size} processing {len(my_files)} files")
    
    for file_path in tqdm(my_files, desc=f"Worker-{rank}", position=rank):
        if process_audio_file(str(file_path), crest_threshold):
            deleted_count += 1
    
    logger.info(f"Worker {rank}/{world_size} deleted {deleted_count} files")


def main(args):
    """
    Main function to remove audio files that exceed crest factor threshold.
    """
    config = load_config(args.config_path, 'preprocess')
    
    podcasts_path = config.get('podcasts_path')
    if not podcasts_path:
        podcasts_path = config.get('podcasts_path', '../../../podcasts')
        logger.warning("Using default podcasts_path")
    
    # Crest factor threshold parameter
    crest_threshold = config.get('crest_treshold', 10.0)  # Default 10.0 if not specified
    num_workers = config.get('num_workers', 4)
    
    # Use CPU workers
    num_processes = num_workers
    
    logger.info(f"""
        Running crest factor removal:
        Podcasts path: {podcasts_path}
        Crest factor threshold: {crest_threshold}
        Files with crest factor (peak/rms) > {crest_threshold} will be deleted
        Number of processes: {num_processes}
        """)
    
    # Get all audio files
    audio_paths = get_audio_paths(podcasts_path)
    if not audio_paths:
        logger.info("No audio files found for processing.")
        return
    
    logger.info(f"Found {len(audio_paths)} audio files to check")
    
    # Process files using torch multiprocessing
    if num_processes > 1:
        mp.spawn(
            run_worker,
            args=(num_processes, audio_paths, crest_threshold),
            nprocs=num_processes,
            join=True
        )
    else:
        # Single process mode
        deleted_count = 0
        for file_path in tqdm(audio_paths, desc="Checking crest factor"):
            if process_audio_file(str(file_path), crest_threshold):
                deleted_count += 1
        
        logger.info(f"Deleted {deleted_count} files that exceeded crest factor threshold")
    
    logger.info("Crest factor check completed.")


if __name__ == "__main__":
    torchaudio.set_audio_backend('soundfile')
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(
        description="Remove audio files that exceed crest factor threshold (peak/rms > threshold)."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    
    main(args)
