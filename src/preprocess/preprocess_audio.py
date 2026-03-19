import argparse
import torch
import torch.multiprocessing as mp
import torchaudio
from pathlib import Path
from typing import List

import numpy as np
import pyloudnorm as pyln
from loguru import logger
from tqdm import tqdm

import soundfile as sf

from src.utils.utils import load_config, get_audio_paths

torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


def normalize_audio_loudness(
    audio: np.ndarray, 
    rate: int, 
    peak: float = -1.0, 
    loudness: float = -23.0, 
    block_size: float = 0.400
) -> np.ndarray:
    """
    Perform loudness normalization (ITU-R BS.1770-4) on audio data.

    Args:
        audio: audio data array
        rate: sample rate
        peak: peak normalize audio to N dB. Defaults to -1.0.
        loudness: loudness normalize audio to N dB LUFS. Defaults to -23.0.
        block_size: block size for loudness measurement in seconds. Defaults to 0.400 (400 ms).

    Returns:
        loudness normalized audio array
    """
    # Peak normalize audio to [peak] dB
    audio = pyln.normalize.peak(audio, peak)

    # Measure the loudness first
    meter = pyln.Meter(rate, block_size=block_size)  # create BS.1770 meter
    _loudness = meter.integrated_loudness(audio)

    return pyln.normalize.loudness(audio, _loudness, loudness)


def process_audio_file(
    audio_path: str,
    peak: float,
    loudness: float,
    block_size: float
):
    """
    Process a single audio file: normalize loudness and overwrite the original file.

    Args:
        audio_path: Path to the audio file to process
        peak: Peak normalization level in dB
        loudness: Target loudness level in LUFS
        block_size: Block size for loudness measurement in seconds
    """
    try:
        audio, sample_rate = torchaudio.load_with_torchcodec(audio_path)
        audio_np = audio.numpy()

        # torchaudio returns (channels, samples), pyloudnorm expects (samples,) or (samples, channels≤5)
        if audio_np.shape[0] == 1:
            audio_np = audio_np.squeeze(0)
        else:
            audio_np = audio_np.T

        normalized_audio = normalize_audio_loudness(
            audio_np,
            sample_rate,
            peak=peak,
            loudness=loudness,
            block_size=block_size
        )

        # Convert back to torchaudio format (channels, samples)
        if normalized_audio.ndim == 1:
            normalized_audio = normalized_audio[np.newaxis, :]
        else:
            normalized_audio = normalized_audio.T

        torchaudio.save(audio_path, torch.from_numpy(normalized_audio), sample_rate)
        
        logger.debug(f"Normalized: {audio_path}")
        
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_worker(
    rank: int,
    world_size: int,
    all_file_paths: List[Path],
    peak: float,
    loudness: float,
    block_size: float
):
    """
    Worker function for processing files on a specific GPU/process.
    
    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        all_file_paths: List of all audio file paths to process
        peak: Peak normalization level in dB
        loudness: Target loudness level in LUFS
        block_size: Block size for loudness measurement in seconds
    """
    if not all_file_paths:
        return
    
    # Distribute files across workers
    my_files = all_file_paths[rank::world_size]
    
    if not my_files:
        return
    
    logger.info(f"Worker {rank}/{world_size} processing {len(my_files)} files")
    
    for file_path in tqdm(my_files, desc=f"Worker-{rank}", position=rank):
        process_audio_file(
            str(file_path),
            peak,
            loudness,
            block_size
        )


def main(args):
    """
    Main function to normalize audio loudness for all audio files in the specified directory.
    """
    config = load_config(args.config_path, 'preprocess')
    
    podcasts_path = config.get('podcasts_path')
    if not podcasts_path:
        podcasts_path = config.get('podcasts_path', '../../../podcasts')
        logger.warning("Using default podcasts_path")
    
    # Loudness normalization parameters
    peak = config.get('peak', -1.0)
    loudness = config.get('loudness', -23.0)
    block_size = config.get('block_size', 0.400)
    num_workers = config.get('num_workers', 4)
    
    # Use CPU workers (loudness normalization doesn't require GPU)
    # But we can use multiple CPU cores
    num_processes = num_workers
    
    logger.info(f"""
        Running loudness normalization:
        Podcasts path: {podcasts_path}
        Peak normalization: {peak} dB
        Target loudness: {loudness} LUFS
        Block size: {block_size} seconds
        Number of processes: {num_processes}
        """)
    
    # Get all audio files
    audio_paths = get_audio_paths(podcasts_path)
    if not audio_paths:
        logger.info("No audio files found for processing.")
        return
    
    logger.info(f"Found {len(audio_paths)} audio files to process")
    
    # Process files using torch multiprocessing
    if num_processes > 1:
        mp.spawn(
            run_worker,
            args=(num_processes, audio_paths, peak, loudness, block_size),
            nprocs=num_processes,
            join=True
        )
    else:
        # Single process mode
        for file_path in tqdm(audio_paths, desc="Normalizing loudness"):
            process_audio_file(
                str(file_path),
                peak,
                loudness,
                block_size
            )
    
    logger.info("All files have been processed and normalized.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(
        description="Normalize audio loudness (ITU-R BS.1770-4) for all audio files in the dataset."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    
    main(args)
