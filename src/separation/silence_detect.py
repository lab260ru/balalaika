import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torchaudio
from loguru import logger
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from tqdm import tqdm

from src.utils.utils import get_audio_paths, load_config

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


silence_detect_model = None
vad_device = None


def init_worker(device_str: str):
    """Initialize VAD model for the worker process."""
    global silence_detect_model, vad_device
    logger.info(f"Initializing VAD model on {device_str}...")
    
    try:
        if 'cuda' in device_str:
            torch.cuda.set_device(device_str)
        silence_detect_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        silence_detect_model.to(device_str)
        vad_device = torch.device(device_str)
        logger.info(f"Worker initialized successfully for silence detection on {device_str}.")
    except Exception as e:
        logger.error(f"Failed to initialize worker on {device_str}: {e}")
        silence_detect_model = None
        vad_device = None


def calculate_silence_metrics(path: Path, vad_threshold: float = 0.5) -> Dict:
    """
    Calculate silence percentage and maximum silence duration using Silero VAD.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate for VAD
        
    Returns:
        Dict with silence_percent and max_silence_duration
    """
    global silence_detect_model, vad_device
    
    if silence_detect_model is None:
        logger.error(f"VAD model is not initialized. Skipping {path.name}.")
        return None
    
    try:
        # Load audio
        audio = read_audio(str(path))
        if vad_device is not None:
            audio = audio.to(vad_device)
        
        speech_timestamps = get_speech_timestamps(
            audio,
            silence_detect_model,
            threshold=vad_threshold,
            return_seconds=True
        )
        # Calculate total audio duration in seconds
        total_duration = audio.shape[-1] / 16_000  # 16_000 sample rate for silero vad 
        
        # Calculate total speech duration
        total_speech_duration = sum(
            segment['end'] - segment['start'] 
            for segment in speech_timestamps
        )
        
        # Calculate silence duration
        total_silence_duration = total_duration - total_speech_duration
        
        # Calculate silence percentage
        silence_percent = (total_silence_duration / total_duration * 100) if total_duration > 0 else 0
        
        # Calculate maximum silence duration
        max_silence_duration = 0.0
        if len(speech_timestamps) > 1:
            # Find gaps between speech segments
            silence_gaps = []
            for i in range(len(speech_timestamps) - 1):
                gap_start = speech_timestamps[i]['end']
                gap_end = speech_timestamps[i + 1]['start']
                gap_duration = gap_end - gap_start
                silence_gaps.append(gap_duration)
            
            # Include silence at the beginning
            if speech_timestamps[0]['start'] > 0:
                silence_gaps.append(speech_timestamps[0]['start'])
            
            # Include silence at the end
            if speech_timestamps[-1]['end'] < total_duration:
                silence_gaps.append(total_duration - speech_timestamps[-1]['end'])
            
            max_silence_duration = max(silence_gaps) if silence_gaps else 0.0
        elif len(speech_timestamps) == 1:
            # Only one speech segment, check before and after
            max_silence_duration = max(
                speech_timestamps[0]['start'],
                total_duration - speech_timestamps[0]['end']
            )
        else:
            # No speech detected, entire file is silence
            max_silence_duration = total_duration
        
        return {
            'filepath': str(path),
            'silence_percent': round(silence_percent, 2),
            'max_silence_duration': round(max_silence_duration, 2)
        }
        
    except Exception as e:
        logger.error(f"Failed to process {path.name}: {e}")
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def process_file(path: Path, vad_threshold: float = 0.5) -> Dict:
    """Wrapper function for processing a single file."""
    return calculate_silence_metrics(path=path, vad_threshold=vad_threshold)


def get_unprocessed_audio_paths(podcasts_path: str, result_csv_path: Path) -> List[Path]:
    """Get list of audio paths that haven't been processed yet."""
    all_audio_paths = set(get_audio_paths(podcasts_path))
    processed_audio_paths = []
    
    if result_csv_path.exists():
        logger.info(f"Resuming from existing results file: {result_csv_path}")
        df = pd.read_csv(result_csv_path)
        if 'silence_percent' not in df.columns:
            logger.warning("Column 'silence_percent' not found in results file. Treating all entries as unprocessed.")
            processed_audio_paths = set()
        else:
            processed_mask = df['silence_percent'].notna()
            processed_audio_paths = set(df.loc[processed_mask, 'filepath'].astype(str).tolist())
    else:
        return list(all_audio_paths)
    
    unprocessed_paths = all_audio_paths - processed_audio_paths
    return list(unprocessed_paths)


def main(args):
    """Main function to detect silence metrics for all audio files."""
    config = load_config(args.config_path, 'separation')
    podcasts_path = config.get('podcasts_path', './data')

    silence_config = config.get('silence_detect', {})
    num_workers_per_gpu = silence_config.get('num_workers', 1)
    vad_threshold = silence_config.get('vad_threshold', 0.5)

    result_csv_path = Path(podcasts_path) / 'balalaika.csv'
    available_gpu_ids = list(range(torch.cuda.device_count()))
    
    # Use CPU if no GPU available
    if not available_gpu_ids:
        logger.warning("No GPUs available. Using CPU for silence detection.")
        available_gpu_ids = ['cpu']

    logger.info(f"""
                Podcasts path: {podcasts_path}
                Workers per device: {num_workers_per_gpu}
                VAD threshold: {vad_threshold}
                Devices detected: {len(available_gpu_ids)}
                """)

    all_audio_paths = get_unprocessed_audio_paths(podcasts_path, result_csv_path)

    if not all_audio_paths:
        logger.success("All audio files have already been processed.")
        return
    
    logger.info(f"Found {len(all_audio_paths)} new audio files to process.")

    num_devices = len(available_gpu_ids)
    files_for_each_device = [[] for _ in range(num_devices)]
    for i, path in enumerate(all_audio_paths):
        device_assignment_index = i % num_devices
        files_for_each_device[device_assignment_index].append(path)

    all_futures = []
    executors = []

    for i, device_id in enumerate(available_gpu_ids):
        device_str = f'cuda:{device_id}' if device_id != 'cpu' else 'cpu'
        files_for_this_device = files_for_each_device[i]

        if not files_for_this_device:
            continue

        logger.info(f"Creating ProcessPoolExecutor for {device_str} with {num_workers_per_gpu} workers for {len(files_for_this_device)} files.")
        
        initializer_fn = partial(init_worker, device_str=device_str)

        executor = ProcessPoolExecutor(
            max_workers=num_workers_per_gpu,
            initializer=initializer_fn,
            mp_context=multiprocessing.get_context('spawn')
        )
        executors.append(executor)

        for path in files_for_this_device:
            future = executor.submit(process_file, path, vad_threshold=vad_threshold)
            all_futures.append(future)

    results = []
    for future in tqdm(as_completed(all_futures), total=len(all_futures), desc="Processing audio files"):
        try:
            result = future.result()
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"A task encountered an error: {e}")

    for executor in executors:
        executor.shutdown(wait=True)

    if results:
        try:
            new_results_df = pd.DataFrame(results)
            
            if result_csv_path.exists():
                main_df = pd.read_csv(result_csv_path)

                main_df = main_df.set_index('filepath')
                new_results_df = new_results_df.set_index('filepath')

                updated_df = new_results_df.combine_first(main_df)

                updated_df = updated_df.reset_index()

                original_cols = main_df.reset_index().columns.tolist()
                new_cols = [col for col in updated_df.columns if col not in original_cols]
                final_order = original_cols + new_cols
                final_order_unique = list(dict.fromkeys(final_order))
                updated_df = updated_df[final_order_unique]

                if 'Unnamed: 0' in updated_df.columns:
                    updated_df = updated_df.drop('Unnamed: 0', axis=1)

                updated_df.to_csv(result_csv_path, index=False)
                logger.success(f"Processing complete. Updated/added silence metrics for {len(results)} rows in {result_csv_path}")
            else:
                new_results_df.to_csv(result_csv_path, index=False)
                logger.success(f"Created new results file with {len(results)} rows in {result_csv_path}")
        
        except Exception as e:
            logger.error(f"An error occurred while updating the CSV file: {e}")
    else:
        logger.warning("Processing finished, but no new results were generated.")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Detect silence metrics (silence percentage and max silence duration) using Silero VAD.")
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the main YAML configuration file."
    )

    args = parser.parse_args()
    main(args)
