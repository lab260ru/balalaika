import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import gigaam 
import torch
import torchaudio
from loguru import logger
from tqdm import tqdm

from src.utils import get_audio_paths, load_config

model = None


def init_process(
    model_name: str,
    device_str: str 
):
    global model
    model = gigaam.load_model(model_name, device=device_str)


def make_txt(
    path: Path
):
    
    text = model.transcribe(str(path))
    text_path = path.with_name(f"{path.stem}_giga.txt")

    if os.path.exists(text_path):
        return

    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)

def get_valid_audio_paths(src_path: str) -> List[Path]:
    all_audio_paths = get_audio_paths(src_path)

    valid_paths = []
    for audio_path in all_audio_paths:
        giga_path = audio_path.with_name(audio_path.stem + "_giga.txt")
        if not giga_path.exists():
            valid_paths.append(audio_path)
    
    return valid_paths

def main(args):
    config = load_config(args.config_path, 'transcription')

    model_name = args.model_name if args.model_name else config.get('model_name', 'rnnt')
    num_workers_per_gpu = args.num_workers if args.num_workers else config.get('num_workers', 4)
    src_path = args.podcasts_path if args.podcasts_path else config.get('podcasts_path', '../../../podcasts')

    all_audio_paths = get_valid_audio_paths(src_path)
    logger.info(f"Found {len(all_audio_paths)} audio files to process.")

    available_gpu_ids = list(range(torch.cuda.device_count()))
    num_gpus = len(available_gpu_ids)

    logger.info(
        f"""
        Starting transcription with parameters:
        Source Path: {src_path}
        Model Name: {model_name}
        Number of GPUs: {num_gpus} (IDs: {available_gpu_ids})
        Workers per GPU: {num_workers_per_gpu}
        Total Worker Processes: {num_gpus * num_workers_per_gpu}
        """
    )

    files_for_each_gpu = [[] for _ in range(num_gpus)]
    for i, path in enumerate(all_audio_paths):
        gpu_assignment_index = i % num_gpus
        files_for_each_gpu[gpu_assignment_index].append(path)

    all_futures = []
    executors = []

    for i, gpu_id in enumerate(available_gpu_ids):
        device_str = f'cuda:{gpu_id}'
        files_for_this_gpu = files_for_each_gpu[i]

        if not files_for_this_gpu:
            continue

        logger.info(f"Creating ProcessPoolExecutor for {device_str} with {num_workers_per_gpu} workers for {len(files_for_this_gpu)} files.")
        
        executor = ProcessPoolExecutor(
            max_workers=num_workers_per_gpu,
            initializer=init_process,
            initargs=(model_name, device_str)
        )
        executors.append(executor)

        for path in files_for_this_gpu:
            future = executor.submit(make_txt, path)
            all_futures.append(future)

    for future in tqdm(as_completed(all_futures), total=len(all_futures), desc="Overall Transcription Progress"):
        try:
            future.result() 
        except Exception as e:
            logger.error(f"A task processing encountered an error: {e}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    torchaudio.set_audio_backend('soundfile')

    parser = argparse.ArgumentParser(
        description="Transcribe audio files in parallel using multiple GPUs."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--podcasts_path",
        type=str,
        help="Path to the directory containing audio files (e.g., MP3s)."
    )
    parser.add_argument(
        "--num_workers", 
        type=int,
        help="Number of worker processes per GPU for parallel processing."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model to use for transcription (e.g., 'rnnt', 'ctc')."
    )

    args = parser.parse_args()
    main(args)