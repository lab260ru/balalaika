import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from loguru import logger
from tqdm import tqdm

from src.transcription.transcripton_base import *
from src.utils import get_audio_paths, load_config

model = None
SUPPORTED_TIME_STAMPS = ['giga_ctc_lm', 'tone']

def init_worker(model_name: str, device_str: str, **kwargs):
    """
    Initializes the appropriate ASR model wrapper for each worker process.
    """
    global model
    logger.info(f"Initializing worker for model '{model_name}' on {device_str}...")
    
    try:
        if 'giga' in model_name:
            model = GigaAMWrapper(model_id=model_name, device=device_str, **kwargs)
        elif 'tone' in model_name:
            model = ToneWrapper(model_id=model_name, device=device_str, **kwargs)
        elif 'vosk' in model_name:
            model = VOSKCUDAWrapper(model_id=kwargs.get('vosk_path'), device=device_str, **kwargs)
        else:
            raise ValueError(f"Unknown model type for '{model_name}'")
        logger.info(f"Worker initialized successfully for model '{model_name}' on {device_str}.")
    except Exception as e:
        logger.error(f"Failed to initialize worker for model '{model_name}' on {device_str}: {e}")
        model = None


def process_batch(paths: List[Path], model_name_for_output: str, with_timestamps: bool):
    """
    Transcribes a batch of audio files using the globally initialized model.
    Generates .txt and optionally .tst files.
    """
    global model
    if model is None:
        logger.error(f"Model is not initialized in this worker. Skipping batch starting with {paths[0].name}.")
        return

    txt_paths = [path.with_name(f"{path.stem}_{model_name_for_output}.txt") for path in paths]
    tst_paths = [path.with_name(f"{path.stem}_{model_name_for_output}.tst") for path in paths]
    audio_paths_str = [str(p) for p in paths]

    try:
        if with_timestamps:
            plain_texts, tst_contents = model.transcribe_batch_with_timestamps(audio_paths_str)
        else:
            plain_texts = model.transcribe_batch(audio_paths_str)
            

        # Transcribe
        for txt_path, plain_text in zip(txt_paths, plain_texts):
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(plain_text)

        # Timestamps
        if with_timestamps and tst_contents:
            for tst_path, tst_content in zip(tst_paths, tst_contents):
                if tst_content:
                    with open(tst_path, "w", encoding="utf-8") as f:
                        f.write(tst_content)

    except Exception as e:
        logger.error(f"Failed to process batch starting with {paths[0].name}: {e}")


def get_valid_audio_paths(src_path: str, model_name_for_output: str) -> List[Path]:
    """
    Gets all audio paths and filters out those that have already been transcribed.
    """
    all_audio_paths = get_audio_paths(src_path)
    
    valid_paths = [
        p for p in all_audio_paths 
        if not p.with_name(f"{p.stem}_{model_name_for_output}.txt").exists()
    ]
    return valid_paths


def main(args):
    config = load_config(args.config_path, 'transcription')
    model_names = config.get('model_names', ['giga_rnnt'])
    src_path = config.get('podcasts_path', '.')

    logger.info(f"Starting transcription run for models: {model_names}")

    available_gpu_ids = list(range(torch.cuda.device_count()))
    if not available_gpu_ids:
        logger.warning("No CUDA GPUs detected. Using CPU. This will be slow.")
        raise 
    
    num_devices = len(available_gpu_ids)
    for model_name in model_names:
        logger.info(f"--- Processing model: {model_name} ---")
        model_config = config.get('giga') if 'giga' in model_name else  config.get(model_name)
        num_workers_per_gpu = model_config.get('num_workers')
        batch_size = model_config.get('batch_size')
        model_name_for_output = 'vosk' if 'vosk' in model_name else model_name
        with_timestamps = config.get('with_timestamps')
        timestamps_supported = model_name in SUPPORTED_TIME_STAMPS
        current_with_timestamps = with_timestamps and timestamps_supported
        
        if with_timestamps and not timestamps_supported:
            logger.warning(f"Timestamps requested but not supported for model '{model_name}'. Disabling.")

        all_audio_paths = get_valid_audio_paths(src_path, model_name_for_output)
        
        if not all_audio_paths:
            logger.info(f"No new audio files to process for model '{model_name}'. Skipping.")
            continue
        
        logger.info(f"Found {len(all_audio_paths)} new audio files to process for '{model_name}'.")

        files_for_each_device = [[] for _ in range(num_devices)]
        for i, path in enumerate(all_audio_paths):
            files_for_each_device[i % num_devices].append(path)

        all_futures = []
        executors = []
        
        task_fn = partial(process_batch, model_name_for_output=model_name_for_output, with_timestamps=current_with_timestamps)
        for i, device_id in enumerate(available_gpu_ids):
            device_str = f'cuda:{device_id}' if device_id != 'cpu' else 'cpu'
            files_for_this_device = files_for_each_device[i]

            if not files_for_this_device:
                continue

            logger.info(f"Creating ProcessPoolExecutor for {device_str} with {num_workers_per_gpu} workers for {len(files_for_this_device)} files.")
            
            initializer_fn = partial(init_worker, model_name, device_str, **model_config)

            executor = ProcessPoolExecutor(
                max_workers=num_workers_per_gpu,
                initializer=initializer_fn,
                mp_context=multiprocessing.get_context('spawn')
            )
            executors.append(executor)

            for j in range(0, len(files_for_this_device), batch_size):
                batch = files_for_this_device[j:j + batch_size]
                if batch:
                    future = executor.submit(task_fn, batch)
                    all_futures.append(future)

        for future in tqdm(as_completed(all_futures), total=len(all_futures), desc=f"Transcribing ({model_name})"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"A task encountered an error: {e}")

        for executor in executors:
            executor.shutdown(wait=True)
        
        logger.info(f"Finished processing model: {model_name}")

    logger.info(f"Starting ROVER processing")
    rover_wrapper = ROVERWrapper(podcasts_path=src_path, model_names=model_names)
    rover_wrapper.aggregate_and_save()

    logger.info(f"Finished processing ROVER")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Transcribe audio files in parallel using multiple GPUs/CPUs.")
    parser.add_argument("--config_path", type=str, help="Path to the configuration YAML file.")
    args = parser.parse_args()
    main(args)
