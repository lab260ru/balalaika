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

def init_worker(model_name: str, device_str: str, **kwargs):
    """
    Initializes the appropriate ASR model wrapper for each worker process.
    """
    global model
    logger.info(f"Initializing worker for model '{model_name}' on {device_str}...")
    
    try:
        if 'giga' in model_name:
            model = GigaAMWrapper(model_id=model_name, device=device_str, lm_path=kwargs.get('lm_path'))
        elif 'tone' in model_name:
            model = ToneWrapper(model_id=model_name, device=device_str)
        elif 'vosk' in model_name:
            model = VoskWrapper(model_id=kwargs.get('vosk_path'), device=device_str)
        else:
            raise ValueError(f"Unknown model type for '{model_name}'")
        logger.info(f"Worker initialized successfully for model '{model_name}' on {device_str}.")
    except Exception as e:
        logger.error(f"Failed to initialize worker for model '{model_name}' on {device_str}: {e}")
        model = None


def process_file(path: Path, model_name_for_output: str, with_timestamps: bool):
    """
    Transcribes a single audio file using the globally initialized model.
    Generates .txt and optionally .tst files.
    """
    global model
    if model is None:
        logger.error(f"Model is not initialized in this worker. Skipping {path.name}.")
        return

    txt_path = path.with_name(f"{path.stem}_{model_name_for_output}.txt")
    tst_path = path.with_name(f"{path.stem}_{model_name_for_output}.tst")

    if txt_path.exists():
        logger.info(f"Skipping already transcribed file: {txt_path}")
        return

    try:
        if with_timestamps:
            plain_text, tst_content = model.transcribe_with_timestamps(str(path))
        else:
            plain_text = model.transcribe(str(path))
            tst_content = ""

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(plain_text)

        if with_timestamps and tst_content:
            with open(tst_path, "w", encoding="utf-8") as f:
                f.write(tst_content)
    except Exception as e:
        logger.error(f"Failed to process {path.name}: {e}")


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
    config = load_config(args.config_path, 'transcription') if args.config_path else {}

    model_names = args.model_names or config.get('model_names', ['giga_rnnt'])
    num_workers_per_gpu = args.num_workers or config.get('num_workers', 1)
    src_path = args.podcasts_path or config.get('podcasts_path', '.')
    lm_path = args.lm_path or config.get('lm_path', None)
    with_timestamps = args.with_timestamps or config.get('with_timestamps', False)
    vosk_path = args.vosk_path or config.get('vosk_path', None)

    logger.info(f"Starting transcription run for models: {model_names}")

    available_gpu_ids = list(range(torch.cuda.device_count()))
    if not available_gpu_ids:
        logger.warning("No CUDA GPUs detected. Using CPU. This will be slow.")
        raise 
    
    num_devices = len(available_gpu_ids)

    for model_name in model_names:
        logger.info(f"--- Processing model: {model_name} ---")

        model_name_for_output = 'vosk' if 'vosk' in model_name else model_name
        timestamps_supported = 'giga_ctc' in model_name or 'tone' in model_name or 'vosk' in model_name

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
            device_assignment_index = i % num_devices
            files_for_each_device[device_assignment_index].append(path)

        all_futures = []
        executors = []
        
        init_kwargs = {'lm_path': lm_path, 'vosk_path': vosk_path}
        task_fn = partial(process_file, model_name_for_output=model_name_for_output, with_timestamps=current_with_timestamps)

        for i, device_id in enumerate(available_gpu_ids):
            device_str = f'cuda:{device_id}' if device_id != 'cpu' else 'cpu'
            files_for_this_device = files_for_each_device[i]

            if not files_for_this_device:
                continue

            logger.info(f"Creating ProcessPoolExecutor for {device_str} with {num_workers_per_gpu} workers for {len(files_for_this_device)} files.")
            
            initializer_fn = partial(init_worker, model_name, device_str, **init_kwargs)

            executor = ProcessPoolExecutor(
                max_workers=num_workers_per_gpu,
                initializer=initializer_fn,
                mp_context=multiprocessing.get_context('spawn')
            )
            executors.append(executor)

            for path in files_for_this_device:
                future = executor.submit(task_fn, path)
                all_futures.append(future)

        for future in tqdm(as_completed(all_futures), total=len(all_futures), desc=f"Transcribing ({model_name})"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"A task encountered an error: {e}")

        for executor in executors:
            executor.shutdown(wait=True)
        
        logger.info(f"Finished processing model: {model_name}")

    rover_wrapper = ROVERWrapper(podcasts_path=src_path)
    rover_wrapper.aggregate_and_save()

    logger.info(f"Finished processing ROVER")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    torchaudio.set_audio_backend('soundfile')

    parser = argparse.ArgumentParser(description="Transcribe audio files in parallel using multiple GPUs/CPUs.")
    parser.add_argument("--config_path", type=str, help="Path to the configuration YAML file.")
    parser.add_argument("--podcasts_path", type=str, help="Path to the directory containing audio files.")
    parser.add_argument("--num_workers", type=int, help="Number of worker processes per device.")
    parser.add_argument("--model_names", nargs='+', help="One or more model names to use (e.g., 'giga_ctc_lm' 'tone' 'vosk').")
    parser.add_argument("--lm_path", type=str, help="Path to the KenLM language model binary file for GigaAM-CTC.")
    parser.add_argument('--vosk_path', type=str, help="Path to the Vosk model directory.")
    parser.add_argument('--with_timestamps', action='store_true', help="Enable to generate .tst files with word timestamps (if supported by model).")

    args = parser.parse_args()
    main(args)
