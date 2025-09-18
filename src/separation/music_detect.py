import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import torch
import torchaudio
import transformers
from loguru import logger
from safetensors import safe_open
from tqdm import tqdm

from src.separation.model import WavLMForEndpointing
from src.utils import get_audio_paths, load_config

g_model = None
g_processor = None
g_device = None
g_threshold = None


def init_process(
    model_name: str,
    checkpoint_path: str,
    device: str,
    threshold: float
) -> None:
    """
    Initializes the model, processor, device, and threshold for a worker process.
    This function runs once for each process in the ProcessPoolExecutor.
    """
    global g_model, g_processor, g_device, g_threshold

    logger.info(f"Initializing the process on the device {device}...")

    g_device = device
    g_threshold = threshold

    try:
        g_processor = transformers.AutoFeatureExtractor.from_pretrained(model_name)
        config = transformers.AutoConfig.from_pretrained(model_name)
        model = WavLMForEndpointing(config)

        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            state_dict = {key: f.get_tensor(key) for key in f.keys()}
        
        # Load the state dict and then move the model to the correct device
        model.load_state_dict(state_dict)
        model.to(g_device)
        model.eval()

        g_model = model
        logger.info(f"The process on the {device} device has been successfully initialized.")
    except Exception as e:
        logger.error(f"Error initializing the process on the device {device}: {e}")
        # Re-raise the exception to prevent the process from starting incorrectly
        raise


def process_audio_file(audio_path: Path) -> Optional[str]:
    """
    Processes a single audio file to check if it should be removed.
    Returns the path of the deleted file if removed, otherwise returns None.
    """
    global g_model, g_processor, g_device, g_threshold

    # If globals aren't initialized, something went wrong with init_process
    if g_model is None or g_processor is None or g_device is None:
        logger.error(f"Global variables are not initialized in the process.")
        return None

    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # TODO: don't forget to remove the code

        if (waveform.shape[-1] / sample_rate) <= 5:
            os.remove(audio_path)
            logger.info(f"{audio_path} -- removed {waveform.shape[-1] / sample_rate}")
            
        # TODO: don't forget to remove the code

        # Resample if not 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Convert stereo to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Ensure waveform is on the correct device for processing
        waveform = waveform.to(g_device)

        inputs = g_processor(
            waveform.squeeze().cpu().numpy(), # Use .cpu().numpy() to handle device
            sampling_rate=16000,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
       
        # Move input tensors to the correct device
        inputs = {key: inputs[key].to(g_device) for key in inputs.keys()}

        with torch.no_grad():
            result = g_model(**inputs)

        # Check the condition and remove the file
        if result['logits'][0][0].item() > g_threshold:
            # Use os.remove for a more direct delete operation
            # Add a log message to confirm the action
            logger.info(f"Удаление файла: {audio_path} (score: {result['logits'][0][0].item():.4f})")
            os.remove(audio_path)
            return str(audio_path) # Return the path of the removed file
        
        # Return None if the file is not removed
        return None

    except Exception as e:
        logger.error(f"Error in file processing{audio_path}: {e}")
        return None # Return None in case of an error


def main(args):
    """
    Main function to orchestrate the multiprocessing and audio processing.
    """
    config = load_config(args.config_path, 'separation')

    num_workers_per_gpu = config.get('num_workers_detect', 2)
    model_name = 'microsoft/wavlm-base-plus'
    checkpoint_path = config.get('checkpoint_path', '/path/to/your/model.safetensors')
    audio_data_path = config.get('podcasts_path', '/path/to/your/audio')
    threshold = config.get('threshold', 0.5)

    all_audio_files = get_audio_paths(audio_data_path)
    if not all_audio_files:
        logger.warning(f"Audio files not found in {audio_data_path}")
        return

    if not torch.cuda.is_available():
        logger.error("There are no available GPUs. Exit.")
        return

    available_gpu_ids = list(range(torch.cuda.device_count()))
    num_gpus = len(available_gpu_ids)

    logger.info(
        f"""
        Starting audio processing:
        - Data path: {audio_data_path}
        - Model: {model_name}
        - Checkpoint: {checkpoint_path}
        - Number of GPUs: {num_gpus} (IDs: {available_gpu_ids})
        - Workers on GPU: {num_workers_per_gpu}
        - Total workers: {num_gpus * num_workers_per_gpu}
        - Total files to process: {len(all_audio_files)}
        - Threshold {threshold}
        """
    )

    # Use a dictionary to map GPU IDs to a list of files for better clarity
    files_per_gpu = {gpu_id: [] for gpu_id in available_gpu_ids}
    for i, path in enumerate(all_audio_files):
        gpu_assignment_index = i % num_gpus
        gpu_id = available_gpu_ids[gpu_assignment_index]
        files_per_gpu[gpu_id].append(path)

    all_futures = []
    executors = []

    for gpu_id, files_for_this_gpu in files_per_gpu.items():
        if not files_for_this_gpu:
            continue

        device_str = f'cuda:{gpu_id}'
        logger.info(f"Creating a ProcessPoolExecutor for {device_str} with {num_workers_per_gpu} workers for {len(files_for_this_gpu)} files.")

        executor = ProcessPoolExecutor(
            max_workers=num_workers_per_gpu,
            initializer=init_process,
            initargs=(model_name, checkpoint_path, device_str, threshold)
        )
        executors.append(executor)

        for path in files_for_this_gpu:
            future = executor.submit(process_audio_file, path)
            all_futures.append(future)

    logger.info(f"All {len(all_futures)} issues have been sent for processing. Waiting for completion...")

    deleted_files_count = 0
    with tqdm(total=len(all_futures), desc="Audio file processing") as pbar:
        for future in as_completed(all_futures):
            result = future.result()
            if result:
                deleted_files_count += 1
            pbar.update(1)

    for executor in executors:
        executor.shutdown()
    
    logger.info(f"Processing is completed. Total deleted {deleted_files_count} files.")


if __name__ == "__main__":
    # It's good practice to set the start method at the top level
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    main(args)