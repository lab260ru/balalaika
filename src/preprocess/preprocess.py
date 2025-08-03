import argparse
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any

import torch
import torchaudio
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv

from huggingface_hub import login

from src.utils import load_config
from src.libs.smart_turn.offline_svad import OfflineVAD

def get_audio_paths(directory: str) -> List[str]:
    audio_paths = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if len(os.path.basename(full_path).split('_')) == 4:
            continue
        if os.path.isdir(full_path):
            audio_paths.extend(get_audio_paths(full_path))
        elif entry.endswith(".mp3"):
            audio_paths.append(full_path)
    return audio_paths

def postprocess_vad_result(vad_result: List[Dict[str, Any]], duration: float = 15.0) -> Tuple[List[float], List[float]]:
    speech_intervals = [
        {'start_time': item['start_time'], 'end_time': item['end_time']}
        for item in vad_result if item['prediction'] == 1
    ]
    
    if not speech_intervals:
        return [], []

    timesteps_starts = []
    timesteps_ends = []

    current_start = speech_intervals[0]['start_time']
    current_end = speech_intervals[0]['end_time']

    for interval in speech_intervals[1:]:
        next_start = interval['start_time']
        next_end = interval['end_time']

        if next_end - current_start <= duration:
            current_end = next_end
        else:
            if duration / 3 <= current_end - current_start <= duration:
                timesteps_starts.append(current_start)
                timesteps_ends.append(current_end)
            current_start = next_start
            current_end = next_end

    if duration / 3 <= current_end - current_start <= duration:
        timesteps_starts.append(current_start)
        timesteps_ends.append(current_end)

    return timesteps_starts, timesteps_ends

def cut_audio(
    audio: torch.Tensor,
    sr: int,
    start_timestamps: List[float],
    end_timestamps: List[float],
    output_folder: str,
    album_id: str,
    episode_id: str,
    format: str = 'mp3',
    duration: float = 15.0
):
    try:
        os.makedirs(output_folder, exist_ok=True)
        segments_created = 0
        for start_time, end_time in zip(start_timestamps, end_timestamps):
            if end_time - start_time <= duration / 3:
                continue
            
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            end_sample = min(audio.shape[-1], end_sample)
            if end_sample <= start_sample:
                continue

            segment = audio[:, start_sample:end_sample]
            output_audio_filename = f"{start_time:.2f}_{end_time:.2f}_{album_id}_{episode_id}.{format}"
            output_audio_path = os.path.join(output_folder, output_audio_filename)
            
            torchaudio.save(output_audio_path, segment, sr)
            segments_created += 1

        logger.success(f"Processed {segments_created} segments: {output_folder}")

    except Exception as e:
        logger.error(f"Error in cut_audio: {e}")
        raise

def init_vad_process(gpu_id: int, vad_args: Dict[str, Any]):
    global smart_vad
    
    if torch.cuda.is_available():
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(device)
    else:
        device = "cpu"
        logger.warning(f"No GPU available, using CPU for process")
    
    smart_vad = OfflineVAD(
        silero_vad_threshold=vad_args['silero_vad_threshold'],
        smart_vad_threshold=vad_args['smart_vad_threshold'],
        smart_vad_path=vad_args['smart_vad_path'],
        device=device
    )
    logger.info(f"VAD initialized on {device}")

def process_audio_file(path_audio: str, duration: float):
    global smart_vad
    
    album_id = os.path.basename(os.path.dirname(path_audio))
    episode_id = os.path.splitext(os.path.basename(path_audio))[0]
    episode_folder = os.path.join(os.path.dirname(path_audio), episode_id)

    try:
        audio, sr = torchaudio.load(path_audio)
        if audio.shape[-1] / sr <= duration:
            return
    except Exception as e:
        logger.error(f"Broken file {path_audio}: {e}")
        if os.path.exists(path_audio):
            os.remove(path_audio)
        return

    try:
        vad_result = smart_vad.process_file(path_audio)
        import pickle
        with open('data.pkl', 'wb') as file:
            pickle.dump(vad_result, file)
        timesteps_starts, timesteps_ends = postprocess_vad_result(vad_result, duration=duration)
        
        if not timesteps_starts:
            logger.warning(f"No speech segments found in {path_audio}")
            return

        cut_audio(
            audio=audio,
            sr=sr,
            start_timestamps=timesteps_starts,
            end_timestamps=timesteps_ends,
            output_folder=episode_folder,
            album_id=album_id,
            episode_id=episode_id,
            format='mp3',
            duration=duration
        )

    except Exception as e:
        logger.error(f"Processing error {path_audio}: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if os.path.exists(episode_folder) and os.listdir(episode_folder):
        os.remove(path_audio)
        logger.info(f"Original file deleted: {path_audio}")

def main(args):
    load_dotenv()
    hf_key=os.environ.get('HF_TOKEN')
    login(token=hf_key)

    config = load_config(args.config_path, 'preprocess')

    podcasts_path = args.podcasts_path or config.get('podcasts_path', '../../../podcasts')
    duration = args.duration or config.get('duration', 15)
    num_workers = args.num_workers or config.get('num_workers', 4)
    smart_vad_model = args.smart_vad_model or config.get('smart_vad_model', "pipecat-ai/smart-turn-v2")

    num_gpus = torch.cuda.device_count()
    available_gpus = list(range(num_gpus))
    if num_gpus == 0:
        logger.error("No GPUs available. Exiting.")
        return
        

    audio_paths = get_audio_paths(podcasts_path)
    if not audio_paths:
        logger.info("No audio files found.")
        return

    vad_args = {
        'silero_vad_threshold': 0.5,
        'smart_vad_threshold': 0.5,
        'smart_vad_path': smart_vad_model
    }

    logger.info(f"""
    Starting processing with:
    Podcasts path: {podcasts_path}
    Smart VAD model: {smart_vad_model}
    Segment duration: {duration} seconds
    Workers per GPU: {num_workers}
    Available GPUs: {available_gpus}
    Files to process: {len(audio_paths)}
    """)

    all_futures = []
    executors = []
    
    for gpu_id in available_gpus:
        executor = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_vad_process,
            initargs=(gpu_id, vad_args)
        )

        executors.append(executor)
 
        files_for_gpu = []
        for i, path in enumerate(audio_paths):
            if i % len(available_gpus) == gpu_id % len(available_gpus):
                files_for_gpu.append(path)
        
        logger.info(f"GPU {gpu_id}: processing {len(files_for_gpu)} files")
    
        for path in files_for_gpu:
            future = executor.submit(process_audio_file, path, duration)
            all_futures.append(future)

    with tqdm(total=len(all_futures), desc="Processing podcasts") as pbar:
        for future in as_completed(all_futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing file: {e}")
            finally:
                pbar.update(1)

    for executor in executors:
        executor.shutdown()

if __name__ == "__main__":
    torchaudio.set_audio_backend('soundfile')
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Process audio files using smart-turn VAD model.")
    parser.add_argument("--config_path", type=str, help="Path to YAML configuration file")
    parser.add_argument("--podcasts_path", type=str, help="Path to podcasts folder")
    parser.add_argument("--smart_vad_model", type=str, help="Name of smart-turn model")
    parser.add_argument("--duration", type=int, help="Target segment duration in seconds")
    parser.add_argument("--num_workers", type=int, help="Number of worker processes per GPU")
    
    args = parser.parse_args()
    main(args)