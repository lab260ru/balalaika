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

from src.utils import load_config, get_audio_paths
from src.libs.smart_turn.offline_svad import OfflineVAD

from typing import List, Dict, Any, Tuple

torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


smart_vad = None


def initializer_wrapper(args_dict: Dict[str, Any], gpus_count: int):
    """
    Initializer for each worker process. Assigns a GPU and creates a VAD instance.
    This function runs once when each worker process is created.
    """
    # Get a unique ID for each worker process (note: _identity is an internal API)
    worker_id = multiprocessing.current_process()._identity[0] - 1
    # Assign a GPU to the worker in a round-robin fashion
    gpu_id = worker_id % gpus_count if gpus_count > 0 else 0
    
    # Call the original VAD initialization function for this specific worker
    init_vad_process(gpu_id=gpu_id, vad_args=args_dict)

def postprocess_vad_result(
    vad_result: List[Dict[str, Any]],
    duration: float = 15.0,
    min_duration: float = 1.0,
    gap_threshold: float = 30.0
) -> Tuple[List[float], List[float]]:

    if not vad_result:
        return [], []

    primary_segments = []
    
    current_segment_start = None
    
    for i, item in enumerate(vad_result):
        if item['prediction'] == 1:
            if current_segment_start is None:
                current_segment_start = item['start_time']
            segment_end = item['end_time']
            segment_duration = segment_end - current_segment_start
            
            if segment_duration > duration:
                if i > 0 and vad_result[i-1]['prediction'] == 1:
                    primary_segments.append((current_segment_start, vad_result[i-1]['end_time']))
                current_segment_start = item['start_time']

    if current_segment_start is not None:
        last_item = vad_result[-1]
        primary_segments.append((current_segment_start, last_item['end_time']))

    primary_segments = [(s, e) for s, e in primary_segments if min_duration <= e - s <= duration]
    
    primary_segments = sorted(list(set(primary_segments)), key=lambda x: x[0])

    if not primary_segments:
        return [], []
    
    # stage 2 
    filled_segments = []
    last_end_time = 0.0
    
    for start, end in primary_segments:
        gap_duration = start - last_end_time
        if gap_duration > gap_threshold:
            filled_segments.extend(_fill_gap(vad_result, last_end_time, start, duration, min_duration))
        last_end_time = end

    audio_end = vad_result[-1]['end_time']
    if audio_end - last_end_time > gap_threshold:
        filled_segments.extend(_fill_gap(vad_result, last_end_time, audio_end, duration, min_duration))
    
    all_segments = primary_segments + filled_segments
    all_segments = sorted(list(set(all_segments)), key=lambda x: x[0])
    
    all_starts = [s for s, e in all_segments]
    all_ends = [e for s, e in all_segments]

    return all_starts, all_ends


def _fill_gap(
    vad_result: List[Dict[str, Any]],
    gap_start: float,
    gap_end: float,
    duration: float,
    min_duration: float
) -> List[Tuple[float, float]]:

    filled_segments = []
    
    current_time = gap_start
    while current_time < gap_end:
        next_interval_index = -1
        for i, item in enumerate(vad_result):
            if item['start_time'] >= current_time:
                next_interval_index = i
                break
        
        if next_interval_index == -1:
            break
            
        seg_start = vad_result[next_interval_index]['start_time']
        seg_end = vad_result[next_interval_index]['end_time']

        for j in range(next_interval_index + 1, len(vad_result)):
            if vad_result[j]['end_time'] - seg_start > duration or vad_result[j]['end_time'] > gap_end:
                break
            seg_end = vad_result[j]['end_time']
        
        seg_duration = seg_end - seg_start
        if min_duration <= seg_duration <= duration:
            filled_segments.append((seg_start, seg_end))
            
        current_time = seg_end
        
    return filled_segments

def cut_audio(
    audio: torch.Tensor,
    sr: int,
    start_timestamps: List[float],
    end_timestamps: List[float],
    output_folder: str,
    album_id: str,
    episode_id: str,
    format: str = 'opus',
    duration: float = 15.0
):
    try:
        os.makedirs(output_folder, exist_ok=True)
        segments_created = 0
        for start_time, end_time in zip(start_timestamps, end_timestamps):
            if end_time - start_time <= duration / 2:
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
        **vad_args,
        device=device
    )
    logger.info(f"VAD initialized on {device}")

def process_audio_file(path_audio: str, duration: float):
    global smart_vad
    
    album_id = os.path.basename(os.path.dirname(path_audio))
    episode_id = os.path.splitext(os.path.basename(path_audio))[0]
    episode_folder = os.path.join(os.path.dirname(path_audio), episode_id)

    # TODO: don't forget to remove the code
    audio, sr = torchaudio.load(path_audio)
    if audio.shape[-1] / sr <= 2:
        logger.info(f"{path_audio} -- removed {audio.shape[-1] / sr} duration")
        os.remove(path_audio)
    # # TODO: don't forget to remove the code

    try:
        if audio.shape[-1] / sr <= duration:
            return
    except Exception as e:
        logger.error(f"Broken file {path_audio}: {e}")
        if os.path.exists(path_audio):
            os.remove(path_audio)
        return

    try:
        vad_result = smart_vad.process_file(path_audio)
        timesteps_starts, timesteps_ends = postprocess_vad_result(vad_result, duration=duration)
        if not timesteps_starts:
            logger.warning(f"No speech segments found in {path_audio}, removed")
            os.remove(path_audio)
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
    hf_key = os.environ.get('HF_TOKEN')
    if hf_key:
        login(token=hf_key)
    else:
        logger.warning("HF_TOKEN not found in environment.")

    config = load_config(args.config_path, 'preprocess')

    podcasts_path = config.get('podcasts_path', '../../../podcasts')
    duration = config.get('duration', 15)
    vad_args = config.get('vad_args')

    num_gpus = torch.cuda.device_count()

    workers_per_gpu = config.get('num_workers', 4)

    if num_gpus > 0:
        total_workers = num_gpus * workers_per_gpu
        logger.info (f"Found {num_gpus} GPU. We run {workers_per_gpu} processes for each one.")
        logger.info (f"Total number of work processes: {total_workers}.")

    audio_paths = get_audio_paths(podcasts_path)
    if not audio_paths:
        logger.info ("No audio files found for processing.")
        return


    logger.info (f"""
        Running parallel processing:
        The path to Podcasts: {podcasts_path}
        Segment duration: {duration} seconds
        Total number of workers: {total_workers}
        Files to process: {len(audio_paths)}
        """)


    with ProcessPoolExecutor(
        max_workers=total_workers,
        initializer=initializer_wrapper,
        initargs=(vad_args, num_gpus)
    ) as executor:
        
        futures = [executor.submit(process_audio_file, path, duration) for path in audio_paths]

        for future in tqdm(as_completed(futures), total=len(audio_paths), desc="Podcast Processing"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"The task ended with an error:{e}")

    logger.info("All files have been processed.")

if __name__ == "__main__":  
    torchaudio.set_audio_backend('soundfile')
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Process audio files using smart-turn VAD model.")
    parser.add_argument("--config_path", type=str, help="Path to YAML configuration file")
    args = parser.parse_args()
    main(args)