import argparse
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, List, Tuple

import torch
import torchaudio
import yaml
from loguru import logger
from tqdm import tqdm
from faster_whisper import WhisperModel
from dotenv import load_dotenv

from huggingface_hub import login

from src.utils import load_config

def get_audio_paths(directory: str) -> List[str]:
    audio_paths = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if len(os.path.basename(full_path).split('_')) == 4:
            continue
        if os.path.isdir(full_path):
            audio_paths.extend(get_audio_paths(full_path))
        elif entry.endswith(".mp3") :
            audio_paths.append(full_path)
    return audio_paths

def get_whisper_segments(
    model: Any,
    path_audio: str,
    beam_size: int = 5
) -> Tuple[List[float], List[float], List[str]]:
    
    segments, info  = model.transcribe(
        path_audio, 
        beam_size=beam_size,
        language='ru'
        )

    timesteps_starts = []
    timesteps_ends = []
    
    timestamps_text = []

    for segment in segments:
        timesteps_starts.append(segment.start)
        timesteps_ends.append(segment.end + 0.05)
        timestamps_text.append(segment.text)
    
    assert len(timesteps_starts) == len(timesteps_ends) == len(timestamps_text), "Mismatch in timestamps lengths."
    return timesteps_starts, timesteps_ends, timestamps_text


def get_piece_idx(
    timesteps_starts: List[float],
    timesteps_ends: List[float],
    duration: float = 15.0,
) -> List[Tuple[int, int]]:
    """
    Groups speech segments into optimal chunks respecting duration constraints,
    returning indices of the original segments rather than time intervals.

    Processes speech segments to identify optimal groupings that:
    1. Do not exceed specified maximum duration when combined
    2. Contain complete consecutive speech segments
    3. Maintain natural segmentation boundaries

    Args:
        timesteps_starts: Start times of speech segments (seconds)
        timesteps_ends: End times of speech segments (seconds)
        duration: Maximum allowed combined duration in seconds (default: 15.0)

    Returns:
        List of tuples representing segment index ranges:
        - Each tuple contains (start_index, end_index) of segments in original lists
        - Index ranges reference positions in timesteps_starts/timesteps_ends
        - Resulting combined segments would be between duration//3 and duration seconds
        - Maintains original segment order

    Example:
        Input segments:
        starts = [0.0, 2.5, 5.5, 12.5]
        ends = [2.15, 5.15, 12.15, 18.15]
        
        Output with duration=15:
        [(0, 2), (3, 3)]  # First 3 segments grouped, last segment standalone
    """
        
    if not timesteps_starts or not timesteps_ends:
        return []

    pieces = []
    n = len(timesteps_starts)
    m = len(timesteps_ends)
    start_idx = 0

    while start_idx < n:
        left_s = timesteps_starts[start_idx]
        end_idx = start_idx
        temp_duration = 0.0

        while end_idx < m:
            right_s = timesteps_ends[end_idx]
            temp_duration = right_s - left_s
            if temp_duration > duration:
                break
            end_idx += 1

        if end_idx > start_idx:
            current_duration = timesteps_ends[end_idx - 1] - left_s
            if (current_duration >= duration / 3) and (current_duration <= duration):
                pieces.append((start_idx, end_idx - 1))

        start_idx = max(end_idx, start_idx + 1)

    return pieces

def cut_audio(
    audio: torch.Tensor,
    sr: int,
    pieces: List[Tuple[int, int]],
    satrt_timestamps: List[float],
    end_timestamps: List[float],
    text_segments: List[str],
    output_folder: str,
    album_id: str,
    episode_id: str,
    format: str = 'mp3',
    duration:float = 15.0
) -> None:
    try:
        os.makedirs(output_folder, exist_ok=True)
        for i, (start_idx, end_idx) in enumerate(pieces):

            start_time = satrt_timestamps[start_idx]
            end_time = end_timestamps[end_idx]

            if end_time - start_time <= duration / 3 :
                continue
            
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            end_sample = min(audio.shape[-1], end_sample)
            assert end_sample > start_sample

            segment = audio[:, start_sample:end_sample]
            output_audio_filename = f"{start_time:.2f}_{end_time:.2f}_{album_id}_{episode_id}.{format}"
            output_whisper_filename = f"{start_time:.2f}_{end_time:.2f}_{album_id}_{episode_id}_whisper.txt"

            output_audio_path = os.path.join(output_folder, output_audio_filename)
            output_whisper_path = os.path.join(output_folder, output_whisper_filename)

            whisper_text = ' '.join(text_segments[start_idx:end_idx + 1])
            with open(output_whisper_path, 'w', encoding='utf-8') as f:
                f.write(whisper_text)

            torchaudio.save(output_audio_path, segment, sr)

        logger.success(f"The folder has been processed : {output_folder}")

    except Exception as e:
        logger.error(f"Error : {e}")
        raise

def process_audio_file(
    path_audio: str,
    duration: float,
    beam_size: int
) -> None:
    album_id = os.path.basename(os.path.dirname(path_audio))
    episode_id = os.path.splitext(os.path.basename(path_audio))[0]
    
    logger.info(f"Processing: Album={album_id}, Episode={episode_id}")

    episode_folder = os.path.join(os.path.dirname(path_audio), episode_id)

    try:
        audio, sr = torchaudio.load(path_audio)

        if audio.shape[-1] / sr <= duration:
            return
    except Exception as e:
        os.remove(path_audio)
        logger.info(f"broken file {path_audio}: {e}")
        return
    
    try:
        timesteps_starts, timesteps_ends, timestamps_text = get_whisper_segments(model, path_audio, beam_size)
        pieces = get_piece_idx(timesteps_starts, timesteps_ends, duration)

        cut_audio(
            audio=audio,
            sr=sr,
            pieces=pieces,
            satrt_timestamps=timesteps_starts,
            end_timestamps=timesteps_ends,
            text_segments=timestamps_text,
            output_folder=episode_folder,
            album_id=album_id,
            episode_id=episode_id,
            format='mp3',
            duration=duration
        )


    except Exception as e:
        logger.error(f"Processing error {path_audio}: {e}")
        
    if len(os.listdir(episode_folder)) > 0 or len(pieces) == 0  : # the audio was cut or we couldn't cut it considering our length
        os.remove(path_audio)
        logger.info(f"Temporary file deleted: {path_audio}")

def init_process(
    whisper_model: str,
    device: str,
    compute_type: str = 'float16',
    device_index = [0]
)-> None:
    
    global model
    model = WhisperModel(
        whisper_model,
        device=device,
        compute_type=compute_type,
        device_index=device_index,
        )
    

def main(args):
    
    load_dotenv()
    hf_key = os.getenv("HF_TOKEN")
    login(token=hf_key)

    config = load_config(args.config_path, 'preprocess')

    podcasts_path = args.podcasts_path if args.podcasts_path else config.get('podcasts_path', '../../../podcasts')
    duration = args.duration if args.duration else config.get('duration', 15)
    device = args.device if args.device else config.get('device', 'cuda')
    num_workers = args.num_workers if args.num_workers else config.get('num_workers', 4)
    whisper_model = args.whisper_model if args.whisper_model else config.get('whisper_model', 'large-v3')
    compute_type = args.compute_type if args.compute_type else config.get('compute_type', 'float16')
    beam_size = args.beam_size if args.beam_size else config.get('beam_size', 5)
    device_index = list(range(torch.cuda.device_count()))

    audio_paths = get_audio_paths(podcasts_path)

    max_workers = min(num_workers, os.cpu_count())

    logger.info(
    f"""
    Using parms 
    podcasts_path:{podcasts_path} 
    whisper_model:{whisper_model}
    duration:{duration} 
    device:{device} 
    num_workers:{num_workers}
    device_index:{device_index}
    compute_type:{compute_type}
    beam_size:{beam_size}
    """)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_process,
        initargs=(whisper_model, device, compute_type, device_index, )
        ) as executor:
        futures = [
            executor.submit(process_audio_file, path_audio, duration, beam_size)
            for path_audio in audio_paths
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing file: {e}")

if __name__ == "__main__":
    torchaudio.set_audio_backend('soundfile')
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Process audio files using whisper model.")
    parser.add_argument(
        "--config_path",
        help="Path to YAML configuration file",
        type=str,
    )
    parser.add_argument(
        "--podcasts_path",
        help="Path to podcasts folder", 
        type=str, 
    )
    parser.add_argument(
        "--whisper_model",
        help="name of model", 
        type=str, 
    )
    parser.add_argument(
        "--compute_type",
        help="compute type", 
        type=str, 
    )
    parser.add_argument(
        "--beam_size",
        help="beam size", 
        type=int,
    )
    parser.add_argument(
        "--duration", 
        help="Duration in seconds", 
        type=int, 
        )
    parser.add_argument(
        "--device", 
        help="Device", 
        type=str, 
    )
    parser.add_argument(
        "--num_workers", 
        help="Number of workers", 
        type=int, 
    )

    args = parser.parse_args()
    main(args)