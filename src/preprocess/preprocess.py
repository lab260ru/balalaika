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

def postprocess_vad_result(
    vad_result: List[Dict[str, Any]],
    duration: float = 15.0,
    min_duration: float = 1.0,
    gap_threshold: float = 30.0
) -> Tuple[List[float], List[float]]:
    """
    Сбалансированный алгоритм для постобработки результатов VAD.
    Приоритет: законченные фразы, но заполняются большие промежутки между ними.
    """
    if not vad_result:
        return [], []

    # Этап 1: Сохраняем все семантически законченные фразы (prediction=1)
    primary_segments = []
    
    # Итерируем по результатам и объединяем последовательные "законченные" сегменты
    current_segment_start = None
    
    for i, item in enumerate(vad_result):
        if item['prediction'] == 1:
            if current_segment_start is None:
                # Начало нового потенциального сегмента
                current_segment_start = item['start_time']
            # Конец текущего потенциального сегмента
            segment_end = item['end_time']
            segment_duration = segment_end - current_segment_start
            
            # Проверяем, не превышаем ли мы максимальную длительность
            if segment_duration > duration:
                # Сохраняем предыдущий сегмент и начинаем новый
                if i > 0 and vad_result[i-1]['prediction'] == 1:
                    primary_segments.append((current_segment_start, vad_result[i-1]['end_time']))
                current_segment_start = item['start_time']

    # Сохраняем последний накопленный сегмент
    if current_segment_start is not None:
        last_item = vad_result[-1]
        primary_segments.append((current_segment_start, last_item['end_time']))

    # Фильтруем сегменты по длительности
    primary_segments = [(s, e) for s, e in primary_segments if min_duration <= e - s <= duration]
    
    # Сортируем и убираем дубликаты
    primary_segments = sorted(list(set(primary_segments)), key=lambda x: x[0])

    if not primary_segments:
        return [], []
    
    # Этап 2: Заполняем большие промежутки между сохранёнными сегментами
    filled_segments = []
    last_end_time = 0.0
    
    for start, end in primary_segments:
        gap_duration = start - last_end_time
        if gap_duration > gap_threshold:
            filled_segments.extend(_fill_gap(vad_result, last_end_time, start, duration, min_duration))
        last_end_time = end

    # Проверяем промежуток в конце
    audio_end = vad_result[-1]['end_time']
    if audio_end - last_end_time > gap_threshold:
        filled_segments.extend(_fill_gap(vad_result, last_end_time, audio_end, duration, min_duration))
    
    # Объединяем итоговые сегменты
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
    """
    Вспомогательная функция: разбивает промежуток [gap_start, gap_end] на сегменты
    длиной [min_duration, duration], используя интервалы из vad_result.
    """
    filled_segments = []
    
    current_time = gap_start
    while current_time < gap_end:
        # Находим следующий интервал, который начинается после current_time
        next_interval_index = -1
        for i, item in enumerate(vad_result):
            if item['start_time'] >= current_time:
                next_interval_index = i
                break
        
        if next_interval_index == -1:
            break
            
        seg_start = vad_result[next_interval_index]['start_time']
        seg_end = vad_result[next_interval_index]['end_time']

        # Объединяем интервалы, пока не превысим max_duration или не дойдем до конца промежутка
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
        # TODO: don't forget to remove the code
        audio, sr = torchaudio.load(path_audio)
        if audio.shape[-1] / sr <= 5:
            logger.info(f"{path_audio} -- removed {audio.shape[-1] / sr} duration")
        if audio.shape[-1] / sr <= duration:
            return
    except Exception as e:
        logger.error(f"Broken file {path_audio}: {e}")
        if os.path.exists(path_audio):
            os.remove(path_audio)
        return

    try:
        vad_result = smart_vad.process_file(path_audio)
        # import pickle
        # with open('data.pkl', 'wb') as file:
        #     pickle.dump(vad_result, file)
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
        'silero_vad_threshold': 0.4,
        'smart_vad_threshold': 0.4,
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