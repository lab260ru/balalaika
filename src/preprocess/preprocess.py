import argparse
import os
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torchaudio
import pandas as pd
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login

from src.utils.utils import load_config, get_audio_paths
from src.libs.smart_turn.offline_svad import SmartVAD

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

CHUNK_DURATION_S = 15 * 60  

sortformer_model = None
smart_vad = None

def get_providers(cuda_id: int) -> list:
    return [
        ("CUDAExecutionProvider", {
            "device_id": cuda_id,
            "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
        }),
        "CPUExecutionProvider"
    ]

def init_models(gpu_id: int, config: Dict[str, Any]):
    global sortformer_model, smart_vad
    device = f"cuda:{gpu_id}" 
    providers = get_providers(gpu_id)

    try:
        from src.preprocess.sortformer_onnx import Sortformer, DiarizationConfig
    except ImportError:
        logger.error("Sortformer module or Sortformer class not found in src.preprocess.sortformer_onnx")
        raise

    model_config = DiarizationConfig()
    sortformer_model = Sortformer(
        model_path=config.get('sortformer_model'), 
        config=model_config, 
        providers=providers
    )

    vad_args = config.get('vad_args', {})
    smart_vad = SmartVAD(
        smart_vad_model=vad_args.get('smart_vad_model', './models/smart-turn-v3.0.onnx'),
        smart_vad_threshold=vad_args.get('smart_vad_threshold', 0.4),
        device=device
    )
    logger.info(f"Models initialized on {device}")


def parse_diarization_output(raw_results) -> List[Tuple[float, float, int]]:
    segments = []
    if not raw_results or len(raw_results) == 0: return segments
    inner_results = raw_results[0] if isinstance(raw_results[0], list) else raw_results
    for seg in inner_results:
        try:
            if isinstance(seg, str):
                parts = seg.strip().split()
                if len(parts) >= 3:
                    segments.append((float(parts[0]), float(parts[1]), int(parts[2].replace('speaker_', ''))))
            elif isinstance(seg, (list, tuple)) and len(seg) >= 3:
                segments.append((float(seg[0]), float(seg[1]), int(seg[2])))
        except (ValueError, IndexError):
            pass
    return sorted(segments, key=lambda x: x[0])

def diarize_audio(audio: torch.Tensor, sr: int, chunk_duration: float = CHUNK_DURATION_S) -> List[Tuple[float, float, int]]:
    global sortformer_model
    total_samples = audio.shape[-1]
    chunk_samples = int(chunk_duration * sr)
    all_segments, offset = [], 0

    while offset < total_samples:
        end = min(offset + chunk_samples, total_samples)
        chunk = audio[:, offset:end]
        
        audio_np = chunk.squeeze(0).numpy() if chunk.dim() > 1 else chunk.numpy()
        raw = sortformer_model.diarize(audio=audio_np, sample_rate=sr, include_tensor_outputs=False)
        segs = parse_diarization_output(raw)
        
        offset_sec = offset / sr
        segs = [(s + offset_sec, e + offset_sec, spk) for s, e, spk in segs]
        
        if len(segs) > 2 and total_samples > chunk_samples:
            segs = segs[1:-1]
            
        all_segments.extend(segs)
        offset = end

    return sorted(all_segments, key=lambda x: x[0])


def filter_single_speaker_segments(segments: List[Tuple[float, float, int]], min_duration: float = 1.0, max_duration: float = 15.0) -> List[Tuple[float, float, int]]:
    filtered = []
    segments = sorted(segments, key=lambda x: x[0])
    for i, (start, end, spk) in enumerate(segments):
        dur = end - start
        if not (min_duration <= dur <= max_duration): continue
        overlap = False
        for j, (s2, e2, _) in enumerate(segments):
            if i != j and start < e2 and end > s2:
                overlap = True
                break
        if not overlap:
            filtered.append((start, end, spk))
    return filtered

def apply_eos_classification(audio: torch.Tensor, sr: int, segments: List[Tuple[float, float, int]], max_duration: float = 15.0) -> List[Tuple[float, float, int]]:
    global smart_vad
    if not smart_vad or not segments: return segments

    audio_np = audio.squeeze(0).numpy() if audio.dim() > 1 else audio.numpy()
    classified = []
    for s, e, spk in segments:
        segment_audio = audio_np[int(s * sr):min(int(e * sr), len(audio_np))]
        if len(segment_audio) == 0: continue
        pred = smart_vad.predict_endpoint(segment_audio)['prediction']
        classified.append((s, e, spk, pred))

    merged = []
    i = 0
    while i < len(classified):
        start, end, spk, pred = classified[i]
        if pred == 1: # EOS detected
            merged.append((start, end, spk))
            i += 1
            continue
        
        j = i + 1
        while j < len(classified):
            ns, ne, nspk, npred = classified[j]
            if nspk != spk or ne - start > max_duration: break
            end = ne
            j += 1
            if npred == 1: break

        if end - start >= 1.0:
            merged.append((start, end, spk))
        i = max(j, i + 1)
    return merged


def get_chunk_metrics(c_start: float, c_end: float, raw_segments: List[Tuple[float, float, int]]) -> Tuple[float, float, int]:
    chunk_dur = c_end - c_start
    if chunk_dur <= 0: return 0.0, 0.0, 0

    intervals = []
    speakers_in_chunk = set()
    
    for rs, re, spk in raw_segments:
        overlap_s = max(c_start, rs)
        overlap_e = min(c_end, re)
        if overlap_s < overlap_e:
            intervals.append([overlap_s, overlap_e])
            speakers_in_chunk.add(spk)

    intervals.sort(key=lambda x: x[0])
    if not intervals:
        return 100.0, round(chunk_dur, 2), 0

    merged_speech = []
    for interval in intervals:
        if not merged_speech:
            merged_speech.append(interval)
        else:
            prev = merged_speech[-1]
            if interval[0] <= prev[1]:
                prev[1] = max(prev[1], interval[1])
            else:
                merged_speech.append(interval)

    speech_dur = sum(e - s for s, e in merged_speech)
    silence_dur = max(0.0, chunk_dur - speech_dur)
    silence_pct = (silence_dur / chunk_dur) * 100

    gaps = [merged_speech[0][0] - c_start]
    for i in range(len(merged_speech) - 1):
        gaps.append(merged_speech[i+1][0] - merged_speech[i][1])
    gaps.append(c_end - merged_speech[-1][1])
    max_gap = max(gaps)

    return round(silence_pct, 2), round(max_gap, 2), len(speakers_in_chunk)


def cut_audio(audio: torch.Tensor, sr: int, final_segments: List[Tuple[float, float, int]], raw_segments: List[Tuple[float, float, int]], output_folder: str, album_id: str, episode_id: str, fmt: str = 'mp3', max_duration: float = 15.0) -> List[Dict]:
    os.makedirs(output_folder, exist_ok=True)
    results = []

    for start, end, spk in final_segments:
        dur = end - start
        if dur <= 0.5: continue
        
        s_sample, e_sample = int(start * sr), min(int(end * sr), audio.shape[-1])
        if e_sample <= s_sample: continue

        sil_pct, max_sil, unique_spk = get_chunk_metrics(start, end, raw_segments)

        segment = audio[:, s_sample:e_sample]
        fname = f"{start:.2f}_{end:.2f}_{album_id}_{episode_id}.{fmt}"
        out_path = os.path.join(output_folder, fname)
        torchaudio.save(out_path, segment, sr)

        results.append({
            'filepath': os.path.abspath(out_path),
            'speaker_id': spk,
            'start': round(start, 2),
            'end': round(end, 2),
            'total_duration': round(dur, 2),
            'playlist_id': album_id,
            'podcast_id': episode_id,
            'silence_percent': sil_pct,
            'max_silence_duration': max_sil,
            'is_single_speaker': unique_spk == 1
        })
    return results


def process_audio_file(path_audio: str, config: Dict[str, Any]) -> List[Dict]:
    limit_dur = config.get('duration', 15)
    chunk_duration = config.get('chunk_duration', CHUNK_DURATION_S)

    p_audio = Path(path_audio)
    album_id, episode_id = p_audio.parent.name, p_audio.stem
    episode_folder = p_audio.parent / episode_id

    try:
        audio, sr = torchaudio.load(path_audio)
    except Exception as e:
        logger.error(f"Broken file {path_audio}: {e}")
        return []

    total_audio_duration = audio.shape[-1] / sr

    try:
        if audio.shape[0] > 1: 
            audio = torch.mean(audio, dim=0, keepdim=True)

        raw_segments = diarize_audio(audio, sr, chunk_duration)
        if not raw_segments: return []

        if total_audio_duration <= limit_dur:
            sil_pct, max_sil, unique_spk = get_chunk_metrics(0.0, total_audio_duration, raw_segments)
            main_spk = raw_segments[0][2] if raw_segments else -1
            
            return [{
                'filepath': os.path.abspath(path_audio),
                'speaker_id': main_spk,
                'start': 0.0,
                'end': round(total_audio_duration, 2),
                'total_duration': round(total_audio_duration, 2),
                'playlist_id': album_id,
                'podcast_id': episode_id,
                'silence_percent': sil_pct,
                'max_silence_duration': max_sil,
                'is_single_speaker': unique_spk == 1
            }]

        clean_segments = filter_single_speaker_segments(raw_segments, min_duration=1.0, max_duration=limit_dur)
        final_segments = apply_eos_classification(audio, sr, clean_segments, max_duration=limit_dur)
        
        if not final_segments: return []

        seg_results = cut_audio(audio, sr, final_segments, raw_segments, str(episode_folder), album_id, episode_id, max_duration=limit_dur)

        if seg_results:
            logger.success(f"Processed {len(seg_results)} chunks from: {p_audio.name}")
            if p_audio.exists():
                os.remove(p_audio)
        
        return seg_results

    except Exception as e:
        logger.error(f"Processing error {path_audio}: {e}")
        return []
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def main(args):
    load_dotenv()
    if hf_key := os.environ.get('HF_TOKEN'): login(token=hf_key)

    config = load_config(args.config_path, 'preprocess')
    podcasts_path = Path(config.get('podcasts_path', '../../../podcasts'))
    num_workers_per_gpu = config.get('num_workers', 1)
    
    num_gpus = torch.cuda.device_count()
    total_workers = max(1, num_gpus * num_workers_per_gpu)

    csv_path = podcasts_path / 'balalaika.csv'
    existing_df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    raw_audio_paths = get_audio_paths(str(podcasts_path))
    paths_to_process = []
    
    chunk_pattern = re.compile(r'^\d+\.\d+_\d+\.\d+_') 

    for p_str in raw_audio_paths:
        p = Path(p_str)
        if chunk_pattern.match(p.name): continue
        paths_to_process.append(p)

    if not paths_to_process:
        logger.info("No new files to process.")
        return

    logger.info(f"Files to process: {len(paths_to_process)} on {num_gpus} GPU(s)")

    all_results = []
    files_per_gpu = [[] for _ in range(num_gpus)] if num_gpus > 0 else [paths_to_process]
    
    if num_gpus > 0:
        for i, p in enumerate(paths_to_process):
            files_per_gpu[i % num_gpus].append(p)

    for gpu_id in range(max(1, num_gpus)):
        gpu_files = files_per_gpu[gpu_id]
        if not gpu_files: continue

        logger.info(f"GPU:{gpu_id} processing {len(gpu_files)} files...")
        
        with ProcessPoolExecutor(
            max_workers=num_workers_per_gpu, 
            initializer=init_models, 
            initargs=(gpu_id, config)
        ) as executor:
            futures = [executor.submit(process_audio_file, str(p), config) for p in gpu_files]
            for future in tqdm(as_completed(futures), total=len(gpu_files), desc=f"GPU {gpu_id}"):
                try:
                    res = future.result()
                    if res: all_results.extend(res)
                except Exception as e:
                    logger.error(f"Task error: {e}")

    if all_results:
        new_df = pd.DataFrame(all_results)
        
        if not existing_df.empty:
            df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=['filepath'], keep='last')
        else:
            df = new_df

        base_cols = [
            'filepath', 'speaker_id', 'start', 'end', 'total_duration', 
            'playlist_id', 'podcast_id', 'silence_percent', 
            'max_silence_duration', 'is_single_speaker'
        ]
        
        cols = [c for c in base_cols if c in df.columns] + [c for c in df.columns if c not in base_cols]
        df = df[cols]
        
        df.to_csv(csv_path, index=False)
        logger.success(f"Successfully processed {len(all_results)} samples. Metadata saved to {csv_path}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file")
    main(parser.parse_args())