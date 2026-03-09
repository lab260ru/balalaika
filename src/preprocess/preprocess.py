import argparse
import os
import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torchaudio
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login

from src.utils.utils import load_config, get_audio_paths
from src.libs.smart_turn.offline_svad import SmartVAD

from dotenv import load_dotenv

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

load_dotenv()

CHUNK_DURATION_S = 15 * 60  # 15 minutes — Sortformer OOM limit on 48GB

sortformer_model = None
smart_vad = None


def initializer_wrapper(config: Dict[str, Any], gpus_count: int):
    worker_id = multiprocessing.current_process()._identity[0] - 1
    gpu_id = worker_id % gpus_count if gpus_count > 0 else 0
    init_models(gpu_id, config)


def init_models(gpu_id: int, config: Dict[str, Any]):
    global sortformer_model, smart_vad

    device = f"cuda:{gpu_id}" 
    providers = [
        ("TensorrtExecutionProvider", {
            "trt_max_workspace_size": 6 * 1024**3, 
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "./trt_cache",  
        }),
        ("CUDAExecutionProvider",
            {
                "device_id": gpu_id
            }
        ),
        "CPUExecutionProvider"
    ]
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    try:
        from src.preprocess.sortformer_onnx import Sortformer
        from src.preprocess.sortformer_onnx import DiarizationConfig
    except ImportError:
        logger.error("Sortformer model not found")
        raise

    config = DiarizationConfig()
    sortformer_model = Sortformer(model_path=config.get('sortformer_model'), config=config, providers=providers)

    sortformer_model = sortformer_model.to(device)
    sortformer_model.eval()

    vad_args = config.get('vad_args', {})
    smart_vad = SmartVAD(
        smart_vad_model=vad_args.get('smart_vad_model', './models/smart-turn-v3.0.onnx'),
        smart_vad_threshold=vad_args.get('smart_vad_threshold', 0.4),
        device=device
    )

    logger.info(f"Sortformer + SmartVAD initialized on {device}")


# ---------------------------------------------------------------------------
#  Sortformer diarization
# ---------------------------------------------------------------------------

def parse_diarization_output(raw_results) -> List[Tuple[float, float, int]]:
    """Parse Sortformer diarize() output into (start, end, speaker_id) tuples."""
    segments = []
    if not raw_results or len(raw_results) == 0:
        return segments

    file_results = raw_results[0] if raw_results else []
    for seg in file_results:
        try:
            if isinstance(seg, str):
                cleaned = seg.strip('[] \n')
                parts = [p.strip() for p in cleaned.split(',')]
                if len(parts) >= 3:
                    segments.append((float(parts[0]), float(parts[1]), int(float(parts[2]))))
            elif isinstance(seg, (list, tuple)) and len(seg) >= 3:
                segments.append((float(seg[0]), float(seg[1]), int(seg[2])))
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse segment {seg}: {e}")
    return sorted(segments, key=lambda x: x[0])


def _diarize_chunk(
    chunk_audio: torch.Tensor,
    sr: int,
    offset: float
) -> List[Tuple[float, float, int]]:
    """Diarize a single audio chunk via Sortformer temp-file interface."""
    global sortformer_model

    fd, temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)

    try:
        torchaudio.save(temp_path, chunk_audio, sr)
        raw = sortformer_model.diarize(audio=temp_path, batch_size=1, verbose=False)
        logger.debug(f"Raw diarization output: {raw}")
        segments = parse_diarization_output(raw)
        return [(s + offset, e + offset, spk) for s, e, spk in segments]
    except Exception as e:
        logger.error(f"Diarization chunk failed: {e}")
        return []
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def diarize_audio(
    audio: torch.Tensor,
    sr: int,
    chunk_duration: float = CHUNK_DURATION_S
) -> List[Tuple[float, float, int]]:
    """Run Sortformer diarization; split into <=chunk_duration pieces if needed.
    For each chunk the first and last segments are discarded (edge artefacts).
    """
    total_samples = audio.shape[-1]
    total_duration = total_samples / sr

    if total_duration <= chunk_duration:
        return _diarize_chunk(audio, sr, offset=0.0)

    logger.info(f"Audio {total_duration:.0f}s > {chunk_duration:.0f}s, splitting into chunks")
    chunk_samples = int(chunk_duration * sr)
    all_segments: List[Tuple[float, float, int]] = []
    offset = 0

    while offset < total_samples:
        end = min(offset + chunk_samples, total_samples)
        chunk = audio[:, offset:end]
        offset_sec = offset / sr

        segments = _diarize_chunk(chunk, sr, offset_sec)

        if len(segments) > 2:
            segments = segments[1:-1]
        else:
            offset = end
            continue

        all_segments.extend(segments)
        offset = end

    return sorted(all_segments, key=lambda x: x[0])


# ---------------------------------------------------------------------------
#  Filtering & EOS classification
# ---------------------------------------------------------------------------

def filter_single_speaker_segments(
    segments: List[Tuple[float, float, int]],
    min_duration: float = 1.0,
    max_duration: float = 15.0
) -> List[Tuple[float, float, int]]:
    """Keep only non-overlapping segments within duration limits."""
    if not segments:
        return []

    segments = sorted(segments, key=lambda x: x[0])
    filtered = []

    for i, (start, end, spk) in enumerate(segments):
        dur = end - start
        if dur < min_duration or dur > max_duration:
            continue

        has_overlap = any(
            j != i and start < e2 and end > s2
            for j, (s2, e2, _) in enumerate(segments)
        )
        if not has_overlap:
            filtered.append((start, end, spk))

    return filtered


def apply_eos_classification(
    audio: torch.Tensor,
    sr: int,
    segments: List[Tuple[float, float, int]],
    max_duration: float = 15.0
) -> List[Tuple[float, float, int]]:
    """Run EOS classification; merge incomplete segments with the next
    same-speaker segment when possible (respecting *max_duration*)."""
    global smart_vad
    if not smart_vad or not segments:
        return segments

    audio_np = audio.squeeze(0).numpy() if audio.dim() > 1 else audio.numpy()

    classified = []
    for start, end, spk in segments:
        s_sample = int(start * sr)
        e_sample = min(int(end * sr), len(audio_np))
        seg_audio = audio_np[s_sample:e_sample]
        result = smart_vad.predict_endpoint(seg_audio)
        classified.append((start, end, spk, result['prediction']))

    merged: List[Tuple[float, float, int]] = []
    i = 0
    while i < len(classified):
        start, end, spk, pred = classified[i]

        if pred == 1:
            merged.append((start, end, spk))
            i += 1
            continue

        j = i + 1
        while j < len(classified):
            ns, ne, nspk, npred = classified[j]
            if nspk != spk or ne - start > max_duration:
                break
            end = ne
            j += 1
            if npred == 1:
                break

        if end - start >= 1.0:
            merged.append((start, end, spk))
        i = max(j, i + 1)

    return merged


# ---------------------------------------------------------------------------
#  Silence metrics
# ---------------------------------------------------------------------------

def calculate_silence_metrics(
    segments: List[Tuple[float, float, int]],
    total_duration: float
) -> Tuple[float, float]:
    """Derive silence percentage and max silence duration from diarization segments."""
    if not segments:
        return 100.0, total_duration

    sorted_segs = sorted(segments, key=lambda x: x[0])
    total_speech = sum(e - s for s, e, _ in sorted_segs)
    silence_pct = ((total_duration - total_speech) / total_duration * 100) if total_duration > 0 else 0

    gaps = [sorted_segs[0][0]]
    for i in range(len(sorted_segs) - 1):
        gaps.append(sorted_segs[i + 1][0] - sorted_segs[i][1])
    gaps.append(total_duration - sorted_segs[-1][1])
    max_silence = max(gaps)

    return round(silence_pct, 2), round(max_silence, 2)


# ---------------------------------------------------------------------------
#  RTTM & audio cutting
# ---------------------------------------------------------------------------

def save_rttm(segments: List[Tuple[float, float, int]], audio_path: str):
    rttm_path = str(Path(audio_path).with_suffix('.rttm'))
    file_id = Path(audio_path).stem
    with open(rttm_path, 'w') as f:
        for start, end, spk in segments:
            f.write(
                f"SPEAKER {file_id} 1 {start:.3f} {end - start:.3f} "
                f"<NA> <NA> speaker_{spk} <NA> <NA>\n"
            )


def cut_audio(
    audio: torch.Tensor,
    sr: int,
    segments: List[Tuple[float, float, int]],
    output_folder: str,
    album_id: str,
    episode_id: str,
    fmt: str = 'mp3',
    max_duration: float = 15.0
) -> List[Dict]:
    os.makedirs(output_folder, exist_ok=True)
    results = []

    for start, end, spk in segments:
        if end - start <= max_duration / 5:
            continue

        s_sample = int(start * sr)
        e_sample = min(int(end * sr), audio.shape[-1])
        if e_sample <= s_sample:
            continue

        segment = audio[:, s_sample:e_sample]
        fname = f"{start:.2f}_{end:.2f}_{album_id}_{episode_id}.{fmt}"
        out_path = os.path.join(output_folder, fname)
        torchaudio.save(out_path, segment, sr)

        results.append({
            'filepath': os.path.abspath(out_path),
            'is_single_speaker': True,
            'speaker_id': spk,
            'start': f"{start:.2f}",
            'end': f"{end:.2f}",
            'total_duration': round(end - start, 2),
            'playlist_id': album_id,
            'podcast_id': episode_id,
        })

    return results


# ---------------------------------------------------------------------------
#  Per-file processing
# ---------------------------------------------------------------------------

def process_audio_file(path_audio: str, config: Dict[str, Any]) -> List[Dict]:
    """Process a single audio file: Sortformer diarization -> EOS check -> cut."""
    global sortformer_model, smart_vad

    duration = config.get('duration', 15)
    chunk_duration = config.get('chunk_duration', CHUNK_DURATION_S)

    album_id = os.path.basename(os.path.dirname(path_audio))
    episode_id = os.path.splitext(os.path.basename(path_audio))[0]
    episode_folder = os.path.join(os.path.dirname(path_audio), episode_id)

    try:
        audio, sr = torchaudio.load(path_audio)
    except Exception as e:
        logger.error(f"Broken file {path_audio}: {e}")
        if os.path.exists(path_audio):
            os.remove(path_audio)
        return []

    total_duration = audio.shape[-1] / sr
    if total_duration <= duration:
        return []

    try:
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # 1. Sortformer diarization
        all_segments = diarize_audio(audio, sr, chunk_duration)

        if not all_segments:
            logger.warning(f"No speech segments in {path_audio}, removing")
            os.remove(path_audio)
            return []

        save_rttm(all_segments, path_audio)

        # 2. Filter: single-speaker, no overlap, duration bounds
        filtered = filter_single_speaker_segments(
            all_segments, min_duration=1.0, max_duration=duration
        )

        if not filtered:
            logger.warning(f"No clean single-speaker segments in {path_audio}, removing")
            os.remove(path_audio)
            return []
        
        logger.debug(f"Filtered segments: {filtered}")
        # 3. EOS semantic classification + merging incomplete segments
        filtered = apply_eos_classification(audio, sr, filtered, max_duration=duration)

        if not filtered:
            logger.warning(f"No segments after EOS filtering in {path_audio}, removing")
            # os.remove(path_audio)
            return []

        # 4. Silence metrics (derived from full diarization output)
        silence_pct, max_sil = calculate_silence_metrics(all_segments, total_duration)

        # 5. Cut audio
        seg_results = cut_audio(
            audio=audio, sr=sr, segments=filtered,
            output_folder=episode_folder, album_id=album_id,
            episode_id=episode_id, fmt='mp3', max_duration=duration
        )

        for r in seg_results:
            r['silence_percent'] = silence_pct
            r['max_silence_duration'] = max_sil

        logger.success(f"Processed {len(seg_results)} segments: {path_audio}")

    except Exception as e:
        logger.error(f"Processing error {path_audio}: {e}")
        return []
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if os.path.exists(episode_folder) and os.listdir(episode_folder):
        os.remove(path_audio)
        logger.info(f"Original file deleted: {path_audio}")

    return seg_results


# ---------------------------------------------------------------------------
#  Main entry
# ---------------------------------------------------------------------------

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
    num_gpus = torch.cuda.device_count()
    workers_per_gpu = config.get('num_workers', 1)

    if num_gpus > 0:
        total_workers = num_gpus * workers_per_gpu
        logger.info(
            f"Found {num_gpus} GPU(s), {workers_per_gpu} worker(s)/GPU, "
            f"total: {total_workers}"
        )
    else:
        total_workers = workers_per_gpu
        logger.warning("No GPUs available, using CPU")

    audio_paths = get_audio_paths(podcasts_path)
    if not audio_paths:
        logger.info("No audio files found for processing.")
        return

    logger.info(f"""
        Running Sortformer segmentation pipeline:
        Podcasts path: {podcasts_path}
        Max segment duration: {duration}s
        Total workers: {total_workers}
        Files to process: {len(audio_paths)}
    """)

    all_results: List[Dict] = []

    with ProcessPoolExecutor(
        max_workers=total_workers,
        initializer=initializer_wrapper,
        initargs=(config, num_gpus)
    ) as executor:
        futures = [
            executor.submit(process_audio_file, str(p), config)
            for p in audio_paths
        ]

        for future in tqdm(as_completed(futures), total=len(audio_paths),
                           desc="Sortformer Processing"):
            try:
                results = future.result()
                if results:
                    all_results.extend(results)
            except Exception as e:
                logger.error(f"Task error: {e}")

    if all_results:
        csv_path = Path(podcasts_path) / 'balalaika.csv'
        df = pd.DataFrame(all_results)

        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=['filepath'], keep='last')

        df.to_csv(csv_path, index=False)
        logger.success(f"Saved metadata for {len(all_results)} segments to {csv_path}")

    logger.info("All files have been processed.")


if __name__ == "__main__":
    torchaudio.set_audio_backend('soundfile')
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(
        description="Audio segmentation using Sortformer diarization + Smart VAD EOS"
    )
    parser.add_argument("--config_path", type=str, help="Path to YAML config file")
    args = parser.parse_args()
    main(args)
