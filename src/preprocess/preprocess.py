"""Sortformer (ONNX) diarization + Smart-Turn refinement + chunk export.

Production notes:

* **Audio quality is preserved across chunking.** Each chunk is written with
  the same container as its source file by default (``chunk_format: auto``).
  FLAC inputs stay FLAC, WAV stays WAV — no silent transcoding to MP3 like the
  earlier default. Operators can pin a specific extension via the
  ``chunk_format`` config key (``flac``/``wav``/``mp3``/``ogg``/``opus``).
* **Filter audit is recorded.** After all workers finish, the stage logs
  ``files_in/out`` and ``hours_in/out`` to ``filter_summary.csv`` so the final
  report can attribute removed audio to this stage.
* **Per-stage log file.** ``setup_logging`` initialises a stderr sink and a
  rotating file sink so long runs never lose log lines.
"""

import argparse
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torchaudio
from dotenv import load_dotenv
from huggingface_hub import login
from loguru import logger
from tqdm import tqdm

from src.libs.smart_turn.offline_svad import SmartVAD
from src.utils.audit import record_stage_summary, safe_audio_duration
from src.utils.logging_setup import setup_logging
from src.utils.runtime_env import runtime_cfg
from src.utils.utils import get_audio_paths, load_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

DEFAULT_CHUNK_DURATION_S = 15 * 60
DEFAULT_MIN_SEGMENT_DURATION_S = 1.0
DEFAULT_MIN_SAVE_DURATION_S = 0.5

LOSSLESS_EXTENSIONS = {".flac", ".wav"}
SUPPORTED_CHUNK_EXTS = {"flac", "wav", "mp3", "ogg", "opus"}

sortformer_model = None
smart_vad = None


def get_providers(cuda_id: int, config_path: Optional[str] = None) -> list:
    rt = runtime_cfg(config_path)
    cache_root = Path(str(rt["trt_cache_path"])) / f"trt_cache_{cuda_id}"
    cache_root.mkdir(parents=True, exist_ok=True)
    return [
        ("TensorrtExecutionProvider", {
            "device_id": cuda_id,
            "trt_max_workspace_size": int(rt["trt_workspace_bytes"]),
            "trt_fp16_enable": bool(rt["trt_fp16"]),
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache_root),
        }),
        ("CUDAExecutionProvider", {"device_id": cuda_id}),
    ]


def init_models(gpu_id: int, config: Dict[str, Any], config_path: Optional[str] = None):
    global sortformer_model, smart_vad
    device = f"cuda:{gpu_id}"
    providers = get_providers(gpu_id, config_path)

    try:
        from src.preprocess.sortformer_onnx import DiarizationConfig, Sortformer
    except ImportError:
        logger.error("Sortformer module or Sortformer class not found in src.preprocess.sortformer_onnx")
        raise

    model_config = DiarizationConfig()
    sortformer_model = Sortformer(
        model_path=config.get('sortformer_model'),
        config=model_config,
        providers=providers,
    )

    vad_args = config.get('vad_args', {})
    smart_vad_model = vad_args.get('smart_vad_model')
    if not smart_vad_model:
        raise ValueError(
            "preprocess.vad_args.smart_vad_model must be set in config; "
            "no path-style default is used to avoid silent fallbacks."
        )
    smart_vad = SmartVAD(
        smart_vad_model=smart_vad_model,
        smart_vad_threshold=vad_args.get('smart_vad_threshold', 0.4),
        device=device,
    )
    logger.info(f"Models initialized on {device}")


def parse_diarization_output(raw_results) -> List[Tuple[float, float, int]]:
    segments = []
    if not raw_results or len(raw_results) == 0:
        return segments
    inner_results = raw_results[0] if isinstance(raw_results[0], list) else raw_results
    for seg in inner_results:
        try:
            if isinstance(seg, str):
                parts = seg.strip().split()
                if len(parts) >= 3:
                    segments.append(
                        (float(parts[0]), float(parts[1]), int(parts[2].replace('speaker_', '')))
                    )
            elif isinstance(seg, (list, tuple)) and len(seg) >= 3:
                segments.append((float(seg[0]), float(seg[1]), int(seg[2])))
        except (ValueError, IndexError):
            pass
    return sorted(segments, key=lambda x: x[0])


def diarize_audio(audio: torch.Tensor, sr: int, chunk_duration: float = DEFAULT_CHUNK_DURATION_S) -> List[Tuple[float, float, int]]:
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


def filter_single_speaker_segments(
    segments: List[Tuple[float, float, int]],
    min_duration: float = DEFAULT_MIN_SEGMENT_DURATION_S,
    max_duration: float = 15.0,
) -> List[Tuple[float, float, int]]:
    filtered = []
    segments = sorted(segments, key=lambda x: x[0])
    for i, (start, end, spk) in enumerate(segments):
        dur = end - start
        if not (min_duration <= dur <= max_duration):
            continue
        overlap = False
        for j, (s2, e2, _) in enumerate(segments):
            if i != j and start < e2 and end > s2:
                overlap = True
                break
        if not overlap:
            filtered.append((start, end, spk))
    return filtered


def apply_eos_classification(
    audio: torch.Tensor,
    sr: int,
    segments: List[Tuple[float, float, int]],
    max_duration: float = 15.0,
    min_duration: float = DEFAULT_MIN_SEGMENT_DURATION_S,
) -> List[Tuple[float, float, int]]:
    global smart_vad
    if not smart_vad or not segments:
        return segments

    audio_np = audio.squeeze(0).numpy() if audio.dim() > 1 else audio.numpy()
    classified = []
    for s, e, spk in segments:
        segment_audio = audio_np[int(s * sr):min(int(e * sr), len(audio_np))]
        if len(segment_audio) == 0:
            continue
        pred = smart_vad.predict_endpoint(segment_audio)['prediction']
        classified.append((s, e, spk, pred))

    merged = []
    i = 0
    while i < len(classified):
        start, end, spk, pred = classified[i]
        if pred == 1:  # EOS detected
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

        if end - start >= min_duration:
            merged.append((start, end, spk))
        i = max(j, i + 1)
    return merged


def get_chunk_metrics(
    c_start: float,
    c_end: float,
    raw_segments: List[Tuple[float, float, int]],
) -> Tuple[float, float, int]:
    chunk_dur = c_end - c_start
    if chunk_dur <= 0:
        return 0.0, 0.0, 0

    intervals = []
    speakers_in_chunk = set()

    for rs, re_, spk in raw_segments:
        overlap_s = max(c_start, rs)
        overlap_e = min(c_end, re_)
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
        gaps.append(merged_speech[i + 1][0] - merged_speech[i][1])
    gaps.append(c_end - merged_speech[-1][1])
    max_gap = max(gaps)

    return round(silence_pct, 2), round(max_gap, 2), len(speakers_in_chunk)


def resolve_chunk_format(source_path: Path, chunk_format: str) -> str:
    """Pick the chunk extension for a source file.

    ``auto`` preserves the source extension whenever it's a format we know how
    to write back through ``torchaudio.save``. Otherwise we pin lossless FLAC
    to avoid quality loss from unsolicited transcoding.
    """
    fmt = (chunk_format or "auto").strip().lower().lstrip(".")
    if fmt == "auto":
        src_ext = source_path.suffix.lower().lstrip(".")
        return src_ext if src_ext in SUPPORTED_CHUNK_EXTS else "flac"
    if fmt not in SUPPORTED_CHUNK_EXTS:
        logger.warning(
            f"Unknown chunk_format='{chunk_format}', falling back to source extension."
        )
        src_ext = source_path.suffix.lower().lstrip(".")
        return src_ext if src_ext in SUPPORTED_CHUNK_EXTS else "flac"
    return fmt


def _save_audio_chunk(out_path: str, segment: torch.Tensor, sr: int, fmt: str) -> None:
    """Write a chunk preserving as much fidelity as possible.

    For lossless containers (flac/wav) ``torchaudio.save`` is bit-exact at the
    requested sample format. For lossy containers we still let torchaudio pick
    a sane default rather than hard-coding bitrates we can't tune per system.
    """
    fmt = fmt.lower().lstrip(".")
    if fmt in {"flac", "wav"}:
        torchaudio.save(out_path, segment, sr, format=fmt)
    else:
        torchaudio.save(out_path, segment, sr, format=fmt)


def cut_audio(
    audio: torch.Tensor,
    sr: int,
    final_segments: List[Tuple[float, float, int]],
    raw_segments: List[Tuple[float, float, int]],
    output_folder: str,
    album_id: str,
    episode_id: str,
    fmt: str = 'flac',
    max_duration: float = 15.0,
    min_save_duration: float = DEFAULT_MIN_SAVE_DURATION_S,
) -> List[Dict]:
    os.makedirs(output_folder, exist_ok=True)
    results = []

    for start, end, spk in final_segments:
        dur = end - start
        if dur <= min_save_duration:
            continue

        s_sample, e_sample = int(start * sr), min(int(end * sr), audio.shape[-1])
        if e_sample <= s_sample:
            continue

        sil_pct, max_sil, unique_spk = get_chunk_metrics(start, end, raw_segments)

        segment = audio[:, s_sample:e_sample]
        fname = f"{start:.2f}_{end:.2f}_{album_id}_{episode_id}.{fmt}"
        out_path = os.path.join(output_folder, fname)
        _save_audio_chunk(out_path, segment, sr, fmt)

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
            'is_single_speaker': unique_spk == 1,
        })
    return results


def process_audio_file(path_audio: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single source recording.

    Returns a dict with ``segments`` (list of metadata dicts for each chunk
    written) and ``source_duration_s`` so the parent process can build the
    audit summary without re-probing files.
    """
    limit_dur = config.get('duration', 15)
    chunk_duration = config.get('chunk_duration', DEFAULT_CHUNK_DURATION_S)
    chunk_format_cfg = config.get('chunk_format', 'auto')
    min_segment_dur = float(config.get('min_segment_duration', DEFAULT_MIN_SEGMENT_DURATION_S))
    min_save_dur = float(config.get('min_save_duration', DEFAULT_MIN_SAVE_DURATION_S))

    p_audio = Path(path_audio)
    album_id, episode_id = p_audio.parent.name, p_audio.stem
    episode_folder = p_audio.parent / episode_id

    chunk_fmt = resolve_chunk_format(p_audio, chunk_format_cfg)

    try:
        audio, sr = torchaudio.load(path_audio)
    except Exception as e:
        logger.error(f"Broken file {path_audio}: {e}")
        return {"segments": [], "source_duration_s": 0.0}

    total_audio_duration = audio.shape[-1] / sr

    try:
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        raw_segments = diarize_audio(audio, sr, chunk_duration)
        if not raw_segments:
            return {"segments": [], "source_duration_s": total_audio_duration}

        if total_audio_duration <= limit_dur:
            sil_pct, max_sil, unique_spk = get_chunk_metrics(0.0, total_audio_duration, raw_segments)
            main_spk = raw_segments[0][2] if raw_segments else -1

            return {
                "segments": [{
                    'filepath': os.path.abspath(path_audio),
                    'speaker_id': main_spk,
                    'start': 0.0,
                    'end': round(total_audio_duration, 2),
                    'total_duration': round(total_audio_duration, 2),
                    'playlist_id': album_id,
                    'podcast_id': episode_id,
                    'silence_percent': sil_pct,
                    'max_silence_duration': max_sil,
                    'is_single_speaker': unique_spk == 1,
                }],
                "source_duration_s": total_audio_duration,
            }

        clean_segments = filter_single_speaker_segments(
            raw_segments, min_duration=min_segment_dur, max_duration=limit_dur
        )
        final_segments = apply_eos_classification(
            audio, sr, clean_segments, max_duration=limit_dur,
            min_duration=min_segment_dur,
        )

        if not final_segments:
            return {"segments": [], "source_duration_s": total_audio_duration}

        seg_results = cut_audio(
            audio,
            sr,
            final_segments,
            raw_segments,
            str(episode_folder),
            album_id,
            episode_id,
            fmt=chunk_fmt,
            max_duration=limit_dur,
            min_save_duration=min_save_dur,
        )

        if seg_results:
            logger.success(
                f"Processed {len(seg_results)} chunks ({chunk_fmt}) from: {p_audio.name}"
            )
            if p_audio.exists():
                os.remove(p_audio)

        return {"segments": seg_results, "source_duration_s": total_audio_duration}

    except Exception as e:
        logger.error(f"Processing error {path_audio}: {e}")
        return {"segments": [], "source_duration_s": total_audio_duration}
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _measure_source_hours(paths: List[Path]) -> float:
    """Pre-flight scan to estimate hours available before processing."""
    total = 0.0
    for p in paths:
        total += safe_audio_duration(p)
    return total / 3600.0


def main(args):
    setup_logging("preprocess", log_dir=args.log_dir)
    load_dotenv()
    if hf_key := os.environ.get('HF_TOKEN'):
        login(token=hf_key)

    config = load_config(args.config_path, 'preprocess')
    podcasts_path = Path(config.get('podcasts_path', '../../../podcasts'))
    num_workers_per_gpu = config.get('num_workers', 1)

    chunk_format_cfg = config.get('chunk_format', 'auto')
    logger.info(f"Chunk format policy: '{chunk_format_cfg}' (lossless input stays lossless).")

    num_gpus = torch.cuda.device_count()
    total_workers = max(1, num_gpus * num_workers_per_gpu)
    logger.info(f"GPUs: {num_gpus}, workers/GPU: {num_workers_per_gpu}, total workers: {total_workers}")

    csv_path = podcasts_path / 'balalaika.csv'
    existing_df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    raw_audio_paths = get_audio_paths(str(podcasts_path))
    paths_to_process: List[Path] = []

    chunk_pattern = re.compile(r'^\d+\.\d+_\d+\.\d+_')

    for p_str in raw_audio_paths:
        p = Path(p_str)
        if chunk_pattern.match(p.name):
            continue
        paths_to_process.append(p)

    if not paths_to_process:
        logger.info("No new files to process.")
        return

    logger.info(f"Files to process: {len(paths_to_process)} on {num_gpus} GPU(s)")

    hours_in = _measure_source_hours(paths_to_process)
    logger.info(f"Source audio total: {hours_in:.2f}h across {len(paths_to_process)} files")

    all_results: List[Dict[str, Any]] = []
    files_per_gpu: List[List[Path]] = (
        [[] for _ in range(num_gpus)] if num_gpus > 0 else [paths_to_process]
    )

    if num_gpus > 0:
        for i, p in enumerate(paths_to_process):
            files_per_gpu[i % num_gpus].append(p)

    for gpu_id in range(max(1, num_gpus)):
        gpu_files = files_per_gpu[gpu_id]
        if not gpu_files:
            continue

        logger.info(f"GPU:{gpu_id} processing {len(gpu_files)} files...")

        with ProcessPoolExecutor(
            max_workers=num_workers_per_gpu,
            initializer=init_models,
            initargs=(gpu_id, config, args.config_path),
        ) as executor:
            futures = [executor.submit(process_audio_file, str(p), config) for p in gpu_files]
            for future in tqdm(as_completed(futures), total=len(gpu_files), desc=f"GPU {gpu_id}"):
                try:
                    res = future.result()
                    if res and res.get("segments"):
                        all_results.extend(res["segments"])
                except Exception as e:
                    logger.error(f"Task error: {e}")

    hours_out = sum(float(r.get('total_duration', 0.0)) for r in all_results) / 3600.0

    if all_results:
        new_df = pd.DataFrame(all_results)

        if not existing_df.empty:
            df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(
                subset=['filepath'], keep='last'
            )
        else:
            df = new_df

        base_cols = [
            'filepath', 'speaker_id', 'start', 'end', 'total_duration',
            'playlist_id', 'podcast_id', 'silence_percent',
            'max_silence_duration', 'is_single_speaker',
        ]

        cols = [c for c in base_cols if c in df.columns] + [c for c in df.columns if c not in base_cols]
        df = df[cols]

        df.to_csv(csv_path, index=False)
        logger.success(
            f"Successfully processed {len(all_results)} samples. Metadata saved to {csv_path}"
        )

    record_stage_summary(
        podcasts_path=podcasts_path,
        stage="preprocess",
        files_in=len(paths_to_process),
        files_out=len(all_results),
        hours_in=hours_in,
        hours_out=hours_out,
        params={
            "duration": config.get("duration", 15),
            "chunk_duration": config.get("chunk_duration", DEFAULT_CHUNK_DURATION_S),
            "chunk_format": chunk_format_cfg,
            "min_segment_duration": config.get(
                "min_segment_duration", DEFAULT_MIN_SEGMENT_DURATION_S
            ),
            "min_save_duration": config.get(
                "min_save_duration", DEFAULT_MIN_SAVE_DURATION_S
            ),
        },
    )


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    main(parser.parse_args())
