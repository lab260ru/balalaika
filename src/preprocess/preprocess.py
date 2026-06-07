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
* **Per-GPU partial CSVs.** Each GPU subprocess streams chunk metadata into
  its own ``preprocess_part_<gpu_id>.csv`` row-by-row (``flush()`` after every
  row). A background merger in the main process folds those partials into
  ``balalaika.csv`` every ``csv.flush_every_rows`` rows (default 10 000), and
  a final ``absorb_partial_csvs`` runs at stage end. A Ctrl+C / OOM kill
  therefore loses at most the rows from the last in-flight chunk.
"""

import argparse
import multiprocessing
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
from dotenv import load_dotenv
from huggingface_hub import login
from loguru import logger
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder
from tqdm import tqdm

from src.libs.smart_turn.offline_svad import SmartVAD
from src.preprocess.audio_postprocessing import (
    fused_audio_preprocessing_enabled,
    postprocess_audio_tensor,
)
from src.utils.audit import record_stage_summary, safe_audio_duration
from src.utils.csv_manager import (
    PartialCsvWriter,
    PeriodicCsvMerger,
    absorb_partial_csvs,
    ensure_main_csv,
    discover_audio_paths,
    load_csv_settings,
)
from src.utils.datasets.preprocess import create_diarization_dataloader
from src.utils.gpu import apply_torch_perf_defaults, get_onnx_providers
from src.utils.logging_setup import setup_logging
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config

apply_torch_perf_defaults()

DEFAULT_CHUNK_DURATION_S = 15 * 60
DEFAULT_MIN_SEGMENT_DURATION_S = 1.0
DEFAULT_MIN_SAVE_DURATION_S = 0.5
DEFAULT_MAX_MERGE_GAP_S = 0.5

LOSSLESS_EXTENSIONS = {".flac", ".wav"}
SUPPORTED_CHUNK_EXTS = {"flac", "wav", "mp3", "ogg", "opus"}

PARTIAL_PREFIX = "preprocess"
PARTIAL_FIELDS = (
    "filepath",
    "speaker_id",
    "start",
    "end",
    "total_duration",
    "playlist_id",
    "podcast_id",
    "silence_percent",
    "max_silence_duration",
    "is_single_speaker",
)
FUSED_PARTIAL_FIELDS = PARTIAL_FIELDS + ("crest_factor", "loudness_normalized")

sortformer_model = None
smart_vad = None


def init_models(gpu_id: int, config: Dict[str, Any], config_path: Optional[str] = None):
    global sortformer_model, smart_vad
    device = f"cuda:{gpu_id}"
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    providers = get_onnx_providers(gpu_id, use_tensorrt=True, config_path=config_path)

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
        device=device,
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
        resample_rate=int(vad_args.get('smart_vad_sample_rate', 16_000)),
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

        inference_started_at = time.perf_counter()
        raw = sortformer_model.diarize(audio=chunk, sample_rate=sr, include_tensor_outputs=False)
        logger.debug(
            f"perf model=sortformer event=inference "
            f"seconds={time.perf_counter() - inference_started_at:.6f} "
            f"sample_rate={sr} frames={int(chunk.shape[-1])}"
        )
        segs = parse_diarization_output(raw)

        offset_sec = offset / sr
        segs = [(s + offset_sec, e + offset_sec, spk) for s, e, spk in segs]

        if len(segs) > 2 and total_samples > chunk_samples:
            segs = segs[1:-1]

        all_segments.extend(segs)
        offset = end

    return sorted(all_segments, key=lambda x: x[0])


def build_single_speaker_timeline(
    segments: List[Tuple[float, float, int]],
    max_duration: float = 15.0,
) -> List[Tuple[float, float, int]]:
    """Build a non-overlapping speaker timeline without throwing away long turns.

    Sortformer can emit overlaps and very long same-speaker turns. The old
    path dropped those chunks before EOS refinement, which removed useful
    speech. Here overlaps between different speakers are trimmed at the
    midpoint, same-speaker overlaps are merged, and long turns are split into
    saveable windows.
    """
    timeline: List[Tuple[float, float, int]] = []
    valid_segments = sorted(
        ((float(s), float(e), int(spk)) for s, e, spk in segments if e > s),
        key=lambda x: (x[0], x[1]),
    )

    for start, end, spk in valid_segments:
        if timeline:
            prev_start, prev_end, prev_spk = timeline[-1]
            if spk == prev_spk and start <= prev_end:
                timeline[-1] = (prev_start, max(prev_end, end), prev_spk)
                continue
            if start < prev_end:
                split_at = (start + prev_end) / 2.0
                if split_at > prev_start:
                    timeline[-1] = (prev_start, split_at, prev_spk)
                else:
                    timeline.pop()
                start = split_at

        if end > start:
            timeline.append((start, end, spk))

    if max_duration <= 0:
        return timeline

    split_timeline: List[Tuple[float, float, int]] = []
    for start, end, spk in timeline:
        cursor = start
        while end - cursor > max_duration:
            split_timeline.append((cursor, cursor + max_duration, spk))
            cursor += max_duration
        if end > cursor:
            split_timeline.append((cursor, end, spk))

    return split_timeline


def apply_eos_classification(
    audio: torch.Tensor,
    sr: int,
    segments: List[Tuple[float, float, int]],
    max_duration: float = 15.0,
    min_duration: float = DEFAULT_MIN_SEGMENT_DURATION_S,
    max_merge_gap: float = DEFAULT_MAX_MERGE_GAP_S,
) -> List[Tuple[float, float, int]]:
    global smart_vad
    if not segments:
        return segments

    vad_sr = int(getattr(smart_vad, "sample_rate", sr)) if smart_vad else sr
    if sr != vad_sr:
        vad_device = getattr(smart_vad, "device", "cpu") if smart_vad else "cpu"
        vad_audio = torchaudio.functional.resample(audio.to(vad_device), sr, vad_sr).cpu()
    else:
        vad_audio = audio
    audio_np = vad_audio.squeeze(0).numpy() if vad_audio.dim() > 1 else vad_audio.numpy()
    classified = []
    for s, e, spk in segments:
        segment_audio = audio_np[int(s * vad_sr):min(int(e * vad_sr), len(audio_np))]
        if len(segment_audio) == 0:
            continue
        inference_started_at = time.perf_counter()
        pred = smart_vad.predict_endpoint(segment_audio, sample_rate=vad_sr)['prediction'] if smart_vad else 1
        logger.debug(
            f"perf model=smart_vad event=inference "
            f"seconds={time.perf_counter() - inference_started_at:.6f} "
            f"sample_rate={vad_sr} frames={len(segment_audio)}"
        )
        classified.append((s, e, spk, pred))

    merged: List[Tuple[float, float, int]] = []

    def save_eou_chunk(start: float, end: float, spk: int) -> None:
        if end - start > max_duration:
            start = end - max_duration
        if end - start >= min_duration:
            merged.append((start, end, spk))

    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    cur_spk: Optional[int] = None

    for start, end, spk, pred in classified:
        if cur_start is None or cur_end is None or cur_spk is None:
            cur_start, cur_end, cur_spk = start, end, spk
        else:
            gap = start - cur_end
            can_merge = spk == cur_spk and gap <= max_merge_gap

            if not can_merge:
                cur_start, cur_end, cur_spk = start, end, spk
            else:
                cur_end = end

        if pred == 1 and cur_start is not None and cur_end is not None and cur_spk is not None:
            save_eou_chunk(cur_start, cur_end, cur_spk)
            cur_start = cur_end = cur_spk = None

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
    """Write a chunk in its native sample rate via torchcodec.

    ``segment`` is ``(channels, samples)`` float32 in [-1, 1] — exactly what
    :meth:`torchcodec.decoders.AudioDecoder.get_samples_played_in_range`
    returns. :class:`torchcodec.encoders.AudioEncoder` picks reasonable
    container defaults per extension (FLAC/WAV bit-exact, lossy formats use
    sensible bitrate).
    """
    if segment.ndim == 1:
        segment = segment.unsqueeze(0)
    encoder = AudioEncoder(segment.contiguous(), sample_rate=int(sr))
    save_started_at = time.perf_counter()
    encoder.to_file(out_path)
    logger.debug(
        f"perf audio_save stage=preprocess path={out_path} "
        f"seconds={time.perf_counter() - save_started_at:.6f} "
        f"sample_rate={int(sr)} frames={int(segment.shape[-1])} format={fmt}"
    )


def _new_crest_audit() -> Dict[str, float]:
    return {
        "files_in": 0.0,
        "files_out": 0.0,
        "duration_in_s": 0.0,
        "duration_out_s": 0.0,
        "write_errors": 0.0,
        "postprocess_errors": 0.0,
    }


def _merge_crest_audit(target: Dict[str, float], source: Dict[str, float]) -> None:
    for key in target:
        target[key] += float(source.get(key, 0.0))


def _postprocess_chunk(
    segment: torch.Tensor,
    sample_rate: int,
    config: Dict[str, Any],
    path_label: str,
):
    result = postprocess_audio_tensor(
        segment,
        sample_rate,
        crest_threshold=float(
            config.get("crest_threshold", config.get("crest_treshold", 10.0))
        ),
        peak=float(config.get("peak", -1.0)),
        loudness=float(config.get("loudness", -23.0)),
        block_size=float(config.get("block_size", 0.400)),
    )
    if result.loudness_error:
        logger.error(
            f"Fused loudness normalization failed for {path_label}: "
            f"{result.loudness_error}; saving without normalization."
        )
    return result


def cut_audio(
    source_path: str,
    final_segments: List[Tuple[float, float, int]],
    raw_segments: List[Tuple[float, float, int]],
    output_folder: str,
    album_id: str,
    episode_id: str,
    fmt: str = 'flac',
    max_duration: float = 15.0,
    min_save_duration: float = DEFAULT_MIN_SAVE_DURATION_S,
    config: Optional[Dict[str, Any]] = None,
    crest_audit: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """Cut chunks from ``source_path`` lazily, in the source's native SR.

    The diarization tensor used upstream is downsampled to 16 kHz to keep VRAM
    flat. Here we re-open the original file with ``AudioDecoder`` *without* a
    target ``sample_rate`` and decode only the requested ``[start, end]``
    windows via ``get_samples_played_in_range``. The full waveform is never
    materialised in memory, so cutting a 2 h FLAC into ~hundreds of chunks
    costs only the size of one chunk at a time.
    """
    os.makedirs(output_folder, exist_ok=True)
    results: List[Dict] = []
    config = config or {}
    fuse_audio = fused_audio_preprocessing_enabled(config)
    crest_audit = crest_audit if crest_audit is not None else _new_crest_audit()

    try:
        decoder = AudioDecoder(source_path)
    except Exception as exc:
        logger.error(f"Could not open source for cutting {source_path}: {exc}")
        return results

    native_sr = int(decoder.metadata.sample_rate)
    # Attribute name shifted across torchcodec releases; fall back gracefully
    # so the cutter still clamps end-of-file overflow.
    dur_attr = (
        getattr(decoder.metadata, "duration_seconds", None)
        or getattr(decoder.metadata, "duration_seconds_from_header", None)
    )
    source_duration = float(dur_attr) if dur_attr else 0.0

    for start, end, spk in final_segments:
        dur = end - start
        if dur <= min_save_duration:
            continue

        clamped_end = min(end, source_duration) if source_duration > 0 else end
        if clamped_end <= start:
            continue

        sil_pct, max_sil, unique_spk = get_chunk_metrics(start, end, raw_segments)

        try:
            samples = decoder.get_samples_played_in_range(
                start_seconds=float(start), stop_seconds=float(clamped_end)
            )
        except Exception as exc:
            if fuse_audio:
                crest_audit["write_errors"] += 1
            logger.error(
                f"Failed to decode {start:.2f}-{clamped_end:.2f}s from {source_path}: {exc}"
            )
            continue

        segment = samples.data.to(dtype=torch.float32)
        if segment.ndim == 1:
            segment = segment.unsqueeze(0)
        if segment.shape[-1] == 0:
            if fuse_audio:
                crest_audit["write_errors"] += 1
            continue

        fname = f"{start:.2f}_{end:.2f}_{album_id}_{episode_id}.{fmt}"
        out_path = os.path.join(output_folder, fname)
        crest_factor = None
        loudness_normalized = None
        duration_s = float(segment.shape[-1]) / float(native_sr)
        if fuse_audio:
            crest_audit["files_in"] += 1
            crest_audit["duration_in_s"] += duration_s
            postprocessed = _postprocess_chunk(segment, native_sr, config, out_path)
            if postprocessed.loudness_error:
                crest_audit["postprocess_errors"] += 1
            crest_factor = round(postprocessed.crest_factor, 4)
            if not postprocessed.keep:
                logger.debug(
                    f"Rejected {out_path} before write "
                    f"(crest_factor={postprocessed.crest_factor:.2f})"
                )
                continue
            segment = postprocessed.samples
            loudness_normalized = postprocessed.loudness_normalized
            crest_audit["files_out"] += 1
            crest_audit["duration_out_s"] += duration_s

        try:
            _save_audio_chunk(out_path, segment, native_sr, fmt)
        except Exception as exc:
            if fuse_audio:
                crest_audit["write_errors"] += 1
            logger.error(f"Failed to write chunk {out_path}: {exc}")
            continue

        row = {
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
        }
        if fuse_audio:
            row["crest_factor"] = crest_factor
            row["loudness_normalized"] = True if loudness_normalized else ""
        results.append(row)
    return results


def process_audio_file(path_audio: str, audio: torch.Tensor, sr: int, config: Dict[str, Any]) -> Dict[str, Any]:
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
    max_merge_gap = float(config.get('max_merge_gap', DEFAULT_MAX_MERGE_GAP_S))
    fuse_audio = fused_audio_preprocessing_enabled(config)
    crest_audit = _new_crest_audit()

    p_audio = Path(path_audio)
    album_id, episode_id = p_audio.parent.name, p_audio.stem
    episode_folder = p_audio.parent / episode_id

    chunk_fmt = resolve_chunk_format(p_audio, chunk_format_cfg)

    total_audio_duration = audio.shape[-1] / sr

    try:
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        raw_segments = diarize_audio(audio, sr, chunk_duration)
        if not raw_segments:
            return {
                "segments": [],
                "source_duration_s": total_audio_duration,
                "crest_audit": crest_audit,
            }

        if total_audio_duration <= limit_dur:
            sil_pct, max_sil, unique_spk = get_chunk_metrics(0.0, total_audio_duration, raw_segments)
            main_spk = raw_segments[0][2] if raw_segments else -1

            row = {
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
                }

            if fuse_audio:
                decoder = AudioDecoder(str(p_audio))
                native_sr = int(decoder.metadata.sample_rate)
                native_audio = decoder.get_all_samples().data.to(dtype=torch.float32)
                del decoder
                if native_audio.ndim == 1:
                    native_audio = native_audio.unsqueeze(0)
                duration_s = float(native_audio.shape[-1]) / float(native_sr)
                crest_audit["files_in"] = 1
                crest_audit["duration_in_s"] = duration_s
                postprocessed = _postprocess_chunk(
                    native_audio, native_sr, config, str(p_audio)
                )
                if postprocessed.loudness_error:
                    crest_audit["postprocess_errors"] += 1
                row["crest_factor"] = round(postprocessed.crest_factor, 4)
                if not postprocessed.keep:
                    if p_audio.exists():
                        os.remove(p_audio)
                    return {
                        "segments": [],
                        "source_duration_s": total_audio_duration,
                        "crest_audit": crest_audit,
                    }
                crest_audit["files_out"] = 1
                crest_audit["duration_out_s"] = duration_s
                if postprocessed.loudness_normalized:
                    try:
                        _save_audio_chunk(
                            str(p_audio), postprocessed.samples, native_sr, chunk_fmt
                        )
                    except Exception:
                        crest_audit["write_errors"] += 1
                        raise
                    row["loudness_normalized"] = True
                else:
                    row["loudness_normalized"] = ""

            return {
                "segments": [row],
                "source_duration_s": total_audio_duration,
                "crest_audit": crest_audit,
            }

        clean_segments = build_single_speaker_timeline(
            raw_segments, max_duration=limit_dur
        )
        final_segments = apply_eos_classification(
            audio, sr, clean_segments, max_duration=limit_dur,
            min_duration=min_segment_dur, max_merge_gap=max_merge_gap,
        )

        if not final_segments:
            return {
                "segments": [],
                "source_duration_s": total_audio_duration,
                "crest_audit": crest_audit,
            }

        seg_results = cut_audio(
            str(p_audio),
            final_segments,
            raw_segments,
            str(episode_folder),
            album_id,
            episode_id,
            fmt=chunk_fmt,
            max_duration=limit_dur,
            min_save_duration=min_save_dur,
            config=config,
            crest_audit=crest_audit,
        )

        if seg_results:
            logger.success(
                f"Processed {len(seg_results)} chunks ({chunk_fmt}) from: {p_audio.name}"
            )

        completed_with_only_crest_rejections = (
            fuse_audio
            and crest_audit["files_in"] > 0
            and crest_audit["files_out"] == 0
            and crest_audit["write_errors"] == 0
        )
        source_processed = (
            bool(seg_results)
            and (not fuse_audio or crest_audit["write_errors"] == 0)
        ) or completed_with_only_crest_rejections
        if source_processed:
            if p_audio.exists():
                os.remove(p_audio)

        return {
            "segments": seg_results,
            "source_duration_s": total_audio_duration,
            "crest_audit": crest_audit,
        }

    except Exception as e:
        logger.error(f"Processing error {path_audio}: {e}")
        return {
            "segments": [],
            "source_duration_s": total_audio_duration,
            "crest_audit": crest_audit,
        }
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _measure_source_hours(paths: List[Path], max_workers: int = None) -> float:
    """Pre-flight scan to estimate hours available before processing (multiprocessing)."""
    if not paths:
        return 0.0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        total_seconds = sum(
            tqdm(
                executor.map(safe_audio_duration, paths),
                total=len(paths),
                desc="Scanning audio",
                unit="file"
            )
        )

    return total_seconds / 3600.0


def _run_diarization_shard(
    gpu_id: int,
    gpu_files: List[str],
    config: Dict[str, Any],
    num_loader_workers: int,
    podcasts_path: str,
) -> Dict[str, Any]:
    """Diarize a shard.

    Chunk metadata is streamed to ``preprocess_part_<gpu_id>.csv`` (one row per
    chunk, ``flush()`` after each write). The main process owns aggregation:
    a background ``PeriodicCsvMerger`` folds these partials into
    ``balalaika.csv`` every N rows, and a final ``absorb_partial_csvs`` runs
    at the end. No retries, no respawns — keep this hot path simple.
    """
    results: List[Dict[str, Any]] = []
    crest_audit = _new_crest_audit()
    partial_fields = (
        FUSED_PARTIAL_FIELDS
        if fused_audio_preprocessing_enabled(config)
        else PARTIAL_FIELDS
    )
    dataloader = create_diarization_dataloader(
        gpu_files,
        batch_size=int(config.get("diarization_batch_size", 1)),
        num_workers=int(config.get("diarization_loader_workers", num_loader_workers)),
        prefetch_factor=int(config.get("diarization_prefetch_factor", 2)),
    )
    batch_size = int(config.get("diarization_batch_size", 1))
    loader_workers = int(config.get("diarization_loader_workers", num_loader_workers))
    prefetch_factor = int(config.get("diarization_prefetch_factor", 2))
    prefetch_batches = loader_workers * prefetch_factor if loader_workers > 0 else 0
    logger.debug(
        f"perf dataloader_config stage=preprocess rank={gpu_id} "
        f"batch_size={batch_size} workers={loader_workers} "
        f"prefetch_factor={prefetch_factor} prefetch_batches={prefetch_batches} "
        f"items={len(gpu_files)}"
    )

    with PartialCsvWriter(
        podcasts_path, PARTIAL_PREFIX, gpu_id, fieldnames=partial_fields
    ) as writer:
        batch_wait_started_at = time.perf_counter()
        for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader), desc=f"GPU {gpu_id}", position=gpu_id)):
            batch_received_at = time.perf_counter()
            logger.debug(
                f"perf dataloader_wait stage=preprocess rank={gpu_id} "
                f"batch={batch_idx} seconds={batch_received_at - batch_wait_started_at:.6f} "
                f"items={len(batch)}"
            )
            for path_audio, audio, sr, error in batch:
                if error:
                    logger.error(f"Broken file {path_audio}: {error}")
                    continue
                try:
                    res = process_audio_file(str(path_audio), audio, sr, config)
                    _merge_crest_audit(
                        crest_audit, res.get("crest_audit", _new_crest_audit())
                    )
                    if res and res.get("segments"):
                        for seg in res["segments"]:
                            write_started_at = time.perf_counter()
                            writer.write({k: seg.get(k, "") for k in partial_fields})
                            logger.debug(
                                f"perf partial_write stage=preprocess rank={gpu_id} "
                                f"seconds={time.perf_counter() - write_started_at:.6f} "
                                f"path={seg.get('filepath', '')}"
                            )
                            results.append(seg)
                except Exception as e:
                    logger.error(f"Task error on GPU {gpu_id}: {e}")
            batch_wait_started_at = time.perf_counter()

    return {"segments": results, "crest_audit": crest_audit}


def process_gpu_batch(gpu_id: int, gpu_files: List[Path], config: Dict[str, Any], config_path: str, num_workers_per_gpu: int, podcasts_path: str) -> Dict[str, Any]:
    logger.info(f"GPU:{gpu_id} processing {len(gpu_files)} files...")
    with ProcessPoolExecutor(
        max_workers=1,
        initializer=init_models,
        initargs=(gpu_id, config, config_path),
    ) as executor:
        future = executor.submit(
            _run_diarization_shard,
            gpu_id,
            [str(p) for p in gpu_files],
            config,
            num_workers_per_gpu,
            podcasts_path,
        )
        results = future.result()

    return results


def main(args):
    setup_logging("preprocess", log_dir=args.log_dir)
    load_dotenv()
    if hf_key := os.environ.get('HF_TOKEN'):
        login(token=hf_key)

    config = load_config(args.config_path, 'preprocess')
    input_mode = str(config.get("input_mode", "raw")).strip().lower().replace("-", "_")
    if input_mode in {"existing_chunks", "prechunked", "pre_chunked", "chunks"}:
        logger.info("preprocess.input_mode=existing_chunks; backfilling chunk metadata without cutting audio.")
        from src.preprocess.preprocess_existing_chunks import main as existing_chunks_main

        existing_chunks_main(args, config=config, logging_configured=True)
        return
    if input_mode not in {"raw", "chunking"}:
        raise ValueError(
            "Unsupported preprocess.input_mode="
            f"{config.get('input_mode')!r}; expected 'raw' or 'existing_chunks'."
        )

    podcasts_path = Path(config.get('podcasts_path', '../../../podcasts'))
    num_workers_per_gpu = config.get('num_workers', 1)

    chunk_format_cfg = config.get('chunk_format', 'auto')
    logger.info(f"Chunk format policy: '{chunk_format_cfg}' (lossless input stays lossless).")
    fuse_audio = fused_audio_preprocessing_enabled(config)
    logger.info(f"Fused crest/loudness preprocessing: {fuse_audio}")

    num_gpus = torch.cuda.device_count()
    total_workers = max(1, num_gpus * num_workers_per_gpu)
    logger.info(f"GPUs: {num_gpus}, workers/GPU: {num_workers_per_gpu}, total workers: {total_workers}")

    raw_audio_paths = discover_audio_paths(podcasts_path, config_path=args.config_path)
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

    # hours_in = _measure_source_hours(paths_to_process, max_workers=4)
    hours_in = 0.0
    logger.info(f"Source audio total: {hours_in:.2f}h across {len(paths_to_process)} files")

    # Make sure balalaika.csv exists; absorb any leftover partials from a prior
    # interrupted run so resume picks up where things left off.
    ensure_main_csv(podcasts_path)
    chunk_value_columns = [c for c in FUSED_PARTIAL_FIELDS if c != "filepath"]
    _, absorbed = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=chunk_value_columns,
    )
    if absorbed:
        logger.info(
            f"Absorbed {absorbed} leftover rows from {PARTIAL_PREFIX}_part_*.csv "
            "before scheduling new work."
        )

    all_results: List[Dict[str, Any]] = []
    crest_audit = _new_crest_audit()
    files_per_gpu: List[List[Path]] = (
        [[] for _ in range(num_gpus)] if num_gpus > 0 else [paths_to_process]
    )

    if num_gpus > 0:
        for i, p in enumerate(paths_to_process):
            files_per_gpu[i % num_gpus].append(p)

    csv_settings = load_csv_settings(args.config_path)
    processed = 0
    errors = 0
    error_details: list[dict] = []

    try:
        with PeriodicCsvMerger(
            podcasts_path,
            prefix=PARTIAL_PREFIX,
            value_columns=chunk_value_columns,
            **csv_settings,
        ):
            with ThreadPoolExecutor(max_workers=max(1, num_gpus)) as thread_executor:
                gpu_futures = []
                for gpu_id in range(max(1, num_gpus)):
                    gpu_files = files_per_gpu[gpu_id]
                    if not gpu_files:
                        continue

                    gpu_futures.append(
                        thread_executor.submit(
                            process_gpu_batch,
                            gpu_id,
                            gpu_files,
                            config,
                            args.config_path,
                            num_workers_per_gpu,
                            str(podcasts_path),
                        )
                    )

                for future in as_completed(gpu_futures):
                    try:
                        batch_result = future.result()
                        all_results.extend(batch_result["segments"])
                        _merge_crest_audit(
                            crest_audit, batch_result["crest_audit"]
                        )
                        processed += 1
                    except Exception as e:
                        logger.error(f"Failed to aggregate results from a GPU batch: {e}")
                        errors += 1
                        error_details.append({"reason": str(e)})
    except KeyboardInterrupt:
        logger.warning("Preprocess stage interrupted; final partial absorb still runs.")

    # Final merge: fold everything the workers wrote into balalaika.csv and
    # delete the per-GPU partial CSVs.
    _, final_absorbed = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=chunk_value_columns,
    )
    if final_absorbed:
        logger.success(
            f"Processed {final_absorbed} chunk rows. "
            f"Metadata atomically written to {podcasts_path / 'balalaika.csv'}."
        )

    hours_out = sum(float(r.get('total_duration', 0.0)) for r in all_results) / 3600.0

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
            "max_merge_gap": config.get("max_merge_gap", DEFAULT_MAX_MERGE_GAP_S),
            "fuse_audio_preprocessing": fuse_audio,
        },
    )

    if fuse_audio and crest_audit["files_in"]:
        record_stage_summary(
            podcasts_path=podcasts_path,
            stage="crest_factor",
            files_in=int(crest_audit["files_in"]),
            files_out=int(crest_audit["files_out"]),
            hours_in=crest_audit["duration_in_s"] / 3600.0,
            hours_out=crest_audit["duration_out_s"] / 3600.0,
            params={
                "threshold": config.get(
                    "crest_threshold", config.get("crest_treshold", 10.0)
                ),
                "fused": True,
            },
        )

    if fuse_audio:
        errors += int(
            crest_audit["write_errors"] + crest_audit["postprocess_errors"]
        )

    write_stage_status(
        stage=1,
        stage_name="preprocess",
        log_dir=args.log_dir or "./logs",
        processed=processed,
        skipped=0,
        errors=errors,
        error_details=error_details,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    main(parser.parse_args())
