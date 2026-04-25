"""ITU-R BS.1770-4 loudness normalization that **preserves the source format**.

The original implementation re-encoded everything through ``torchaudio.save``,
which silently degraded MP3 files and made the FLAC → ``.mp3`` mismatch
operators reported. This rewrite:

* Reads samples through ``torchaudio.load_with_torchcodec`` (or torchaudio as a
  fallback) so we don't drop torchcodec's sample-perfect decoding.
* Uses ``soundfile`` to write **lossless containers (FLAC/WAV)** at native
  precision (PCM_16/24 floats become FLAC PCM_24 / WAV FLOAT) which is
  bit-equivalent within the encoder's quantization budget.
* Falls back to ``torchaudio.save`` for lossy containers (MP3/OGG/OPUS) so we
  don't break inputs that aren't covered by libsndfile.
* Sets up a per-stage log file via :func:`setup_logging`.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
import torch.multiprocessing as mp
import torchaudio
from loguru import logger
from tqdm import tqdm

from src.utils.logging_setup import setup_logging
from src.utils.utils import get_audio_paths, load_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


LOSSLESS_EXTS = {".flac", ".wav"}
SOUNDFILE_FORMAT = {
    ".flac": ("FLAC", "PCM_24"),
    ".wav": ("WAV", "FLOAT"),
}


def normalize_audio_loudness(
    audio: np.ndarray,
    rate: int,
    peak: float = -1.0,
    loudness: float = -23.0,
    block_size: float = 0.400,
) -> np.ndarray:
    """ITU-R BS.1770-4 loudness normalization (peak then integrated LUFS)."""
    audio = pyln.normalize.peak(audio, peak)
    meter = pyln.Meter(rate, block_size=block_size)
    measured = meter.integrated_loudness(audio)
    return pyln.normalize.loudness(audio, measured, loudness)


def _load_audio(audio_path: str):
    """Decode an audio file as ``(channels, samples)`` plus its sample rate.

    Prefers ``torchaudio.load_with_torchcodec`` (the original code path) and
    falls back to plain ``torchaudio.load`` when torchcodec isn't bundled.
    """
    if hasattr(torchaudio, "load_with_torchcodec"):
        try:
            return torchaudio.load_with_torchcodec(audio_path)
        except Exception as exc:
            logger.debug(f"torchcodec failed for {audio_path}: {exc}; falling back to torchaudio.load")
    return torchaudio.load(audio_path)


def _write_audio(audio_path: str, samples: np.ndarray, sample_rate: int) -> None:
    """Write back samples preserving the original format losslessly when we can.

    ``samples`` is shaped ``(channels, frames)`` (torchaudio convention).
    """
    suffix = Path(audio_path).suffix.lower()
    if samples.ndim == 1:
        write_arr = samples
    else:
        write_arr = samples.T  # soundfile expects (frames, channels)

    if suffix in SOUNDFILE_FORMAT:
        fmt, subtype = SOUNDFILE_FORMAT[suffix]
        sf.write(
            audio_path,
            write_arr.astype(np.float32, copy=False),
            sample_rate,
            format=fmt,
            subtype=subtype,
        )
        return

    tensor = torch.from_numpy(samples if samples.ndim == 2 else samples[np.newaxis, :])
    torchaudio.save(audio_path, tensor, sample_rate)


def process_audio_file(
    audio_path: str,
    peak: float,
    loudness: float,
    block_size: float,
):
    """Normalize loudness for a single file in-place."""
    try:
        audio, sample_rate = _load_audio(audio_path)
        audio_np = audio.numpy()

        if audio_np.shape[0] == 1:
            mono = audio_np.squeeze(0)
            normalized = normalize_audio_loudness(
                mono, sample_rate, peak=peak, loudness=loudness, block_size=block_size
            )
            normalized_2d = normalized[np.newaxis, :]
        else:
            multi = audio_np.T  # (frames, channels)
            normalized = normalize_audio_loudness(
                multi, sample_rate, peak=peak, loudness=loudness, block_size=block_size
            )
            normalized_2d = normalized.T

        _write_audio(audio_path, normalized_2d, sample_rate)

        logger.debug(f"Normalized: {audio_path}")
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_worker(
    rank: int,
    world_size: int,
    all_file_paths: List[Path],
    peak: float,
    loudness: float,
    block_size: float,
):
    """Worker that processes a sharded subset of files."""
    if not all_file_paths:
        return

    my_files = all_file_paths[rank::world_size]
    if not my_files:
        return

    logger.info(f"Worker {rank}/{world_size} processing {len(my_files)} files")

    for file_path in tqdm(my_files, desc=f"Worker-{rank}", position=rank):
        process_audio_file(str(file_path), peak, loudness, block_size)


def main(args):
    setup_logging("preprocess_audio", log_dir=args.log_dir)

    config = load_config(args.config_path, 'preprocess')

    podcasts_path = config.get('podcasts_path')
    if not podcasts_path:
        podcasts_path = config.get('podcasts_path', '../../../podcasts')
        logger.warning("Using default podcasts_path")

    peak = config.get('peak', -1.0)
    loudness = config.get('loudness', -23.0)
    block_size = config.get('block_size', 0.400)
    num_workers = config.get('num_workers', 4)

    num_processes = num_workers

    logger.info(
        f"""
        Running loudness normalization (format-preserving):
        Podcasts path: {podcasts_path}
        Peak normalization: {peak} dB
        Target loudness: {loudness} LUFS
        Block size: {block_size} seconds
        Number of processes: {num_processes}
        """
    )

    audio_paths = get_audio_paths(podcasts_path)
    if not audio_paths:
        logger.info("No audio files found for processing.")
        return

    logger.info(f"Found {len(audio_paths)} audio files to process")

    if num_processes > 1:
        mp.spawn(
            run_worker,
            args=(num_processes, audio_paths, peak, loudness, block_size),
            nprocs=num_processes,
            join=True,
        )
    else:
        for file_path in tqdm(audio_paths, desc="Normalizing loudness"):
            process_audio_file(str(file_path), peak, loudness, block_size)

    logger.info("All files have been processed and normalized.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(
        description="Normalize audio loudness (ITU-R BS.1770-4) for all audio files in the dataset."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    args = parser.parse_args()

    main(args)
