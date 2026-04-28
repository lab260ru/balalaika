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
from typing import List, Set

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
import torch
import torch.multiprocessing as mp
import torchaudio
from loguru import logger
from tqdm import tqdm

from src.utils.logging_setup import setup_logging
from src.utils.utils import get_audio_paths, load_config, load_audio

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


LOSSLESS_EXTS = {".flac", ".wav"}
NORMALIZED_COLUMN = "loudness_normalized"
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
) -> bool:
    """Normalize loudness for a single file in-place."""
    try:
        audio, sample_rate = load_audio(audio_path)
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
        return True
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        return False
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _resolve_path(path: str | Path) -> str:
    return str(Path(path).resolve())


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_normalized_paths(csv_path: Path) -> Set[str]:
    """Read already-normalized files from balalaika.csv."""
    if not csv_path.exists():
        return set()

    df = pd.read_csv(csv_path)
    if 'filepath' not in df.columns or NORMALIZED_COLUMN not in df.columns:
        return set()

    df['filepath'] = df['filepath'].apply(_resolve_path)
    normalized = df[df[NORMALIZED_COLUMN].apply(_truthy)]
    return set(normalized['filepath'].tolist())


def cleanup_partial_csvs(podcasts_path: Path) -> None:
    for part_path in podcasts_path.glob("loudness_part_*.csv"):
        part_path.unlink()


def update_normalization_csv(podcasts_path: Path, num_workers: int) -> None:
    """Merge worker partials and mark successful files as normalized."""
    csv_path = podcasts_path / 'balalaika.csv'
    parts = [podcasts_path / f'loudness_part_{rank}.csv' for rank in range(num_workers)]
    existing_parts = [p for p in parts if p.exists()]

    if not existing_parts:
        logger.info("No newly normalized files to mark in CSV.")
        return

    results_df = pd.concat([pd.read_csv(p) for p in existing_parts], ignore_index=True)
    for part_path in existing_parts:
        part_path.unlink()

    results_df['filepath'] = results_df['filepath'].apply(_resolve_path)
    results_df = results_df.drop_duplicates(subset=['filepath'], keep='last')
    results_df[NORMALIZED_COLUMN] = True

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if 'filepath' not in df.columns:
            logger.warning(f"{csv_path} has no filepath column; creating normalization-only CSV.")
            df = pd.DataFrame(columns=['filepath'])
    else:
        logger.warning(f"balalaika.csv not found at {csv_path}; creating normalization-only CSV.")
        df = pd.DataFrame(columns=['filepath'])

    df['filepath'] = df['filepath'].apply(_resolve_path) if not df.empty else df.get('filepath', pd.Series(dtype=str))
    if NORMALIZED_COLUMN not in df.columns:
        df[NORMALIZED_COLUMN] = False

    df = df.merge(
        results_df[['filepath', NORMALIZED_COLUMN]],
        on='filepath',
        how='outer',
        suffixes=('', '_new'),
    )
    df[NORMALIZED_COLUMN] = df[f'{NORMALIZED_COLUMN}_new'].fillna(df[NORMALIZED_COLUMN]).apply(_truthy)
    df = df.drop(columns=[f'{NORMALIZED_COLUMN}_new'])
    df.to_csv(csv_path, index=False)
    logger.success(f"Marked {len(results_df)} files as loudness-normalized in {csv_path}.")


def run_worker(
    rank: int,
    world_size: int,
    all_file_paths: List[Path],
    peak: float,
    loudness: float,
    block_size: float,
    output_dir: str,
):
    """Worker that processes a sharded subset of files."""
    if not all_file_paths:
        return

    my_files = all_file_paths[rank::world_size]
    if not my_files:
        return

    logger.info(f"Worker {rank}/{world_size} processing {len(my_files)} files")

    normalized = []
    for file_path in tqdm(my_files, desc=f"Worker-{rank}", position=rank):
        if process_audio_file(str(file_path), peak, loudness, block_size):
            normalized.append({'filepath': _resolve_path(file_path), NORMALIZED_COLUMN: True})

    if normalized:
        part_path = Path(output_dir) / f'loudness_part_{rank}.csv'
        pd.DataFrame(normalized).to_csv(part_path, index=False)
        logger.info(f"Worker {rank} marked {len(normalized)} normalized files.")


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
    num_workers = 16

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

    podcasts_path = Path(podcasts_path)
    csv_path = podcasts_path / 'balalaika.csv'
    cleanup_partial_csvs(podcasts_path)

    audio_paths = get_audio_paths(str(podcasts_path))
    if not audio_paths:
        logger.info("No audio files found for processing.")
        return

    normalized_paths = load_normalized_paths(csv_path)
    paths_to_process = [p for p in audio_paths if _resolve_path(p) not in normalized_paths]
    skipped = len(audio_paths) - len(paths_to_process)

    logger.info(
        f"Found {len(audio_paths)} audio files; "
        f"skipping {skipped} already normalized; processing {len(paths_to_process)}."
    )

    if not paths_to_process:
        logger.info("All audio files are already loudness-normalized.")
        return

    if num_processes > 1:
        mp.spawn(
            run_worker,
            args=(num_processes, paths_to_process, peak, loudness, block_size, str(podcasts_path)),
            nprocs=num_processes,
            join=True,
        )
    else:
        normalized = []
        for file_path in tqdm(paths_to_process, desc="Normalizing loudness"):
            if process_audio_file(str(file_path), peak, loudness, block_size):
                normalized.append({'filepath': _resolve_path(file_path), NORMALIZED_COLUMN: True})
        if normalized:
            pd.DataFrame(normalized).to_csv(podcasts_path / 'loudness_part_0.csv', index=False)

    update_normalization_csv(podcasts_path, num_processes)
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
