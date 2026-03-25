import argparse
import os
import torch
import torch.multiprocessing as mp
import torchaudio
import pandas as pd
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger
from tqdm import tqdm

from src.utils.utils import load_config, get_audio_paths


def calculate_crest_factor(audio: np.ndarray) -> float:
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return float('inf')
    return peak / rms


def run_worker(
    rank: int,
    world_size: int,
    all_file_paths: List[str],
    crest_threshold: float,
    output_dir: str,
):
    my_files = all_file_paths[rank::world_size]
    if not my_files:
        return

    results = []
    logger.info(f"Worker {rank}/{world_size} processing {len(my_files)} files")

    for path_str in tqdm(my_files, desc=f"Worker-{rank}", position=rank):
        try:
            audio_tensor, _ = torchaudio.load_with_torchcodec(path_str)
            if audio_tensor.shape[0] > 1:
                audio = audio_tensor.mean(dim=0).numpy()
            else:
                audio = audio_tensor.squeeze(0).numpy()
            cf = calculate_crest_factor(audio)
        except Exception as e:
            logger.error(f"Error processing {path_str}: {e}")
            continue
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if cf > crest_threshold:
            try:
                os.remove(path_str)
                logger.debug(f"Deleted {path_str} (crest_factor={cf:.2f})")
            except OSError as e:
                logger.error(f"Could not delete {path_str}: {e}")

        results.append({'filepath': str(Path(path_str).resolve()), 'crest_factor': round(cf, 4)})

    if results:
        part_path = Path(output_dir) / f'crest_part_{rank}.csv'
        pd.DataFrame(results).to_csv(part_path, index=False)
        logger.info(f"Worker {rank} saved {len(results)} crest_factor results.")


def update_csv(podcasts_path: Path, num_workers: int, crest_threshold: float):
    csv_path = podcasts_path / 'balalaika.csv'
    parts = [podcasts_path / f'crest_part_{i}.csv' for i in range(num_workers)]
    existing_parts = [p for p in parts if p.exists()]

    if not existing_parts:
        logger.warning("No crest_part_*.csv files found; skipping CSV update.")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            before = len(df)
            df = df[df['filepath'].apply(lambda p: Path(p).exists())]
            if before != len(df):
                df.to_csv(csv_path, index=False)
                logger.info(f"Removed {before - len(df)} missing rows from CSV.")
        return

    results_df = pd.concat([pd.read_csv(p) for p in existing_parts], ignore_index=True)
    for p in existing_parts:
        p.unlink()

    if not csv_path.exists():
        logger.warning(f"balalaika.csv not found at {csv_path}; skipping CSV update.")
        return

    df = pd.read_csv(csv_path)
    df['filepath'] = df['filepath'].apply(lambda p: str(Path(p).resolve()))
    results_df['filepath'] = results_df['filepath'].apply(lambda p: str(Path(p).resolve()))

    if 'crest_factor' in df.columns:
        df = df.drop(columns=['crest_factor'])
    df = df.merge(results_df[['filepath', 'crest_factor']], on='filepath', how='left')

    before = len(df)
    df = df[df['filepath'].apply(lambda p: Path(p).exists())]
    removed = before - len(df)
    deleted_count = (results_df['crest_factor'] > crest_threshold).sum()
    logger.info(
        f"Crest filter: {deleted_count} files deleted (threshold={crest_threshold}), "
        f"{removed} rows removed from CSV."
    )

    df.to_csv(csv_path, index=False)
    logger.success(f"CSV updated: {len(df)} rows remain.")


def main(args):
    config = load_config(args.config_path, 'preprocess')

    podcasts_path = config.get('podcasts_path')
    if not podcasts_path:
        podcasts_path = '../../../podcasts'
        logger.warning("Using default podcasts_path")
    podcasts_path = Path(podcasts_path)

    crest_threshold = config.get('crest_treshold', 10.0)
    num_workers = config.get('num_workers', 4)

    logger.info(
        f"Running crest factor removal: path={podcasts_path}, "
        f"threshold={crest_threshold}, workers={num_workers}"
    )

    audio_paths = [str(p) for p in get_audio_paths(str(podcasts_path))]
    if not audio_paths:
        logger.info("No audio files found for processing.")
        return

    logger.info(f"Found {len(audio_paths)} audio files to check")

    if num_workers > 1:
        mp.spawn(
            run_worker,
            args=(num_workers, audio_paths, crest_threshold, str(podcasts_path)),
            nprocs=num_workers,
            join=True
        )
    else:
        run_worker(0, 1, audio_paths, crest_threshold, str(podcasts_path))

    update_csv(podcasts_path, num_workers, crest_threshold)
    logger.info("Crest factor check completed.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(
        description="Remove audio files that exceed crest factor threshold (peak/rms > threshold)."
    )
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    main(args)
