"""Crest factor (peak/RMS) filter with full audit accounting.

Beyond the original behaviour (compute crest factor, append to ``balalaika.csv``
and delete files above the threshold) the rewritten worker keeps the deleted
file's duration so the rank-0 process can credit those hours to this stage in
``filter_summary.csv``. That gives the final report an accurate "hours
removed by crest factor" figure even after the audio is gone from disk.

A per-stage rotating log file is initialised by :func:`setup_logging` so
operators can replay long batch runs offline.
"""

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torchaudio
from loguru import logger
from tqdm import tqdm

from src.utils.audit import record_stage_summary, safe_audio_duration
from src.utils.logging_setup import setup_logging
from src.utils.utils import get_audio_paths, load_config, load_audio


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
            audio_tensor, sr = load_audio(path_str)
            if audio_tensor.shape[0] > 1:
                audio = audio_tensor.mean(dim=0).numpy()
            else:
                audio = audio_tensor.squeeze(0).numpy()
            cf = calculate_crest_factor(audio)
            duration_s = float(audio.shape[-1]) / float(sr) if sr else 0.0
        except Exception as e:
            logger.error(f"Error processing {path_str}: {e}")
            continue
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        deleted = False
        if cf > crest_threshold:
            try:
                os.remove(path_str)
                deleted = True
                logger.debug(f"Deleted {path_str} (crest_factor={cf:.2f})")
            except OSError as e:
                logger.error(f"Could not delete {path_str}: {e}")

        results.append({
            'filepath': str(Path(path_str).resolve()),
            'crest_factor': round(cf, 4),
            'duration_s': round(duration_s, 4),
            'deleted': deleted,
        })

    if results:
        part_path = Path(output_dir) / f'crest_part_{rank}.csv'
        pd.DataFrame(results).to_csv(part_path, index=False)
        logger.info(f"Worker {rank} saved {len(results)} crest_factor results.")


def update_csv(podcasts_path: Path, num_workers: int, crest_threshold: float) -> dict:
    """Merge worker partials into ``balalaika.csv`` and return audit totals."""
    csv_path = podcasts_path / 'balalaika.csv'
    parts = [podcasts_path / f'crest_part_{i}.csv' for i in range(num_workers)]
    existing_parts = [p for p in parts if p.exists()]

    audit = {
        "files_in": 0,
        "files_out": 0,
        "hours_in": 0.0,
        "hours_out": 0.0,
        "files_deleted": 0,
    }

    if not existing_parts:
        logger.warning("No crest_part_*.csv files found; skipping CSV update.")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            before = len(df)
            df = df[df['filepath'].apply(lambda p: Path(p).exists())]
            if before != len(df):
                df.to_csv(csv_path, index=False)
                logger.info(f"Removed {before - len(df)} missing rows from CSV.")
        return audit

    results_df = pd.concat([pd.read_csv(p) for p in existing_parts], ignore_index=True)
    for p in existing_parts:
        p.unlink()

    audit["files_in"] = int(len(results_df))
    audit["hours_in"] = float(results_df.get('duration_s', pd.Series([0.0])).sum() / 3600.0)
    audit["files_deleted"] = int(results_df['deleted'].sum()) if 'deleted' in results_df.columns else 0
    survived = results_df[~results_df.get('deleted', pd.Series([False] * len(results_df)))]
    audit["files_out"] = int(len(survived))
    audit["hours_out"] = float(survived.get('duration_s', pd.Series([0.0])).sum() / 3600.0)

    if not csv_path.exists():
        logger.warning(f"balalaika.csv not found at {csv_path}; skipping CSV update.")
        return audit

    df = pd.read_csv(csv_path)
    df['filepath'] = df['filepath'].apply(lambda p: str(Path(p).resolve()))
    results_df['filepath'] = results_df['filepath'].apply(lambda p: str(Path(p).resolve()))

    if 'crest_factor' in df.columns:
        df = df.drop(columns=['crest_factor'])
    df = df.merge(results_df[['filepath', 'crest_factor']], on='filepath', how='left')

    before = len(df)
    df = df[df['filepath'].apply(lambda p: Path(p).exists())]
    removed = before - len(df)
    logger.info(
        f"Crest filter: {audit['files_deleted']} files deleted (threshold={crest_threshold}), "
        f"{removed} rows removed from CSV."
    )

    df.to_csv(csv_path, index=False)
    logger.success(f"CSV updated: {len(df)} rows remain.")
    return audit


def main(args):
    setup_logging("crest_factor", log_dir=args.log_dir)

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
            join=True,
        )
    else:
        run_worker(0, 1, audio_paths, crest_threshold, str(podcasts_path))

    audit = update_csv(podcasts_path, num_workers, crest_threshold)

    if audit["files_in"] == 0 and audio_paths:
        # Fallback: workers wrote nothing (e.g. all read failures).
        # Probe a few files so the report still has *some* hours_in number.
        fallback_hours = sum(safe_audio_duration(p) for p in audio_paths) / 3600.0
        audit["files_in"] = len(audio_paths)
        audit["hours_in"] = fallback_hours
        audit["hours_out"] = fallback_hours

    record_stage_summary(
        podcasts_path=podcasts_path,
        stage="crest_factor",
        files_in=audit["files_in"],
        files_out=audit["files_out"],
        hours_in=audit["hours_in"],
        hours_out=audit["hours_out"],
        params={"threshold": crest_threshold, "deleted": audit["files_deleted"]},
    )

    logger.info("Crest factor check completed.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(
        description="Remove audio files that exceed crest factor threshold (peak/rms > threshold)."
    )
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    args = parser.parse_args()

    main(args)
