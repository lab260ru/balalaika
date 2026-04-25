"""Music detection filter built on a fine-tuned WavLM head.

In addition to scoring each chunk and deleting clips above the threshold, the
worker now records per-file durations into the partial CSV. That lets the
rank-0 process emit a stage row in ``filter_summary.csv`` capturing the actual
hours of audio dropped, even though the files themselves are gone. A per-stage
log file is initialised at startup for offline debugging.
"""

import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger
from musicdetection.audio_cache import create_audio_length_cache
from musicdetection.audio_sampler import LengthBasedBatchSampler
from musicdetection.core.model import WavLMForMusicDetection
from musicdetection.dataset import AudioCollate, MusicDetectionDataset
from safetensors import safe_open
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

from src.utils.audit import record_stage_summary, safe_audio_duration
from src.utils.logging_setup import setup_logging
from src.utils.utils import get_audio_paths, load_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)


def create_loader(paths: List[str], model_name: str, batch_size: int, num_workers: int, cache_file: Path):
    audio_lengths = create_audio_length_cache(file_paths=paths, cache_file=str(cache_file))
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    dataset = MusicDetectionDataset(file_paths=paths, target_sample_rate=processor.sampling_rate)
    sampler = LengthBasedBatchSampler(paths, audio_lengths, batch_size=batch_size, shuffle=False)
    return (
        DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=AudioCollate(processor),
            num_workers=num_workers,
            pin_memory=True,
        ),
        audio_lengths,
    )


def load_model(model_path: str, base_model: str, device: torch.device):
    model = WavLMForMusicDetection(base_model_name=base_model)
    with safe_open(model_path, framework="pt", device="cpu") as f:
        model.load_state_dict({k: f.get_tensor(k) for k in f.keys()})
    model = model.to(device).eval()
    model.device = device
    return model


def run_worker(rank: int, world_size: int, all_paths: List[str], config: dict):
    my_paths = all_paths[rank::world_size]
    if not my_paths:
        return

    device = torch.device(f"cuda:{rank}")
    cfg = config.get('music_detect', {})
    podcasts_path = Path(config.get('podcasts_path', '.'))

    threshold = cfg.get('threshold', 0.5)
    cache_dir = Path(cfg.get('cache_path', './cache')) / f'nisqa_temp_worker_{rank}'
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{device}] Processing {len(my_paths)} files...")

    try:
        dataloader, audio_lengths = create_loader(
            my_paths,
            cfg.get('base_model', 'microsoft/wavlm-base-plus'),
            cfg.get('bs', 32),
            cfg.get('num_workers', 4),
            cache_dir / 'audio_lengths.json',
        )

        model = load_model(
            cfg.get('music_detect_model'),
            cfg.get('base_model', 'microsoft/wavlm-base-plus'),
            device,
        )

        probs, paths = model.predict_proba(dataloader)

        results = []
        deleted_count = 0
        for path, prob in zip(paths, probs.detach().flatten()):
            prob_val = round(float(prob), 6)
            duration_s = float(audio_lengths.get(str(path), 0.0))
            if duration_s <= 0:
                duration_s = safe_audio_duration(path)
            entry = {
                'filepath': str(Path(path).resolve()),
                'music_prob': prob_val,
                'duration_s': round(duration_s, 4),
                'deleted': False,
            }
            if prob_val > threshold:
                try:
                    os.remove(path)
                    deleted_count += 1
                    entry['deleted'] = True
                except OSError as e:
                    logger.warning(f"Could not delete {path}: {e}")
            results.append(entry)

        if results:
            part_path = podcasts_path / f'music_part_{rank}.csv'
            pd.DataFrame(results).to_csv(part_path, index=False)

        logger.success(f"[{device}] Done. Deleted {deleted_count}/{len(my_paths)} files.")

    except Exception as e:
        logger.exception(f"Worker {rank} error: {e}")


def update_csv(podcasts_path: Path, n_gpus: int) -> dict:
    csv_path = podcasts_path / 'balalaika.csv'
    parts = [podcasts_path / f'music_part_{i}.csv' for i in range(n_gpus)]
    existing_parts = [p for p in parts if p.exists()]

    audit = {
        "files_in": 0,
        "files_out": 0,
        "hours_in": 0.0,
        "hours_out": 0.0,
        "files_deleted": 0,
    }

    if not existing_parts:
        logger.warning("No music_part_*.csv files found; skipping CSV update.")
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
    if 'duration_s' in results_df.columns:
        audit["hours_in"] = float(results_df['duration_s'].sum() / 3600.0)
    audit["files_deleted"] = int(results_df['deleted'].sum()) if 'deleted' in results_df.columns else 0
    survived = results_df[~results_df.get('deleted', pd.Series([False] * len(results_df)))]
    audit["files_out"] = int(len(survived))
    if 'duration_s' in survived.columns:
        audit["hours_out"] = float(survived['duration_s'].sum() / 3600.0)

    if not csv_path.exists():
        logger.warning(f"balalaika.csv not found at {csv_path}; skipping CSV update.")
        return audit

    df = pd.read_csv(csv_path)
    df['filepath'] = df['filepath'].apply(lambda p: str(Path(p).resolve()))
    results_df['filepath'] = results_df['filepath'].apply(lambda p: str(Path(p).resolve()))

    if 'music_prob' in df.columns:
        df = df.drop(columns=['music_prob'])
    df = df.merge(results_df[['filepath', 'music_prob']], on='filepath', how='left')

    before = len(df)
    df = df[df['filepath'].apply(lambda p: Path(p).exists())]
    removed = before - len(df)
    logger.info(f"Music detection: removed {removed} rows from CSV (files deleted).")

    df.to_csv(csv_path, index=False)
    logger.success(f"CSV updated: {len(df)} rows remain.")
    return audit


def main(args):
    setup_logging("music_detect", log_dir=args.log_dir)
    mp.set_start_method('spawn', force=True)
    config = load_config(args.config_path, 'separation')
    podcasts_path = config.get('podcasts_path')

    if not podcasts_path:
        logger.error("No podcasts_path in config")
        return

    all_paths = list(get_audio_paths(podcasts_path))
    n_gpus = torch.cuda.device_count()

    if not all_paths:
        logger.warning("No audio files found.")
        return

    if n_gpus == 0:
        logger.error("No GPU found.")
        return

    mp.spawn(run_worker, args=(n_gpus, all_paths, config), nprocs=n_gpus, join=True)
    audit = update_csv(Path(podcasts_path), n_gpus)

    cfg = config.get('music_detect', {})
    record_stage_summary(
        podcasts_path=Path(podcasts_path),
        stage="music_detect",
        files_in=audit["files_in"],
        files_out=audit["files_out"],
        hours_in=audit["hours_in"],
        hours_out=audit["hours_out"],
        params={"threshold": cfg.get('threshold', 0.5), "deleted": audit["files_deleted"]},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    main(parser.parse_args())
