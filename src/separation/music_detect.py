import argparse
import os
import torch
import torch.multiprocessing as mp
from pathlib import Path
from typing import List
from loguru import logger
from tqdm import tqdm
from safetensors import safe_open
from transformers import AutoFeatureExtractor
from torch.utils.data import DataLoader

from musicdetection.audio_cache import create_audio_length_cache
from musicdetection.dataset import MusicDetectionDataset, AudioCollate
from musicdetection.core.model import WavLMForMusicDetection
from musicdetection.audio_sampler import LengthBasedBatchSampler
from src.utils.utils import get_audio_paths, load_config

# Optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)

def create_loader(paths: List[str], model_name: str, batch_size: int, num_workers: int, cache_file: Path):
    """Creates a DataLoader with caching of audio lengths."""

    audio_lengths = create_audio_length_cache(file_paths=paths, cache_file=str(cache_file))
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    
    dataset = MusicDetectionDataset(file_paths=paths, target_sample_rate=processor.sampling_rate)
    sampler = LengthBasedBatchSampler(paths, audio_lengths, batch_size=batch_size, shuffle=False)
    
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=AudioCollate(processor),
        num_workers=num_workers,
        pin_memory=True
    )

def load_model(model_path: str, base_model: str, device: torch.device):
    """Loads the model and weights."""
    model = WavLMForMusicDetection(base_model_name=base_model)
    with safe_open(model_path, framework="pt", device="cpu") as f:
        model.load_state_dict({k: f.get_tensor(k) for k in f.keys()})
    model = model.to(device).eval()
    model.device = device
    return model

def run_worker(rank: int, world_size: int, all_paths: List[str], config: dict):
    my_paths = all_paths[rank::world_size]
    if not my_paths: return

    device = torch.device(f"cuda:{rank}")
    cfg = config.get('music_detect', {})
    
    # Params
    threshold = cfg.get('threshold', 0.5)
    cache_dir = Path(cfg.get('cache_path', './cache')) / f'nisqa_temp_worker_{rank}'
                                            # cache/nisqa_temp_worker_0/nisqa_temp/audio_lengths_cache.json
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{device}] Processing {len(my_paths)} files...")

    try:
        # 1. Setup Data & Model
        dataloader = create_loader(
            my_paths, 
            cfg.get('base_model', 'microsoft/wavlm-base-plus'),
            cfg.get('bs', 32),
            cfg.get('num_workers', 4),
            cache_dir / 'audio_lengths.json'
        )
        
        model = load_model(
            cfg.get('music_detect_model'), 
            cfg.get('base_model', 'microsoft/wavlm-base-plus'), 
            device
        )

        # 2. Inference — predict_proba expects a DataLoader, not a single batch
        probs, paths = model.predict_proba(dataloader)

        # 3. Filter and delete files detected as music
        deleted_count = 0
        for path, prob in zip(paths, probs.detach().flatten()):
            # print(prob)
            if prob > threshold:
                try:
                    os.remove(path)
                    deleted_count += 1
                except OSError:
                    pass

        logger.success(f"[{device}] Done. Deleted {deleted_count} files.")

    except Exception as e:
        logger.exception(f"Worker {rank} error: {e}")

def main(args):
    mp.set_start_method('spawn', force=True)
    config = load_config(args.config_path, 'separation')
    podcasts_path = config.get('podcasts_path')
    all_paths = list(get_audio_paths(podcasts_path))
    n_gpus = torch.cuda.device_count()
    
    if not podcasts_path:
        logger.error("No podcasts_path in config")
        return

    if not all_paths:
        logger.warning("No audio files found.")
        return

    if n_gpus == 0:
        logger.error("No GPU found.")
        return

    mp.spawn(run_worker, args=(n_gpus, all_paths, config), nprocs=n_gpus, join=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    main(parser.parse_args())