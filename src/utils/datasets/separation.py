import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


DISTILLMOS_SAMPLE_RATE = 16_000


class DistillMOSDataset(Dataset):
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        path_str = self.file_paths[idx]
        try:
            waveform, sample_rate = torchaudio.load_with_torchcodec(path_str)
        except Exception:
            logger.warning("Failed to load %s, returning silence", path_str)
            return path_str, torch.zeros(DISTILLMOS_SAMPLE_RATE // 100)
        if waveform.shape[0] > 1:
            waveform = waveform[:1]
        if sample_rate != DISTILLMOS_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                DISTILLMOS_SAMPLE_RATE,
            )
        return path_str, waveform.squeeze(0).contiguous()


def distillmos_collate(batch: List[Tuple[str, torch.Tensor]]) -> Tuple[List[str], torch.Tensor]:
    paths, waves = zip(*batch)
    padded = pad_sequence(waves, batch_first=True)
    return list(paths), padded


def estimate_audio_lengths(file_paths: List[str]) -> Dict[str, float]:
    lengths = {}
    for path_str in tqdm(file_paths, desc="Read audio before start"):
        try:
            info = torchaudio.info(path_str)
            if info.sample_rate and info.num_frames:
                lengths[path_str] = float(info.num_frames) / float(info.sample_rate)
            else:
                lengths[path_str] = 0.0
        except Exception:
            lengths[path_str] = 0.0
    return lengths


def sort_by_length(file_paths: List[str], cache_dir: str = "") -> List[str]:
    cache_file = Path(cache_dir) / "distillmos_sorted_files.json" if cache_dir else None

    if cache_file and cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        current_set = set(file_paths)
        cached_set = set(cached)

        if current_set.issubset(cached_set):
            filtered = [p for p in cached if p in current_set]
            if len(filtered) == len(current_set):
                logger.info("Cache hit: using cached order for %d files", len(filtered))
                return filtered

        if cached_set.issubset(current_set):
            new_files = sorted(current_set - cached_set)
            logger.info("Cache partial hit: %d cached + %d new files", len(cached), len(new_files))
            lengths = estimate_audio_lengths(new_files)
            new_sorted = sorted(new_files, key=lambda p: lengths.get(p, 0.0))
            result = cached + new_sorted
            with open(cache_file, "w") as f:
                json.dump(result, f)
            return result

        logger.info("Cache stale (%d cached vs %d current), rebuilding", len(cached), len(file_paths))

    logger.info("Scanning %d audio files for length sorting...", len(file_paths))
    lengths = estimate_audio_lengths(file_paths)
    sorted_paths = sorted(file_paths, key=lambda p: lengths.get(p, 0.0))

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(sorted_paths, f)
        logger.info("Saved sorted file list (%d files) to %s", len(sorted_paths), cache_file)

    return sorted_paths


def create_distillmos_dataloader(
    file_paths: List[str],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    cache_dir: str = "",
) -> DataLoader:
    dataset = DistillMOSDataset(sort_by_length(file_paths, cache_dir=cache_dir))
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,
        "collate_fn": distillmos_collate,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)
