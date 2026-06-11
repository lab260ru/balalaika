import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchaudio
from loguru import logger as loguru_logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


DISTILLMOS_SAMPLE_RATE = 16_000
ANTISPOOF_SAMPLE_RATE = 16_000
ANTISPOOF_NUM_SAMPLES = 64_600


class DistillMOSDataset(Dataset):
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        path_str = self.file_paths[idx]
        started_at = time.perf_counter()
        try:
            waveform, sample_rate = torchaudio.load_with_torchcodec(path_str)
        except Exception:
            loguru_logger.debug(
                f"dataloader_audio_load dataset=distillmos path={path_str} "
                f"seconds={time.perf_counter() - started_at:.6f} error=load_failed"
            )
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
        loguru_logger.debug(
            f"dataloader_audio_load dataset=distillmos path={path_str} "
            f"seconds={time.perf_counter() - started_at:.6f} "
            f"sample_rate={DISTILLMOS_SAMPLE_RATE} frames={int(waveform.shape[-1])}"
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


class AntiSpoofingDataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        sample_rate: int = ANTISPOOF_SAMPLE_RATE,
        num_samples: int = ANTISPOOF_NUM_SAMPLES,
    ):
        self.file_paths = file_paths
        self.sample_rate = int(sample_rate)
        self.num_samples = int(num_samples)

    def __len__(self) -> int:
        return len(self.file_paths)

    def _pad_random(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim > 1:
            waveform = waveform.squeeze()
        wave_len = int(waveform.shape[0])
        if wave_len <= 0:
            raise ValueError("empty audio")
        if wave_len >= self.num_samples:
            start = random.randint(0, wave_len - self.num_samples)
            return waveform[start : start + self.num_samples]
        num_repeats = int(self.num_samples / wave_len) + 1
        return waveform.repeat(num_repeats)[: self.num_samples]

    def __getitem__(self, idx: int):
        path_str = self.file_paths[idx]
        started_at = time.perf_counter()
        try:
            waveform, source_sample_rate = torchaudio.load_with_torchcodec(path_str)
            waveform = waveform.to(dtype=torch.float32)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
            if int(source_sample_rate) != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform,
                    int(source_sample_rate),
                    self.sample_rate,
                )
            original_length = int(waveform.numel())
            waveform = torchaudio.functional.preemphasis(waveform.unsqueeze(0))
            waveform = self._pad_random(waveform.squeeze(0))
            loguru_logger.debug(
                f"dataloader_audio_load dataset=antispoofing path={path_str} "
                f"seconds={time.perf_counter() - started_at:.6f} "
                f"sample_rate={self.sample_rate} frames={original_length}"
            )
            return path_str, waveform.contiguous(), original_length, ""
        except Exception as exc:
            loguru_logger.debug(
                f"dataloader_audio_load dataset=antispoofing path={path_str} "
                f"seconds={time.perf_counter() - started_at:.6f} error={exc}"
            )
            return path_str, torch.empty(0, dtype=torch.float32), 0, str(exc)


def antispoofing_collate(batch):
    paths, waveforms, lengths, errors = zip(*batch)
    valid_indices = [idx for idx, error in enumerate(errors) if not error]
    valid_paths = [paths[idx] for idx in valid_indices]
    valid_lengths = torch.tensor([lengths[idx] for idx in valid_indices], dtype=torch.int64)
    valid_errors = [(paths[idx], errors[idx]) for idx in range(len(paths)) if errors[idx]]

    if not valid_indices:
        return [], torch.empty(0, 0, dtype=torch.float32), torch.empty(0, dtype=torch.int64), valid_errors

    batch_tensor = torch.stack([waveforms[idx] for idx in valid_indices]).contiguous()
    return valid_paths, batch_tensor, valid_lengths, valid_errors


def create_antispoofing_dataloader(
    file_paths: List[str],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    sample_rate: int = ANTISPOOF_SAMPLE_RATE,
    num_samples: int = ANTISPOOF_NUM_SAMPLES,
) -> DataLoader:
    dataset = AntiSpoofingDataset(
        file_paths,
        sample_rate=sample_rate,
        num_samples=num_samples,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,
        "collate_fn": antispoofing_collate,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)
