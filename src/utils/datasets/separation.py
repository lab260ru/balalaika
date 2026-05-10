from typing import Dict, List, Tuple

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


DISTILLMOS_SAMPLE_RATE = 16_000


class DistillMOSDataset(Dataset):
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        path_str = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load_with_torchcodec(path_str)
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
    for path_str in file_paths:
        try:
            info = torchaudio.info(path_str)
            if info.sample_rate and info.num_frames:
                lengths[path_str] = float(info.num_frames) / float(info.sample_rate)
            else:
                lengths[path_str] = 0.0
        except Exception:
            lengths[path_str] = 0.0
    return lengths


def sort_by_length(file_paths: List[str]) -> List[str]:
    lengths = estimate_audio_lengths(file_paths)
    return sorted(file_paths, key=lambda p: lengths.get(p, 0.0))


def create_distillmos_dataloader(
    file_paths: List[str],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    dataset = DistillMOSDataset(sort_by_length(file_paths))
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": distillmos_collate,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)
