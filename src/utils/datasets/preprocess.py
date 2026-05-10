from typing import List

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class CrestFactorDataset(Dataset):
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        try:
            waveform, sample_rate = torchaudio.load_with_torchcodec(path)
            waveform = waveform.to(dtype=torch.float32)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0).contiguous()
            return path, waveform, int(sample_rate), int(waveform.numel()), ""
        except Exception as exc:
            return path, torch.empty(0, dtype=torch.float32), 0, 0, str(exc)


def crest_factor_collate(batch):
    paths, waveforms, sample_rates, lengths, errors = zip(*batch)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64)
    sample_rates_tensor = torch.tensor(sample_rates, dtype=torch.int64)
    padded = pad_sequence(waveforms, batch_first=True)
    return list(paths), padded.contiguous(), lengths_tensor, sample_rates_tensor, list(errors)


def create_crest_factor_dataloader(
    file_paths: List[str],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    dataset = CrestFactorDataset(file_paths)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": crest_factor_collate,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


class LoudnessNormalizeDataset(Dataset):
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        try:
            waveform, sample_rate = torchaudio.load_with_torchcodec(path)
            return path, waveform.to(dtype=torch.float32).contiguous(), int(sample_rate), ""
        except Exception as exc:
            return path, torch.empty(0, dtype=torch.float32), 0, str(exc)


def loudness_normalize_collate(batch):
    return batch


def create_loudness_normalize_dataloader(
    file_paths: List[str],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    dataset = LoudnessNormalizeDataset(file_paths)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,
        "collate_fn": loudness_normalize_collate,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


class DiarizationDataset(Dataset):
    def __init__(self, file_paths: List[str]):
        self.file_paths = [str(p) for p in file_paths]

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        try:
            waveform, sample_rate = torchaudio.load_with_torchcodec(path)
            return path, waveform.to(dtype=torch.float32).contiguous(), int(sample_rate), ""
        except Exception as exc:
            return path, torch.empty(0, dtype=torch.float32), 0, str(exc)


def diarization_collate(batch):
    return batch


def create_diarization_dataloader(
    file_paths: List[str],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    dataset = DiarizationDataset(file_paths)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,
        "collate_fn": diarization_collate,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)
