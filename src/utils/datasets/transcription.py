from typing import List

import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class TranscriptionDataset(Dataset):
    def __init__(self, file_paths: List[str], sample_rate: int):
        self.file_paths = file_paths
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load_with_torchcodec(path)
        waveform = waveform.to(dtype=torch.float32)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        return path, waveform.squeeze(0).contiguous()


def transcription_collate(batch):
    paths, waveforms = zip(*batch)
    lengths = torch.tensor([w.numel() for w in waveforms], dtype=torch.int64)
    padded = pad_sequence(waveforms, batch_first=True)
    return list(paths), padded.contiguous(), lengths


def create_transcription_dataloader(
    file_paths: List[str],
    sample_rate: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    dataset = TranscriptionDataset(file_paths, sample_rate)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,
        "collate_fn": transcription_collate,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def recognize_batch(model, waveforms: torch.Tensor, lengths: torch.Tensor):
    waveforms_np = np.asarray(waveforms.numpy(), dtype=np.float32)
    lengths_np = np.asarray(lengths.numpy(), dtype=np.int64)
    return list(model._recognize_batch(waveforms_np, lengths_np))
