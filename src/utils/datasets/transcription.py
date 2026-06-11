from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.utils.io_profile import clamp_loader_workers
from src.utils.logging_setup import dataloader_worker_init as _worker_init


class TranscriptionDataset(Dataset):
    def __init__(self, file_paths: List[str], sample_rate: int):
        self.file_paths = file_paths
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        try:
            waveform, sample_rate = torchaudio.load_with_torchcodec(path)
            waveform = waveform.to(dtype=torch.float32)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
            return path, waveform.squeeze(0).contiguous(), None
        except Exception as exc:
            return path, None, str(exc)


def transcription_collate(batch):
    errors = [(path, error) for path, waveform, error in batch if error]
    valid = [(path, waveform) for path, waveform, error in batch if error is None]
    if not valid:
        return [], torch.empty(0, 0, dtype=torch.float32), torch.empty(0, dtype=torch.int64), errors

    paths, waveforms = zip(*valid)
    lengths = torch.tensor([w.numel() for w in waveforms], dtype=torch.int64)
    padded = pad_sequence(waveforms, batch_first=True)
    return list(paths), padded.contiguous(), lengths, errors


def create_transcription_dataloader(
    file_paths: List[str],
    sample_rate: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    dataset = TranscriptionDataset(file_paths, sample_rate)
    num_workers = clamp_loader_workers(num_workers, file_paths)
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
        loader_kwargs["worker_init_fn"] = _worker_init
    return DataLoader(dataset, **loader_kwargs)


def recognize_batch(model, waveforms: torch.Tensor, lengths: torch.Tensor):
    waveforms_np = np.asarray(waveforms.numpy(), dtype=np.float32)
    lengths_np = np.asarray(lengths.numpy(), dtype=np.int64)
    return list(model._recognize_batch(waveforms_np, lengths_np))


class GroupTranscriptionDataset(Dataset):
    """Decode each file ONCE and resample to every rate the model group needs.

    The per-model :class:`TranscriptionDataset` decodes (and re-reads from
    disk) the same audio once per ASR model — 5x redundant I/O and decode
    for the default model list. This dataset is the shared-decode variant:
    one disk read + decode at native rate, then one resample per distinct
    target sample rate (one in practice — all current models are 16 kHz).
    """

    def __init__(self, file_paths: Sequence[str], sample_rates: Sequence[int]):
        self.file_paths = list(file_paths)
        self.sample_rates = sorted(set(int(r) for r in sample_rates))

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        try:
            waveform, sample_rate = torchaudio.load_with_torchcodec(path)
            waveform = waveform.to(dtype=torch.float32)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            by_rate = {}
            for rate in self.sample_rates:
                resampled = (
                    waveform
                    if sample_rate == rate
                    else torchaudio.functional.resample(waveform, sample_rate, rate)
                )
                by_rate[rate] = resampled.squeeze(0).contiguous()
            return path, by_rate, None
        except Exception as exc:
            return path, None, str(exc)


def group_transcription_collate(batch):
    errors = [(path, error) for path, _, error in batch if error]
    valid = [(path, by_rate) for path, by_rate, error in batch if error is None]
    if not valid:
        return [], {}, errors

    paths = [path for path, _ in valid]
    padded_by_rate: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for rate in valid[0][1]:
        waveforms = [by_rate[rate] for _, by_rate in valid]
        lengths = torch.tensor([w.numel() for w in waveforms], dtype=torch.int64)
        padded = pad_sequence(waveforms, batch_first=True)
        padded_by_rate[rate] = (padded.contiguous(), lengths)
    return paths, padded_by_rate, errors


def create_group_transcription_dataloader(
    file_paths: Sequence[str],
    sample_rates: Sequence[int],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    dataset = GroupTranscriptionDataset(file_paths, sample_rates)
    num_workers = clamp_loader_workers(num_workers, file_paths)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,
        "collate_fn": group_transcription_collate,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["worker_init_fn"] = _worker_init
    return DataLoader(dataset, **loader_kwargs)
