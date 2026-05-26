from typing import List

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchcodec.decoders import AudioDecoder

DIARIZATION_SAMPLE_RATE = 16_000


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
        "pin_memory": False,
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
    """Stream audio decoded to ``DIARIZATION_SAMPLE_RATE`` (16 kHz) mono.

    Sortformer and SmartVAD both run at 16 kHz, so we resample inside the
    decoder rather than carrying the native-rate (often 44.1/48 kHz) waveform
    through VRAM. The native file is *not* touched here — chunk export reads
    it lazily later via :class:`torchcodec.decoders.AudioDecoder`, so audio
    cuts keep the source's original quality.
    """

    def __init__(self, file_paths: List[str]):
        self.file_paths = [str(p) for p in file_paths]

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        try:
            decoder = AudioDecoder(path, sample_rate=DIARIZATION_SAMPLE_RATE)
            samples = decoder.get_all_samples()
            waveform = samples.data.to(dtype=torch.float32)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return path, waveform.contiguous(), DIARIZATION_SAMPLE_RATE, ""
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
