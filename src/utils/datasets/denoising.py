from typing import List, Tuple

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


DENOISING_SAMPLE_RATE = 48_000


class DenoisingDataset(Dataset):
    def __init__(self, file_paths: List[str], sample_rate: int = DENOISING_SAMPLE_RATE):
        self.file_paths = file_paths
        self.sample_rate = int(sample_rate)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        try:
            waveform, source_sample_rate = torchaudio.load_with_torchcodec(path)
            waveform = waveform.to(dtype=torch.float32)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if int(source_sample_rate) != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform,
                    int(source_sample_rate),
                    self.sample_rate,
                )
            waveform = waveform.squeeze(0).contiguous()
            return path, waveform, int(waveform.numel()), ""
        except Exception as exc:
            return path, torch.empty(0, dtype=torch.float32), 0, str(exc)


def denoising_collate(batch) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[str]]:
    paths, waveforms, lengths, errors = zip(*batch)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64)
    padded = pad_sequence(waveforms, batch_first=True)
    return list(paths), padded.contiguous(), lengths_tensor, list(errors)


def create_denoising_dataloader(
    file_paths: List[str],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    sample_rate: int = DENOISING_SAMPLE_RATE,
) -> DataLoader:
    dataset = DenoisingDataset(file_paths, sample_rate=sample_rate)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,
        "collate_fn": denoising_collate,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)
