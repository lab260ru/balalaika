import math
from functools import partial
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


DENOISING_SAMPLE_RATE = 48_000


def normalize_to_int16(audio: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(audio))
    scale = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scale)).astype(np.int16)


def next_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return int(value)
    return int(math.ceil(value / multiple) * multiple)


def make_noise_padding(audio: torch.Tensor, pad_len: int) -> torch.Tensor:
    if pad_len <= 0:
        return audio.new_empty((0,))
    tail = audio[-min(audio.numel(), pad_len) :].float()
    rms = torch.sqrt(torch.mean(tail * tail)) if tail.numel() else torch.tensor(0.0)
    if float(rms) <= 0.0:
        return torch.zeros((pad_len,), dtype=audio.dtype)
    noise = torch.randn((pad_len,), dtype=torch.float32) * rms
    return noise.clamp(-32768.0, 32767.0).to(torch.int16)


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
            audio = normalize_to_int16(waveform.squeeze(0).numpy())
            if audio.size == 0:
                raise RuntimeError("empty audio")
            return path, torch.from_numpy(audio), int(audio.shape[0]), ""
        except Exception as exc:
            return path, torch.empty(0, dtype=torch.int16), 0, str(exc)


def denoising_collate(
    batch,
    pad_to_multiple: int = 384,
    pad_mode: str = "noise",
    max_padded_len: int = 96_000,
) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[str]]:
    paths, waveforms, lengths, errors = zip(*batch)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64)
    padded_len = max(int(lengths_tensor.max()), int(pad_to_multiple))
    padded_len = next_multiple(padded_len, int(pad_to_multiple))
    if max_padded_len and padded_len > int(max_padded_len):
        raise RuntimeError(
            f"Batch padded length {padded_len} exceeds max_padded_len={max_padded_len}. "
            "Increase the TensorRT max profile shape or use shorter files."
        )

    padded = torch.zeros((len(batch), 1, padded_len), dtype=torch.int16)
    for idx, waveform in enumerate(waveforms):
        length = int(lengths[idx])
        if length <= 0:
            continue
        padded[idx, 0, :length] = waveform
        pad_len = padded_len - length
        if pad_len > 0 and pad_mode == "noise":
            padded[idx, 0, length:] = make_noise_padding(waveform, pad_len)

    return list(paths), padded.contiguous(), lengths_tensor, list(errors)


def create_denoising_dataloader(
    file_paths: List[str],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    sample_rate: int = DENOISING_SAMPLE_RATE,
    pad_to_multiple: int = 384,
    pad_mode: str = "noise",
    max_padded_len: int = 96_000,
) -> DataLoader:
    dataset = DenoisingDataset(file_paths, sample_rate=sample_rate)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,
        "collate_fn": partial(
            denoising_collate,
            pad_to_multiple=pad_to_multiple,
            pad_mode=pad_mode,
            max_padded_len=max_padded_len,
        ),
        "persistent_workers": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)
