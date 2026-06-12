import math
import time
from functools import partial
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from src.utils.io_profile import clamp_loader_workers


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
        # Cache one transforms.Resample per source rate. torchaudio.functional
        # .resample rebuilds the (sinc) polyphase kernel on EVERY call; the
        # transform precomputes and reuses it. dtype MUST be pinned to float32:
        # transforms.Resample builds its kernel in float64 by default (then
        # casts), whereas functional.resample on a float32 waveform builds a
        # float32 kernel -- the two kernels are NOT equal, so an unpinned
        # transform diverges from the functional call (~2e-5 float drift, which
        # flips int16 LSBs after normalize_to_int16). With dtype=float32 the
        # resampled tensor is bit-identical, just without the per-file rebuild.
        # Each loader worker gets its own dataset copy, so this dict is
        # process-local and needs no locking.
        self._resamplers: dict[int, "torchaudio.transforms.Resample"] = {}

    def __len__(self) -> int:
        return len(self.file_paths)

    def _resample(self, waveform: torch.Tensor, source_sample_rate: int) -> torch.Tensor:
        resampler = self._resamplers.get(source_sample_rate)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(
                orig_freq=source_sample_rate,
                new_freq=self.sample_rate,
                dtype=torch.float32,
            )
            self._resamplers[source_sample_rate] = resampler
        return resampler(waveform)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        started_at = time.perf_counter()
        try:
            waveform, source_sample_rate = torchaudio.load_with_torchcodec(path)
            waveform = waveform.to(dtype=torch.float32)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if int(source_sample_rate) != self.sample_rate:
                waveform = self._resample(waveform, int(source_sample_rate))
            audio = normalize_to_int16(waveform.squeeze(0).numpy())
            if audio.size == 0:
                raise RuntimeError("empty audio")
            logger.debug(
                f"dataloader_audio_load dataset=denoising path={path} "
                f"seconds={time.perf_counter() - started_at:.6f} "
                f"sample_rate={self.sample_rate} frames={int(audio.shape[0])}"
            )
            return path, torch.from_numpy(audio), int(audio.shape[0]), ""
        except Exception as exc:
            logger.debug(
                f"dataloader_audio_load dataset=denoising path={path} "
                f"seconds={time.perf_counter() - started_at:.6f} error={exc}"
            )
            return path, torch.empty(0, dtype=torch.int16), 0, str(exc)


def denoising_collate(
    batch,
    pad_to_multiple: int = 384,
    pad_mode: str = "noise",
    max_padded_len: int = 96_000,
) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[str]]:
    paths, waveforms, lengths, errors = zip(*batch)
    paths = list(paths)
    lengths = list(lengths)
    errors = list(errors)

    # A single file longer than the model's hard input cap used to raise here,
    # which propagated through the DataLoader, killed the GPU worker, and
    # abandoned its whole claimed shard. Instead mark each oversize item as a
    # per-item error (counted + skipped exactly like a decode failure) and zero
    # its length so it never enters padding/inference. Valid files are
    # unaffected: padded_len is computed over the survivors only, so a giant
    # neighbour no longer pads — or fails — the rest of the batch.
    cap = int(max_padded_len) if max_padded_len else 0
    if cap:
        for idx in range(len(lengths)):
            if errors[idx]:
                continue
            needed = next_multiple(int(lengths[idx]), int(pad_to_multiple))
            if needed > cap:
                errors[idx] = (
                    f"audio length {int(lengths[idx])} exceeds model max "
                    f"{cap} samples ({cap / DENOISING_SAMPLE_RATE:.1f}s); skipped"
                )
                lengths[idx] = 0

    lengths_tensor = torch.tensor(lengths, dtype=torch.int64)
    valid_max = max((int(l) for l in lengths), default=0)
    padded_len = max(valid_max, int(pad_to_multiple))
    padded_len = next_multiple(padded_len, int(pad_to_multiple))

    padded = torch.zeros((len(batch), 1, padded_len), dtype=torch.int16)
    for idx, waveform in enumerate(waveforms):
        length = int(lengths[idx])
        if length <= 0:
            continue
        padded[idx, 0, :length] = waveform
        pad_len = padded_len - length
        if pad_len > 0 and pad_mode == "noise":
            padded[idx, 0, length:] = make_noise_padding(waveform, pad_len)

    return paths, padded.contiguous(), lengths_tensor, errors


def denoising_worker_init(_: int) -> None:
    # Each spawned loader worker otherwise inherits torch's default intra-op
    # thread count (up to all physical cores). With several workers x several
    # ORT sessions on a CPU shared with the training job, that oversubscribes
    # the box. Per-file decode/resample is already parallelized across the
    # workers, so per-op threading only adds scheduler thrash. Mirrors
    # crest_factor_worker_init; conv1d parallelizes over output positions so
    # the resampled tensor stays bit-identical at 1 thread.
    torch.set_num_threads(1)


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
    num_workers = clamp_loader_workers(num_workers, file_paths)
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
        loader_kwargs["worker_init_fn"] = denoising_worker_init
    return DataLoader(dataset, **loader_kwargs)
