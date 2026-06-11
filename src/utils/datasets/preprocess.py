import time
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio

from src.utils.io_profile import clamp_loader_workers
from loguru import logger
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
        started_at = time.perf_counter()
        try:
            waveform, sample_rate = torchaudio.load_with_torchcodec(path)
            waveform = waveform.to(dtype=torch.float32)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
            waveform = waveform.contiguous()
            length = int(waveform.numel())
            if length == 0:
                return path, 0.0, 0.0, int(sample_rate), 0, ""

            peak = float(waveform.abs().amax().item())
            sum_squares = float(waveform.square().sum().item())
            logger.debug(
                f"dataloader_audio_load dataset=crest path={path} "
                f"seconds={time.perf_counter() - started_at:.6f} "
                f"sample_rate={int(sample_rate)} frames={length}"
            )
            return path, peak, sum_squares, int(sample_rate), length, ""
        except Exception as exc:
            logger.debug(
                f"dataloader_audio_load dataset=crest path={path} "
                f"seconds={time.perf_counter() - started_at:.6f} error={exc}"
            )
            return path, 0.0, 0.0, 0, 0, str(exc)


def crest_factor_worker_init(_: int) -> None:
    torch.set_num_threads(1)


def crest_factor_collate(batch):
    paths, peaks, sum_squares, sample_rates, lengths, errors = zip(*batch)
    peaks_tensor = torch.tensor(peaks, dtype=torch.float32)
    sum_squares_tensor = torch.tensor(sum_squares, dtype=torch.float32)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64)
    sample_rates_tensor = torch.tensor(sample_rates, dtype=torch.int64)
    return (
        list(paths),
        peaks_tensor,
        sum_squares_tensor,
        lengths_tensor,
        sample_rates_tensor,
        list(errors),
    )


def create_crest_factor_dataloader(
    file_paths: List[str],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    dataset = CrestFactorDataset(file_paths)
    num_workers = clamp_loader_workers(num_workers, file_paths)
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
        loader_kwargs["worker_init_fn"] = crest_factor_worker_init
    return DataLoader(dataset, **loader_kwargs)


class LoudnessNormalizeDataset(Dataset):
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        started_at = time.perf_counter()
        try:
            waveform, sample_rate = torchaudio.load_with_torchcodec(path)
            logger.debug(
                f"dataloader_audio_load dataset=loudness path={path} "
                f"seconds={time.perf_counter() - started_at:.6f} "
                f"sample_rate={int(sample_rate)} frames={int(waveform.shape[-1])}"
            )
            return path, waveform.to(dtype=torch.float32).contiguous(), int(sample_rate), ""
        except Exception as exc:
            logger.debug(
                f"dataloader_audio_load dataset=loudness path={path} "
                f"seconds={time.perf_counter() - started_at:.6f} error={exc}"
            )
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
    num_workers = clamp_loader_workers(num_workers, file_paths)
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
    through VRAM. The native file is *not* re-read from disk here — the file's
    raw encoded bytes are read **once** with :meth:`pathlib.Path.read_bytes`
    and both the 16 kHz decode (here) and the downstream native-rate decode
    (crest/LUFS postprocessing) consume the *same* bytes. ``torchcodec``
    decodes identically from a ``bytes`` source and from a path source, so
    outputs stay bit-identical while halving disk reads on cold-cache HDDs.

    Shipping the raw bytes through the batch costs RAM, so it is gated on
    ``raw_bytes_max_duration_s``: bytes are only carried back when the decoded
    audio is no longer than that many seconds (chunks ~0.5-2 MB each). Longer
    sources (e.g. multi-hour raw podcasts) decode from the path as before and
    ship ``None`` so prefetch RAM stays bounded. ``None`` disables byte reuse
    entirely (legacy path-only behavior).
    """

    def __init__(
        self,
        file_paths: List[str],
        *,
        raw_bytes_max_duration_s: Optional[float] = None,
    ):
        self.file_paths = [str(p) for p in file_paths]
        self.raw_bytes_max_duration_s = raw_bytes_max_duration_s

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        started_at = time.perf_counter()
        ship_bytes = self.raw_bytes_max_duration_s is not None
        try:
            if ship_bytes:
                raw_bytes = Path(path).read_bytes()
                source = raw_bytes
            else:
                raw_bytes = None
                source = path
            decoder = AudioDecoder(source, sample_rate=DIARIZATION_SAMPLE_RATE)
            samples = decoder.get_all_samples()
            waveform = samples.data.to(dtype=torch.float32)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            # Only carry bytes back when the decode is short enough to be cheap.
            if ship_bytes:
                duration_s = float(waveform.shape[-1]) / float(DIARIZATION_SAMPLE_RATE)
                if duration_s > float(self.raw_bytes_max_duration_s):
                    raw_bytes = None
            logger.debug(
                f"dataloader_audio_load dataset=diarization path={path} "
                f"seconds={time.perf_counter() - started_at:.6f} "
                f"sample_rate={DIARIZATION_SAMPLE_RATE} frames={int(waveform.shape[-1])} "
                f"raw_bytes={'yes' if raw_bytes is not None else 'no'}"
            )
            return path, waveform.contiguous(), DIARIZATION_SAMPLE_RATE, "", raw_bytes
        except Exception as exc:
            logger.debug(
                f"dataloader_audio_load dataset=diarization path={path} "
                f"seconds={time.perf_counter() - started_at:.6f} error={exc}"
            )
            return path, torch.empty(0, dtype=torch.float32), 0, str(exc), None


def diarization_collate(batch):
    return batch


def create_diarization_dataloader(
    file_paths: List[str],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    raw_bytes_max_duration_s: Optional[float] = None,
) -> DataLoader:
    dataset = DiarizationDataset(
        file_paths, raw_bytes_max_duration_s=raw_bytes_max_duration_s
    )
    num_workers = clamp_loader_workers(num_workers, file_paths)
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
