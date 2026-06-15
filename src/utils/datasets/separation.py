import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import soundfile as sf
import torch
import torchaudio
from loguru import logger as loguru_logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils.io_profile import clamp_loader_workers
from src.utils.logging_setup import dataloader_worker_init

logger = logging.getLogger(__name__)


DISTILLMOS_SAMPLE_RATE = 16_000
ANTISPOOF_SAMPLE_RATE = 16_000
ANTISPOOF_NUM_SAMPLES = 64_600
MUSICDETECT_SAMPLE_RATE = 16_000

# Containers/subtypes for which a soundfile ranged read has been proven to
# reproduce torchcodec's full-decode float32 values bit-exactly (torch.equal).
# WAV and FLAC are both lossless PCM containers; the subtypes below all decode
# to identical float32 via libsndfile and torchcodec (verified on fixtures).
_RANGED_DECODE_FORMATS = frozenset({"WAV", "WAVEX", "FLAC"})
_RANGED_DECODE_SUBTYPES = frozenset({"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"})


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


def distillmos_collate(
    batch: List[Tuple[str, torch.Tensor]],
) -> Tuple[List[str], torch.Tensor]:
    paths, waves = zip(*batch)
    padded = pad_sequence(waves, batch_first=True)
    return list(paths), padded


def estimate_audio_lengths(file_paths: List[str]) -> Dict[str, float]:
    from src.utils.audit import safe_audio_duration

    lengths = {}
    for path_str in tqdm(file_paths, desc="Read audio before start"):
        lengths[path_str] = safe_audio_duration(path_str)
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
            logger.info(
                "Cache partial hit: %d cached + %d new files",
                len(cached),
                len(new_files),
            )
            lengths = estimate_audio_lengths(new_files)
            new_sorted = sorted(new_files, key=lambda p: lengths.get(p, 0.0))
            result = cached + new_sorted
            with open(cache_file, "w") as f:
                json.dump(result, f)
            return result

        logger.info(
            "Cache stale (%d cached vs %d current), rebuilding",
            len(cached),
            len(file_paths),
        )

    logger.info("Scanning %d audio files for length sorting...", len(file_paths))
    lengths = estimate_audio_lengths(file_paths)
    sorted_paths = sorted(file_paths, key=lambda p: lengths.get(p, 0.0))

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(sorted_paths, f)
        logger.info(
            "Saved sorted file list (%d files) to %s", len(sorted_paths), cache_file
        )

    return sorted_paths


def create_distillmos_dataloader(
    file_paths: List[str],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    cache_dir: str = "",
    assume_sorted: bool = False,
) -> DataLoader:
    # Work shards from prepare_length_bucketed_work_shards are already
    # duration-sorted; re-probing every file (and thrashing the shared JSON
    # cache, which never hits across disjoint shards) is dead work.
    ordered = (
        file_paths if assume_sorted else sort_by_length(file_paths, cache_dir=cache_dir)
    )
    dataset = DistillMOSDataset(ordered)
    num_workers = clamp_loader_workers(num_workers, file_paths)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        # pinned staging makes the stage's non_blocking H2D copy real
        # (~35% faster copy, can overlap compute); without it non_blocking
        # silently degrades to a synchronous pageable copy
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": distillmos_collate,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["worker_init_fn"] = dataloader_worker_init
    return DataLoader(dataset, **loader_kwargs)


class AntiSpoofingDataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        sample_rate: int = ANTISPOOF_SAMPLE_RATE,
        num_samples: int = ANTISPOOF_NUM_SAMPLES,
        ranged_decode: bool = False,
    ):
        self.file_paths = file_paths
        self.sample_rate = int(sample_rate)
        self.num_samples = int(num_samples)
        self.ranged_decode = bool(ranged_decode)

    def __len__(self) -> int:
        return len(self.file_paths)

    def _try_ranged_load(self, path_str: str):
        """Attempt a seek+read of only the random window (plus one predecessor
        sample for preemphasis), avoiding a full decode for long clips.

        Returns ``(waveform, original_length)`` where ``waveform`` is a 1-D
        float32 mono tensor of exactly ``num_samples`` samples post-preemphasis,
        identical to the full-decode-then-crop result, and ``original_length``
        is the source frame count (equal to the full-decode path's value since
        no resampling happens). Returns ``None`` if this clip is not eligible
        for the ranged fast path (caller must fall back to full-decode).

        IMPORTANT: this method consumes exactly one ``random.randint`` call,
        the same as ``_pad_random`` does for clips longer than ``num_samples``,
        so downstream RNG state stays identical to the full-decode path.
        """
        try:
            info = sf.info(path_str)
        except Exception:
            return None

        # Resampling has unbounded context, so a ranged read is not equivalent
        # there. Only proven lossless containers/subtypes qualify.
        if int(info.samplerate) != self.sample_rate:
            return None
        if str(info.format).upper() not in _RANGED_DECODE_FORMATS:
            return None
        if str(info.subtype).upper() not in _RANGED_DECODE_SUBTYPES:
            return None

        wave_len = int(info.frames)
        # Short clips are repeat-padded by _pad_random and consume no RNG; the
        # ranged path only handles the long-clip (crop) case to keep the RNG
        # consumption pattern identical to the full-decode path.
        if wave_len < self.num_samples:
            return None

        # Draw the random start exactly as _pad_random does.
        start = random.randint(0, wave_len - self.num_samples)

        if start > 0:
            # Read one sample early so preemphasis y[n]=x[n]-a*x[n-1] has the
            # correct predecessor at the window start; drop it afterwards.
            read_start = start - 1
            read_count = self.num_samples + 1
            drop_leading = True
        else:
            # No predecessor exists at offset 0; preemphasis leaves y[0]=x[0]
            # unchanged, matching the full-decode crop at start==0 exactly.
            read_start = 0
            read_count = self.num_samples
            drop_leading = False

        block, _ = sf.read(
            path_str,
            start=read_start,
            frames=read_count,
            dtype="float32",
            always_2d=True,
        )
        # always_2d -> (frames, channels); transpose to (channels, frames) to
        # match torchaudio/torchcodec channel-first layout used downstream.
        waveform = torch.from_numpy(block).transpose(0, 1)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
        waveform = torchaudio.functional.preemphasis(waveform.unsqueeze(0)).squeeze(0)
        if drop_leading:
            waveform = waveform[1:]
        return waveform.contiguous(), wave_len

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
            if self.ranged_decode:
                ranged = self._try_ranged_load(path_str)
                if ranged is not None:
                    waveform, original_length = ranged
                    loguru_logger.debug(
                        f"dataloader_audio_load dataset=antispoofing path={path_str} "
                        f"seconds={time.perf_counter() - started_at:.6f} "
                        f"sample_rate={self.sample_rate} frames={original_length} "
                        f"ranged=1"
                    )
                    return path_str, waveform, original_length, ""

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
    valid_lengths = torch.tensor(
        [lengths[idx] for idx in valid_indices], dtype=torch.int64
    )
    valid_errors = [
        (paths[idx], errors[idx]) for idx in range(len(paths)) if errors[idx]
    ]

    if not valid_indices:
        return (
            [],
            torch.empty(0, 0, dtype=torch.float32),
            torch.empty(0, dtype=torch.int64),
            valid_errors,
        )

    batch_tensor = torch.stack([waveforms[idx] for idx in valid_indices]).contiguous()
    return valid_paths, batch_tensor, valid_lengths, valid_errors


def create_antispoofing_dataloader(
    file_paths: List[str],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    sample_rate: int = ANTISPOOF_SAMPLE_RATE,
    num_samples: int = ANTISPOOF_NUM_SAMPLES,
    ranged_decode: bool = False,
) -> DataLoader:
    dataset = AntiSpoofingDataset(
        file_paths,
        sample_rate=sample_rate,
        num_samples=num_samples,
        ranged_decode=ranged_decode,
    )
    num_workers = clamp_loader_workers(num_workers, file_paths)
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
        loader_kwargs["worker_init_fn"] = dataloader_worker_init
    return DataLoader(dataset, **loader_kwargs)


class MusicDetectionDataset(Dataset):
    """Mono 16 kHz waveform loader for the WavLM music detector.

    Replaces the former ``musicdetection.MusicDetectionDataset`` plus
    ``transformers.AutoFeatureExtractor`` pair so the music-detect stage no
    longer imports either package. The wavlm-base-plus feature extractor is
    configured with ``do_normalize=false`` (verified from its
    ``preprocessor_config.json``), so feature extraction reduces to
    pad-to-longest plus an attention mask — done in :func:`musicdetect_collate`.
    Decoding here mirrors the other separation datasets (torchcodec decode +
    ``functional.resample``).
    """

    def __init__(
        self,
        file_paths: List[str],
        sample_rate: int = MUSICDETECT_SAMPLE_RATE,
    ):
        self.file_paths = file_paths
        self.sample_rate = int(sample_rate)

    def __len__(self) -> int:
        return len(self.file_paths)

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
            length = int(waveform.numel())
            if length <= 0:
                raise ValueError("empty audio")
            loguru_logger.debug(
                f"dataloader_audio_load dataset=music_detect path={path_str} "
                f"seconds={time.perf_counter() - started_at:.6f} "
                f"sample_rate={self.sample_rate} frames={length}"
            )
            return path_str, waveform.contiguous(), length, ""
        except Exception as exc:
            loguru_logger.debug(
                f"dataloader_audio_load dataset=music_detect path={path_str} "
                f"seconds={time.perf_counter() - started_at:.6f} error={exc}"
            )
            return path_str, torch.empty(0, dtype=torch.float32), 0, str(exc)


def musicdetect_collate(batch):
    """Pad mono waveforms to the batch max and build a 0/1 attention mask.

    Returns ``(paths, input_values, attention_mask, lengths, errors)`` where
    ``input_values`` is ``float32 [B, T]`` (padding value 0.0) and
    ``attention_mask`` is ``int32 [B, T]`` (1 = real sample, 0 = pad), matching
    the ONNX graph's declared input dtypes.
    """
    paths, waveforms, lengths, errors = zip(*batch)
    valid_indices = [idx for idx, error in enumerate(errors) if not error]
    valid_errors = [
        (paths[idx], errors[idx]) for idx in range(len(paths)) if errors[idx]
    ]

    if not valid_indices:
        return (
            [],
            torch.empty(0, 0, dtype=torch.float32),
            torch.empty(0, 0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int64),
            valid_errors,
        )

    valid_paths = [paths[idx] for idx in valid_indices]
    valid_waveforms = [waveforms[idx] for idx in valid_indices]
    valid_lengths = torch.tensor(
        [lengths[idx] for idx in valid_indices], dtype=torch.int64
    )
    input_values = pad_sequence(valid_waveforms, batch_first=True)
    max_len = int(input_values.shape[1])
    attention_mask = (torch.arange(max_len)[None, :] < valid_lengths[:, None]).to(
        torch.int32
    )
    return (
        valid_paths,
        input_values.contiguous(),
        attention_mask.contiguous(),
        valid_lengths,
        valid_errors,
    )


def create_music_detect_dataloader(
    file_paths: List[str],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    sample_rate: int = MUSICDETECT_SAMPLE_RATE,
) -> DataLoader:
    dataset = MusicDetectionDataset(file_paths, sample_rate=sample_rate)
    num_workers = clamp_loader_workers(num_workers, file_paths)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,
        "collate_fn": musicdetect_collate,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["worker_init_fn"] = dataloader_worker_init
    return DataLoader(dataset, **loader_kwargs)
