from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
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


# --------------------------------------------------------------------------- #
# Persistent (worker-reusing) loaders for the shard claim loop.
#
# The sequential flow builds a fresh DataLoader per claimed work shard. With up
# to 16 (HDD: 4) persistent loader workers that means a full worker spawn +
# prefetch warmup at every shard boundary while the GPU idles. These loaders
# keep ONE DataLoader (and therefore one set of workers) alive across every
# shard a GPU worker claims.
#
# Byte-equivalence to the per-shard map-style loader: a map-style Dataset reads
# its file list from a ``multiprocessing.Manager`` state object shared with the
# (persistent) worker processes. The parent sets the next shard's files, bumps a
# generation counter, then re-iterates the SAME DataLoader. ``shuffle=False`` ⇒
# the sampler emits indices ``0..n-1`` for the current shard, so batches are the
# same sequential slices (``[0:B], [B:2B], …``, partial last batch) as a fresh
# per-shard loader — for any shard size and any worker count. Workers re-read
# (snapshot once, generation-gated) the file list at the start of each epoch.
# --------------------------------------------------------------------------- #


class _ShardState:
    """Manager-backed shard payload shared with persistent loader workers.

    Holds the current shard's file list plus a generation counter. Workers
    snapshot the list once per generation (one bulk IPC per worker per shard),
    so per-index ``__getitem__`` access is local — not an RPC per file.
    """

    def __init__(self, manager):
        self._manager = manager
        self._state = manager.dict()
        self._state["gen"] = 0
        self._state["files"] = manager.list()

    def set_shard(self, files: Sequence[str]) -> None:
        # A brand-new Manager list per shard keeps the proxy small and avoids
        # mutating a list a worker may still be snapshotting.
        self._state["files"] = self._manager.list([str(f) for f in files])
        self._state["gen"] = int(self._state["gen"]) + 1

    @property
    def generation(self) -> int:
        return int(self._state["gen"])

    def current_len(self) -> int:
        return len(self._state["files"])

    def proxy(self):
        return self._state


class _SharedFileMixin:
    """Generation-gated local snapshot of the shared file list."""

    def __init__(self, state_proxy):
        self._state = state_proxy
        self._gen = -1
        self._snapshot: Optional[List[str]] = None

    def _files(self) -> List[str]:
        gen = int(self._state["gen"])
        if gen != self._gen or self._snapshot is None:
            self._snapshot = list(self._state["files"])
            self._gen = gen
        return self._snapshot

    def __len__(self) -> int:
        # Read live so the parent's sampler tracks the current shard length.
        return len(self._state["files"])


class _ShardedTranscriptionDataset(_SharedFileMixin, Dataset):
    """Map-style :class:`TranscriptionDataset` fed by a shared shard list."""

    def __init__(self, state_proxy, sample_rate: int):
        _SharedFileMixin.__init__(self, state_proxy)
        self.sample_rate = sample_rate

    def __getitem__(self, idx: int):
        path = self._files()[idx]
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


class _ShardedGroupTranscriptionDataset(_SharedFileMixin, Dataset):
    """Map-style :class:`GroupTranscriptionDataset` fed by a shared shard list."""

    def __init__(self, state_proxy, sample_rates: Sequence[int]):
        _SharedFileMixin.__init__(self, state_proxy)
        self.sample_rates = sorted(set(int(r) for r in sample_rates))

    def __getitem__(self, idx: int):
        path = self._files()[idx]
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


class _BasePersistentLoader:
    """Owns a Manager, the shared shard state, and one reused DataLoader.

    Use as a context manager. Call :meth:`iter_shard` once per claimed shard;
    it yields the same batch sequence a freshly built per-shard loader would.
    The worker count is resolved from the FIRST shard's files (the disk that
    holds the dataset does not change mid-stage), matching the per-shard
    loader's ``clamp_loader_workers`` decision.
    """

    def __init__(self, batch_size: int, num_workers: int, prefetch_factor: int):
        self._batch_size = int(batch_size)
        self._configured_workers = int(num_workers)
        self._prefetch_factor = int(prefetch_factor)
        self._manager = None
        self._state: Optional[_ShardState] = None
        self._loader: Optional[DataLoader] = None
        self._num_workers: Optional[int] = None

    # Subclasses build their dataset + collate.
    def _make_dataset(self, state_proxy):  # pragma: no cover - abstract
        raise NotImplementedError

    def _collate_fn(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def __enter__(self):
        self._manager = mp.Manager()
        self._state = _ShardState(self._manager)
        return self

    def _ensure_loader(self, first_files: Sequence[str]) -> None:
        if self._loader is not None:
            return
        self._num_workers = clamp_loader_workers(self._configured_workers, list(first_files))
        loader_kwargs = {
            "batch_size": self._batch_size,
            "shuffle": False,
            "num_workers": self._num_workers,
            "pin_memory": False,
            "collate_fn": self._collate_fn(),
            "persistent_workers": self._num_workers > 0,
        }
        if self._num_workers > 0:
            loader_kwargs["prefetch_factor"] = self._prefetch_factor
            loader_kwargs["worker_init_fn"] = _worker_init
        self._loader = DataLoader(self._make_dataset(self._state.proxy()), **loader_kwargs)

    def iter_shard(self, files: Sequence[str]):
        """Set the shard and return an iterator over its batches.

        Identical batch sequence to ``create_*_dataloader(files, …)`` iterated
        once, but the worker processes are reused across calls.
        """
        files = list(files)
        if not files:
            # An empty shard yields no batches; don't anchor the worker-count
            # decision (clamp_loader_workers) on an empty list.
            return iter(())
        self._ensure_loader(files)
        self._state.set_shard(files)
        return iter(self._loader)

    def __exit__(self, exc_type, exc, tb):
        # Drop the DataLoader first so its persistent workers exit before the
        # Manager server they read from goes away.
        if self._loader is not None:
            del self._loader
            self._loader = None
        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None
        self._state = None
        return False


class PersistentTranscriptionLoader(_BasePersistentLoader):
    """Persistent single-rate loader for the sequential model flow."""

    def __init__(self, sample_rate: int, batch_size: int, num_workers: int, prefetch_factor: int):
        super().__init__(batch_size, num_workers, prefetch_factor)
        self._sample_rate = int(sample_rate)

    def _make_dataset(self, state_proxy):
        return _ShardedTranscriptionDataset(state_proxy, self._sample_rate)

    def _collate_fn(self):
        return transcription_collate


class PersistentGroupTranscriptionLoader(_BasePersistentLoader):
    """Persistent shared-decode loader for the grouped model flow."""

    def __init__(self, sample_rates: Sequence[int], batch_size: int, num_workers: int, prefetch_factor: int):
        super().__init__(batch_size, num_workers, prefetch_factor)
        self._sample_rates = list(sample_rates)

    def _make_dataset(self, state_proxy):
        return _ShardedGroupTranscriptionDataset(state_proxy, self._sample_rates)

    def _collate_fn(self):
        return group_transcription_collate
