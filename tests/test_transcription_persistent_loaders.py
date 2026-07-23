"""Persistent (worker-reusing) transcription loaders must yield the EXACT same
sequence of (shard, batch contents, order) as building a fresh per-shard
DataLoader, so the byte-equivalence the stage depends on (CTC models are
order-sensitive — see report.md §11.1) is preserved.

Layer 1 (``test_persistent_*_matches_per_shard_loader``) is the load-bearing
pin: it drives the persistent loaders over several shards of REAL tiny wavs
with worker processes and compares every batch — paths, waveform tensors,
lengths — against the per-shard ``create_*_dataloader`` oracle.

Layer 2 runs the whole stage (``tr.main``) with loader workers and asserts the
sidecars are byte-identical across persistent_loaders=True / False and the
num_workers=0 reference.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

import src.transcription.transcription as tr
from src.utils.datasets.transcription import (
    PersistentGroupTranscriptionLoader,
    PersistentTranscriptionLoader,
    create_group_transcription_dataloader,
    create_transcription_dataloader,
)

SAMPLE_RATE = 16_000


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def _make_wavs(root: Path, n: int) -> list[str]:
    paths = []
    for i in range(n):
        d = root / f"dir{i % 3}"
        d.mkdir(parents=True, exist_ok=True)
        rate = 8000 if i % 4 == 0 else SAMPLE_RATE  # exercise resampling
        dur = 0.20 + 0.03 * i
        t = np.arange(int(dur * rate)) / rate
        amp = (i + 1) / 100.0
        wave = (amp * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
        p = d / f"c{i:03d}.wav"
        sf.write(p, wave, rate)
        paths.append(str(p))
    return paths


def _split_shards(paths: list[str], sizes: list[int]) -> list[list[str]]:
    shards, k = [], 0
    for s in sizes:
        shards.append(paths[k:k + s])
        k += s
    assert k == len(paths)
    return shards


def _per_shard_single(shards, *, batch_size, num_workers, sample_rate):
    """Oracle: a fresh per-shard DataLoader, exactly as the old flow."""
    out = []
    for files in shards:
        dl = create_transcription_dataloader(
            files, sample_rate=sample_rate, batch_size=batch_size,
            num_workers=num_workers, prefetch_factor=2,
        )
        batches = []
        for paths, waveforms, lengths, errs in dl:
            batches.append((list(paths), waveforms.clone(), lengths.clone(), list(errs)))
        out.append(batches)
        del dl
    return out


def _persistent_single(shards, *, batch_size, num_workers, sample_rate):
    out = []
    with PersistentTranscriptionLoader(
        sample_rate=sample_rate, batch_size=batch_size,
        num_workers=num_workers, prefetch_factor=2,
    ) as loader:
        for files in shards:
            batches = []
            for paths, waveforms, lengths, errs in loader.iter_shard(files):
                batches.append((list(paths), waveforms.clone(), lengths.clone(), list(errs)))
            out.append(batches)
    return out


def _per_shard_group(shards, *, batch_size, num_workers, sample_rates):
    out = []
    for files in shards:
        dl = create_group_transcription_dataloader(
            files, sample_rates=sample_rates, batch_size=batch_size,
            num_workers=num_workers, prefetch_factor=2,
        )
        batches = []
        for paths, padded_by_rate, errs in dl:
            snap = {r: (p.clone(), l.clone()) for r, (p, l) in padded_by_rate.items()}
            batches.append((list(paths), snap, list(errs)))
        out.append(batches)
        del dl
    return out


def _persistent_group(shards, *, batch_size, num_workers, sample_rates):
    out = []
    with PersistentGroupTranscriptionLoader(
        sample_rates=sample_rates, batch_size=batch_size,
        num_workers=num_workers, prefetch_factor=2,
    ) as loader:
        for files in shards:
            batches = []
            for paths, padded_by_rate, errs in loader.iter_shard(files):
                snap = {r: (p.clone(), l.clone()) for r, (p, l) in padded_by_rate.items()}
                batches.append((list(paths), snap, list(errs)))
            out.append(batches)
    return out


def _assert_single_equal(a, b):
    assert len(a) == len(b)
    for sa, sb in zip(a, b):  # per shard
        assert len(sa) == len(sb), "batch count differs"
        for (pa, wa, la, ea), (pb, wb, lb, eb) in zip(sa, sb):
            assert pa == pb, "batch paths/order differ"
            assert torch.equal(la, lb), "lengths differ"
            assert wa.shape == wb.shape, "padded shape differs"
            assert torch.equal(wa, wb), "waveform bytes differ"
            assert ea == eb


def _assert_group_equal(a, b):
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        assert len(sa) == len(sb), "batch count differs"
        for (pa, da, ea), (pb, db, eb) in zip(sa, sb):
            assert pa == pb, "batch paths/order differ"
            assert set(da) == set(db), "rate keys differ"
            for r in da:
                wpa, lpa = da[r]
                wpb, lpb = db[r]
                assert torch.equal(lpa, lpb), "group lengths differ"
                assert wpa.shape == wpb.shape and torch.equal(wpa, wpb), "group waveform bytes differ"
            assert ea == eb


# --------------------------------------------------------------------------- #
# Layer 1 — load-bearing batch-sequence equivalence
# --------------------------------------------------------------------------- #

def test_persistent_single_matches_per_shard_loader(tmp_path):
    paths = _make_wavs(tmp_path, 17)
    shards = _split_shards(paths, [5, 5, 4, 3])  # uneven last batches per shard
    bs, nw = 2, 2
    oracle = _per_shard_single(shards, batch_size=bs, num_workers=nw, sample_rate=SAMPLE_RATE)
    got = _persistent_single(shards, batch_size=bs, num_workers=nw, sample_rate=SAMPLE_RATE)
    _assert_single_equal(oracle, got)
    # Sanity: the oracle actually produced the expected sequential batching.
    assert [[p for p, *_ in s] for s in oracle][0] == [shards[0][0:2], shards[0][2:4], shards[0][4:5]]


def test_persistent_single_varied_batch_and_workers(tmp_path):
    paths = _make_wavs(tmp_path, 13)
    shards = _split_shards(paths, [4, 6, 3])
    for bs in (1, 3, 4):
        for nw in (1, 3):
            oracle = _per_shard_single(shards, batch_size=bs, num_workers=nw, sample_rate=SAMPLE_RATE)
            got = _persistent_single(shards, batch_size=bs, num_workers=nw, sample_rate=SAMPLE_RATE)
            _assert_single_equal(oracle, got)


def test_persistent_group_matches_per_shard_loader(tmp_path):
    paths = _make_wavs(tmp_path, 15)
    shards = _split_shards(paths, [5, 5, 5])
    rates = [SAMPLE_RATE, SAMPLE_RATE]  # default model mix is all 16k
    bs, nw = 4, 2
    oracle = _per_shard_group(shards, batch_size=bs, num_workers=nw, sample_rates=rates)
    got = _persistent_group(shards, batch_size=bs, num_workers=nw, sample_rates=rates)
    _assert_group_equal(oracle, got)


def test_persistent_group_multirate(tmp_path):
    paths = _make_wavs(tmp_path, 10)
    shards = _split_shards(paths, [4, 6])
    rates = [8000, SAMPLE_RATE]  # two distinct target rates
    oracle = _per_shard_group(shards, batch_size=3, num_workers=2, sample_rates=rates)
    got = _persistent_group(shards, batch_size=3, num_workers=2, sample_rates=rates)
    _assert_group_equal(oracle, got)


def test_persistent_loader_handles_empty_and_load_errors(tmp_path):
    good = _make_wavs(tmp_path, 4)
    missing = str(tmp_path / "dir0" / "does_not_exist.wav")
    shards = [good[:2], [], [missing] + good[2:]]  # empty shard + a decode error
    oracle = _per_shard_single(shards, batch_size=2, num_workers=2, sample_rate=SAMPLE_RATE)
    got = _persistent_single(shards, batch_size=2, num_workers=2, sample_rate=SAMPLE_RATE)
    _assert_single_equal(oracle, got)
    # The missing file surfaced as a collate-level load error, not a crash.
    flat_errs = [e for shard in got for (_, _, _, errs) in shard for e in errs]
    assert any(missing == p for p, _ in flat_errs)


# --------------------------------------------------------------------------- #
# Layer 2 — whole-stage equivalence with loader workers
# --------------------------------------------------------------------------- #

N_FILES = 12


def _make_tree(root: Path) -> None:
    for i in range(N_FILES):
        d = root / "pl0" / f"pod{i % 3}"
        d.mkdir(parents=True, exist_ok=True)
        rate = 8000 if i % 5 == 0 else SAMPLE_RATE
        dur = 0.3 + 0.1 * i
        t = np.arange(int(dur * rate)) / rate
        amp = (i + 1) / 100.0
        wave = (amp * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        sf.write(d / f"chunk_{i:02d}.wav", wave, rate)


def _write_config(path: Path, podcasts: Path, share_decode: bool,
                  persistent: bool, num_workers: int) -> None:
    config = {
        "runtime": {"audio_paths_source": "rglob", "work_shard_size": 5},
        "transcription": {
            "podcasts_path": str(podcasts),
            "model_names": ["alpha", "beta", "gamma"],
            "consensus_num": 2,
            "with_timestamps": True,
            "use_tensorrt": False,
            "use_vad": False,
            "use_rover": False,
            "share_decode": share_decode,
            "persistent_loaders": persistent,
            "batch_size": 4,
            "num_workers": num_workers,
            "prefetch_factor": 2,
            "retry_empty_outputs": True,
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


class _FakeASR:
    """Waveform-deterministic stand-in (mirrors the share_decode test model)."""

    def __init__(self, name): self.name, self.timestamped = name, False

    def with_timestamps(self):
        self.timestamped = True
        return self

    def _recognize_batch(self, waveforms, lengths):
        results = []
        for row, length in zip(waveforms, lengths):
            idx = int(round(float(np.abs(row[: int(length)]).max()) * 100)) - 1
            text = f"общий текст файла {idx}" if idx % 2 == 0 else f"{self.name} файл {idx}"
            if self.timestamped:
                tokens, stamps = [], []
                for w_i, word in enumerate(text.split()):
                    for ch in word:
                        tokens.append(ch)
                        stamps.append(0.1 * w_i)
                    tokens.append(" ")
                    stamps.append(0.1 * w_i + 0.05)
                from types import SimpleNamespace
                results.append(SimpleNamespace(text=text, tokens=tokens, timestamps=stamps))
            else:
                results.append(text)
        return results


_BATCH = {"alpha": 2, "beta": 3, "gamma": 4}


def _patch(monkeypatch):
    monkeypatch.setattr(tr.onnx_asr, "load_model", lambda name, *a, **k: _FakeASR(name))
    monkeypatch.setattr(tr, "get_onnx_providers", lambda *a, **k: ["CPUExecutionProvider"])
    monkeypatch.setattr(tr, "resolve_batch_size",
                        lambda key, configured, default: _BATCH[key.split(".", 1)[1]])
    monkeypatch.setattr(tr.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(tr.torch.cuda, "set_device", lambda *_: None)
    monkeypatch.setattr(tr, "SUPPORTED_TIMESTAMPS", {"beta", "gamma"})


def _run(tmp_path, tag, *, share_decode, persistent, num_workers, monkeypatch):
    root = tmp_path / tag
    _make_tree(root)
    cfg = tmp_path / f"cfg_{tag}.yaml"
    _write_config(cfg, root, share_decode, persistent, num_workers)
    _patch(monkeypatch)
    tr.main(argparse.Namespace(config_path=str(cfg), log_dir=str(tmp_path / "logs" / tag)))
    sidecars = {}
    for p in sorted(root.rglob("*")):
        if p.suffix in (".txt", ".tst"):
            sidecars[str(p.relative_to(root))] = p.read_text(encoding="utf-8")
    return sidecars


def test_stage_persistent_loaders_byte_identical_shared_decode(tmp_path, monkeypatch):
    ref = _run(tmp_path, "ref0", share_decode=True, persistent=False, num_workers=0, monkeypatch=monkeypatch)
    per_shard = _run(tmp_path, "perW", share_decode=True, persistent=False, num_workers=2, monkeypatch=monkeypatch)
    persistent = _run(tmp_path, "persW", share_decode=True, persistent=True, num_workers=2, monkeypatch=monkeypatch)
    assert ref, "reference produced no sidecars"
    assert per_shard == ref
    assert persistent == ref


def test_stage_persistent_loaders_byte_identical_sequential(tmp_path, monkeypatch):
    ref = _run(tmp_path, "seq0", share_decode=False, persistent=False, num_workers=0, monkeypatch=monkeypatch)
    persistent = _run(tmp_path, "seqP", share_decode=False, persistent=True, num_workers=2, monkeypatch=monkeypatch)
    assert ref, "reference produced no sidecars"
    assert persistent == ref
