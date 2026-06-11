"""share_decode=True must produce byte-identical sidecars to the sequential
per-model flow, while decoding each audio file once for the grouped models.

Fake ASR models stand in for onnx-asr (deterministic text derived from the
waveform itself), so the test pins the orchestration: pending-set union,
annotated shards, macro-batch decode, per-model sub-batching, timestamp
formatting, consensus skipping for tail models, and resume behavior.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

import src.transcription.transcription as tr

SAMPLE_RATE = 16_000
N_FILES = 12


# --- fixture tree -----------------------------------------------------------

def make_tree(root: Path) -> list[Path]:
    paths = []
    for i in range(N_FILES):
        d = root / "pl0" / f"pod{i % 3}"
        d.mkdir(parents=True, exist_ok=True)
        rate = 8000 if i % 5 == 0 else SAMPLE_RATE  # exercise resampling
        dur = 0.3 + 0.1 * i
        t = np.arange(int(dur * rate)) / rate
        # File identity is encoded in the amplitude so a model that only
        # sees the waveform can answer deterministically per file.
        amp = (i + 1) / 100.0
        wave = (amp * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        p = d / f"chunk_{i:02d}.wav"
        sf.write(p, wave, rate)
        paths.append(p)
    return paths


def write_config(path: Path, podcasts_path: Path, share_decode: bool) -> None:
    config = {
        "runtime": {
            "audio_paths_source": "rglob",
            "work_shard_size": 5,
        },
        "transcription": {
            "podcasts_path": str(podcasts_path),
            "model_names": ["alpha", "beta", "gamma"],
            "consensus_num": 2,
            "with_timestamps": True,
            "use_tensorrt": False,
            "use_vad": False,
            "use_rover": False,
            "share_decode": share_decode,
            "batch_size": 4,
            "num_workers": 0,
            "prefetch_factor": 2,
            "retry_empty_outputs": True,
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


# --- fake onnx-asr ----------------------------------------------------------

def file_index_from_waveform(waveform: np.ndarray) -> int:
    return int(round(float(np.abs(waveform).max()) * 100)) - 1


class FakeASR:
    """Waveform-deterministic stand-in for an onnx-asr model."""

    def __init__(self, name: str):
        self.name = name
        self.timestamped = False

    def with_timestamps(self):
        self.timestamped = True
        return self

    def _recognize_batch(self, waveforms: np.ndarray, lengths: np.ndarray):
        results = []
        for row, length in zip(waveforms, lengths):
            idx = file_index_from_waveform(row[: int(length)])
            if idx % 2 == 0:
                text = f"общий текст файла {idx}"  # alpha and beta agree
            else:
                text = f"{self.name} слышит файл {idx}"
            if self.timestamped:
                tokens, stamps = [], []
                for w_i, word in enumerate(text.split()):
                    for ch in word:
                        tokens.append(ch)
                        stamps.append(0.1 * w_i)
                    tokens.append(" ")
                    stamps.append(0.1 * w_i + 0.05)
                results.append(
                    SimpleNamespace(text=text, tokens=tokens, timestamps=stamps)
                )
            else:
                results.append(text)
        return results


BATCH_SIZES = {"alpha": 2, "beta": 3, "gamma": 4}


@pytest.fixture()
def patched(monkeypatch):
    import src.utils.datasets.transcription as ds

    decode_counter = {"n": 0}
    real_load = ds.torchaudio.load_with_torchcodec

    def counting_load(path):
        decode_counter["n"] += 1
        return real_load(path)

    monkeypatch.setattr(
        ds.torchaudio, "load_with_torchcodec", counting_load
    )
    monkeypatch.setattr(tr.onnx_asr, "load_model", lambda name, *a, **k: FakeASR(name))
    monkeypatch.setattr(tr, "get_onnx_providers", lambda *a, **k: ["CPUExecutionProvider"])
    monkeypatch.setattr(
        tr, "resolve_batch_size", lambda key, configured, default: BATCH_SIZES[key.split(".", 1)[1]]
    )
    monkeypatch.setattr(tr.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(tr.torch.cuda, "set_device", lambda *_: None)
    # SUPPORTED_TIMESTAMPS so one grouped model and one tail model emit .tst
    monkeypatch.setattr(tr, "SUPPORTED_TIMESTAMPS", {"beta", "gamma"})
    return decode_counter


def run_stage(tmp_path: Path, share_decode: bool, patched) -> tuple[dict, int]:
    tag = "shared" if share_decode else "sequential"
    root = tmp_path / tag
    make_tree(root)
    config_path = tmp_path / f"config_{tag}.yaml"
    write_config(config_path, root, share_decode)

    patched["n"] = 0
    tr.main(argparse.Namespace(config_path=str(config_path), log_dir=str(tmp_path / "logs")))
    decodes = patched["n"]

    sidecars = {}
    for p in sorted(root.rglob("*")):
        if p.suffix in (".txt", ".tst"):
            sidecars[str(p.relative_to(root))] = p.read_text(encoding="utf-8")
    return sidecars, decodes


def test_shared_decode_outputs_identical_and_decode_count_reduced(tmp_path, patched):
    seq_sidecars, seq_decodes = run_stage(tmp_path, share_decode=False, patched=patched)
    shared_sidecars, shared_decodes = run_stage(tmp_path, share_decode=True, patched=patched)

    assert seq_sidecars, "sequential run produced no sidecars"
    assert shared_sidecars == seq_sidecars

    # Sequential: alpha + beta decode everything; gamma decodes only files
    # without alpha/beta consensus (odd indices). Shared: one decode for the
    # alpha+beta group, gamma unchanged.
    odd_files = len([i for i in range(N_FILES) if i % 2 == 1])
    assert seq_decodes == 2 * N_FILES + odd_files
    assert shared_decodes == N_FILES + odd_files


def test_resume_only_runs_missing_models(tmp_path, patched):
    root = tmp_path / "resume"
    make_tree(root)
    config_path = tmp_path / "config_resume.yaml"
    write_config(config_path, root, share_decode=True)

    # Pre-create alpha sidecars for the first half of the files: the group
    # pass must still decode those files for beta, and alpha must not
    # rewrite them.
    audio = sorted(root.rglob("*.wav"))
    pre = {}
    for p in audio[: N_FILES // 2]:
        sidecar = p.with_name(f"{p.stem}_alpha.txt")
        sidecar.write_text("уже готово", encoding="utf-8")
        pre[sidecar] = "уже готово"

    tr.main(argparse.Namespace(config_path=str(config_path), log_dir=str(tmp_path / "logs")))

    for sidecar, content in pre.items():
        assert sidecar.read_text(encoding="utf-8") == content, "resume overwrote a completed sidecar"
    for p in audio:
        assert p.with_name(f"{p.stem}_alpha.txt").exists()
        assert p.with_name(f"{p.stem}_beta.txt").exists()


def test_annotated_shard_roundtrip(tmp_path):
    from src.utils.work_shards import (
        prepare_length_bucketed_work_shards,
        claim_work_shard,
        read_annotated_work_shard,
    )

    paths = [str(tmp_path / f"f{i}.wav") for i in range(4)]
    durations = {p: 1.0 + i for i, p in enumerate(paths)}
    notes = {paths[0]: "alpha,beta", paths[1]: "alpha", paths[2]: "", paths[3]: "beta"}
    plan = prepare_length_bucketed_work_shards(
        tmp_path, "test_group", paths, durations, shard_size=10, annotations=notes
    )
    items: dict[str, str] = {}
    while (shard := claim_work_shard(plan.work_dir, 0)) is not None:
        items.update(read_annotated_work_shard(shard))
    assert items == {paths[0]: "alpha,beta", paths[1]: "alpha", paths[2]: "", paths[3]: "beta"}


def test_plain_shards_with_annotations_roundtrip(tmp_path):
    """prepare_work_shards (non-bucketed, used by music_detect) carries
    annotations and float durations round-trip exactly via str()."""
    from src.utils.work_shards import (
        prepare_work_shards,
        claim_work_shard,
        read_annotated_work_shard,
        read_work_shard,
    )

    paths = [str(tmp_path / f"f{i}.wav") for i in range(7)]
    durations = {p: 0.1 + 1.7 * i for i, p in enumerate(paths)}
    notes = {p: str(float(d)) for p, d in durations.items()}
    plan = prepare_work_shards(tmp_path, "md_test", paths, shard_size=3, annotations=notes)
    assert plan.total_items == 7

    items: dict[str, str] = {}
    while (shard := claim_work_shard(plan.work_dir, 0)) is not None:
        items.update(read_annotated_work_shard(shard))
    assert set(items) == set(paths)
    for p in paths:
        assert float(items[p]) == durations[p]  # exact: str(float) round-trips

    # plain shards without annotations stay line-per-path (old readers fine)
    plan2 = prepare_work_shards(tmp_path, "md_test2", paths, shard_size=10)
    shard2 = claim_work_shard(plan2.work_dir, 0)
    assert read_work_shard(shard2) == paths
