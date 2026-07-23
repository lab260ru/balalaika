"""Byte-identity + atomicity test for save_audio_atomic (stage 1/3 write path).

The loudness/crest stages re-encode the source file in place. save_audio_atomic
must produce bytes identical to a direct torchaudio.save_with_torchcodec (the
§5 byte-identical bar) while never leaving a truncated source on a crash.

Run: .dev_venv/bin/python -m pytest tests/test_loudness_atomic_write.py -q
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch

from src.preprocess.audio_postprocessing import save_audio_atomic


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _make_tensor(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(1, 24_000, generator=g) * 2 - 1).to(torch.float32)


def test_atomic_write_is_byte_identical(tmp_path):
    import torchaudio

    tensor = _make_tensor()
    sr = 48_000

    direct = tmp_path / "direct.wav"
    torchaudio.save_with_torchcodec(str(direct), tensor, sr)

    atomic = tmp_path / "atomic.wav"
    save_audio_atomic(str(atomic), tensor, sr)

    assert _md5(direct) == _md5(atomic), "atomic write bytes differ from direct save"


def test_atomic_write_overwrites_in_place_identically(tmp_path):
    import torchaudio

    sr = 48_000
    path = tmp_path / "chunk.wav"
    # Pre-existing source file.
    torchaudio.save_with_torchcodec(str(path), _make_tensor(1), sr)

    new_tensor = _make_tensor(2)
    save_audio_atomic(str(path), new_tensor, sr)

    reference = tmp_path / "reference.wav"
    torchaudio.save_with_torchcodec(str(reference), new_tensor, sr)
    assert _md5(path) == _md5(reference)


def test_no_temp_files_left_behind(tmp_path):
    save_audio_atomic(str(tmp_path / "out.wav"), _make_tensor(), 48_000)
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert leftovers == [], f"leftover temp files: {leftovers}"


def test_failed_encode_leaves_source_intact_and_no_temp(tmp_path, monkeypatch):
    import torchaudio

    sr = 48_000
    path = tmp_path / "chunk.wav"
    torchaudio.save_with_torchcodec(str(path), _make_tensor(1), sr)
    original = _md5(path)

    def boom(tmp, *args, **kwargs):
        # Simulate a crash AFTER partially writing the temp file.
        Path(tmp).write_bytes(b"partial-garbage")
        raise RuntimeError("encode died")

    # save_audio_atomic does a local `import torchaudio`, which resolves to this
    # same module object, so patching it here takes effect.
    monkeypatch.setattr(torchaudio, "save_with_torchcodec", boom)

    with pytest.raises(RuntimeError):
        save_audio_atomic(str(path), _make_tensor(2), sr)

    # Source file is untouched (atomic replace never happened).
    assert _md5(path) == original
    # No temp left behind.
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert leftovers == []
