"""Each audio file's bytes must leave the disk exactly once per preprocess stage.

Two cold-cache HDD double-reads were removed:

1. ``preprocess_existing_chunks._postprocess_existing_chunk`` re-decoded the
   chunk at native rate after the DataLoader already decoded it at 16 kHz.
2. ``preprocess.process_audio_file`` re-opened short single-chunk sources with a
   fresh ``AudioDecoder`` after the DataLoader already decoded them.

The fix reads each file's raw bytes once in ``DiarizationDataset`` and feeds the
*same* bytes to both the 16 kHz decode and the native-rate decode. torchcodec
decodes a ``bytes`` source bit-identically to a path source, so this test pins:

* bit-identical native-rate decode (bytes source vs path source);
* bit-identical ``_postprocess_existing_chunk`` / fused single-chunk outputs vs
  the original double-decode reference logic (kept inline below);
* the disk is opened exactly once per file in the new path (instrumented count),
  versus twice in the reference double-decode path.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from src.preprocess import preprocess_existing_chunks as pec
from src.preprocess.audio_postprocessing import postprocess_audio_tensor
from src.utils.datasets.preprocess import (
    DIARIZATION_SAMPLE_RATE,
    DiarizationDataset,
)

# A crest threshold high enough that synthetic tones/noise are always kept and
# loudness-normalized, so the postprocess + save path is fully exercised.
CONFIG = {
    "crest_threshold": 1000.0,
    "peak": -1.0,
    "loudness": -23.0,
    "block_size": 0.400,
    "fuse_audio_preprocessing": True,
}


def _write_fixture(path: Path, rate: int, channels: int, seconds: float, fmt: str) -> None:
    n = int(rate * seconds)
    t = torch.linspace(0.0, seconds, n)
    base = 0.3 * torch.sin(2 * np.pi * 220.0 * t)
    if channels == 1:
        wav = base.unsqueeze(0)
    else:
        wav = torch.stack([base, 0.2 * torch.sin(2 * np.pi * 330.0 * t)])
    AudioEncoder(wav.contiguous(), sample_rate=rate).to_file(str(path))
    _ = fmt


FIXTURES = [
    (16_000, 1, "wav"),
    (16_000, 2, "wav"),
    (48_000, 1, "flac"),
    (48_000, 2, "flac"),
]


@pytest.fixture(params=FIXTURES, ids=lambda p: f"{p[0]}Hz_{p[1]}ch_{p[2]}")
def chunk_file(tmp_path, request):
    rate, channels, fmt = request.param
    path = tmp_path / f"0.00_3.00_album_episode.{fmt}"
    _write_fixture(path, rate, channels, 3.0, fmt)
    return path


# --------------------------------------------------------------------------- #
# Reference logic (the ORIGINAL double-decode path), kept inline for comparison.
# --------------------------------------------------------------------------- #
def _reference_postprocess_double_decode(path_audio: str, config) -> dict:
    """Original ``_postprocess_existing_chunk`` body: decode from the path."""
    native_audio, native_sr = torchaudio.load_with_torchcodec(path_audio)
    native_audio = native_audio.to(dtype=torch.float32).contiguous()
    result = postprocess_audio_tensor(
        native_audio,
        int(native_sr),
        crest_threshold=float(config["crest_threshold"]),
        peak=float(config["peak"]),
        loudness=float(config["loudness"]),
        block_size=float(config["block_size"]),
    )
    return {
        "keep": result.keep,
        "crest_factor": result.crest_factor,
        "normalized": result.loudness_normalized,
        "samples": result.samples,
        "native_sr": int(native_sr),
    }


def test_bytes_source_native_decode_is_bit_identical(chunk_file):
    """AudioDecoder / load_with_torchcodec from bytes == from path, exactly."""
    raw = chunk_file.read_bytes()

    a_path, sr_path = torchaudio.load_with_torchcodec(str(chunk_file))
    a_bytes, sr_bytes = torchaudio.load_with_torchcodec(io.BytesIO(raw))
    assert sr_path == sr_bytes
    assert torch.equal(a_path, a_bytes)

    d_path = AudioDecoder(str(chunk_file)).get_all_samples().data
    d_bytes = AudioDecoder(raw).get_all_samples().data
    assert torch.equal(d_path, d_bytes)


def test_dataset_16k_decode_unchanged_and_ships_bytes(chunk_file):
    """Byte-reuse mode yields the same 16 kHz waveform and carries the bytes."""
    ds_legacy = DiarizationDataset([str(chunk_file)], raw_bytes_max_duration_s=None)
    ds_reuse = DiarizationDataset([str(chunk_file)], raw_bytes_max_duration_s=60.0)

    path0, wav0, sr0, err0, bytes0 = ds_legacy[0]
    path1, wav1, sr1, err1, bytes1 = ds_reuse[0]

    assert err0 == "" and err1 == ""
    assert sr0 == sr1 == DIARIZATION_SAMPLE_RATE
    # 16 kHz decode identical whether or not we also ship bytes.
    assert torch.equal(wav0, wav1)
    # Legacy mode ships no bytes; reuse mode ships the file's exact bytes.
    assert bytes0 is None
    assert bytes1 == chunk_file.read_bytes()


def test_dataset_bound_drops_bytes_for_long_files(chunk_file):
    """A cap below the clip length means no bytes are carried (RAM guard)."""
    ds = DiarizationDataset([str(chunk_file)], raw_bytes_max_duration_s=0.5)
    _, _, _, err, raw_bytes = ds[0]
    assert err == ""
    assert raw_bytes is None  # 3 s clip exceeds the 0.5 s cap


def test_postprocess_existing_chunk_bytes_equals_double_decode(chunk_file, tmp_path):
    """New byte-reuse postprocess == original double-decode, bit for bit."""
    # Reference (old behavior): decode from path inside postprocess.
    ref = _reference_postprocess_double_decode(str(chunk_file), CONFIG)

    # New behavior: bytes the loader read once, fed to the native decode.
    raw = chunk_file.read_bytes()

    # Compare the decoded native tensor + postprocess output directly.
    new_native, new_sr = torchaudio.load_with_torchcodec(io.BytesIO(raw))
    new_native = new_native.to(dtype=torch.float32).contiguous()
    assert new_sr == ref["native_sr"]
    new_result = postprocess_audio_tensor(
        new_native,
        int(new_sr),
        crest_threshold=float(CONFIG["crest_threshold"]),
        peak=float(CONFIG["peak"]),
        loudness=float(CONFIG["loudness"]),
        block_size=float(CONFIG["block_size"]),
    )
    assert new_result.keep == ref["keep"]
    assert new_result.crest_factor == ref["crest_factor"]
    assert new_result.loudness_normalized == ref["normalized"]
    assert torch.equal(new_result.samples, ref["samples"])

    # And the public helper agrees too (returns keep/crest/normalized/dur/err).
    work = tmp_path / "work.flac"
    work.write_bytes(raw)
    keep_b, crest_b, norm_b, dur_b, err_b = pec._postprocess_existing_chunk(
        str(work), CONFIG, raw
    )
    work.write_bytes(raw)  # restore in case the helper rewrote it
    keep_p, crest_p, norm_p, dur_p, err_p = pec._postprocess_existing_chunk(
        str(work), CONFIG, None
    )
    assert (keep_b, round(crest_b, 6), norm_b, round(dur_b, 6), err_b) == (
        keep_p,
        round(crest_p, 6),
        norm_p,
        round(dur_p, 6),
        err_p,
    )


def test_written_file_bytes_identical(chunk_file, tmp_path):
    """The rewritten chunk file is byte-for-byte identical bytes-source vs path."""
    raw = chunk_file.read_bytes()

    target_bytes = tmp_path / "out_bytes.flac"
    target_path = tmp_path / "out_path.flac"
    target_bytes.write_bytes(raw)
    target_path.write_bytes(raw)

    pec._postprocess_existing_chunk(str(target_bytes), CONFIG, raw)
    pec._postprocess_existing_chunk(str(target_path), CONFIG, None)

    assert target_bytes.read_bytes() == target_path.read_bytes()


# --------------------------------------------------------------------------- #
# Disk-open accounting via strace: prove 2 READ opens/file -> 1 READ open/file.
#
# torchcodec/FFmpeg opens the path in C++ (not through Python's builtins.open),
# so a monkeypatched counter cannot see those reads. We count the real
# ``openat`` syscalls with strace instead, filtered to the fixture path, and
# split read-only opens (O_RDONLY) from the in-place rewrite (O_WRONLY|O_CREAT).
# --------------------------------------------------------------------------- #
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

_PROBE = str(Path(__file__).with_name("_open_count_probe.py"))


def _count_read_opens(audio_path: Path, *, reuse: bool) -> int:
    """Run the probe under strace; return # of O_RDONLY opens of ``audio_path``."""
    strace_out = audio_path.with_suffix(audio_path.suffix + ".strace")
    env = dict(__import__("os").environ)
    # The probe imports ``src.*`` — make the worktree root importable.
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    proc = subprocess.run(
        [
            "strace", "-f", "-e", "trace=openat", "-o", str(strace_out),
            sys.executable, _PROBE, str(audio_path), "1" if reuse else "0",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, f"probe failed: {proc.stderr}"
    assert "PROBE_MARKER_END" in proc.stderr, "probe did not reach the measured region"

    target = str(audio_path.resolve())
    read_opens = 0
    for line in strace_out.read_text().splitlines():
        if target not in line or "openat(" not in line:
            continue
        # Count only successful read-only opens; ignore the O_WRONLY rewrite and
        # any failed (ENOENT) probes.
        if "O_RDONLY" in line and "= -1" not in line:
            read_opens += 1
    return read_opens


@pytest.mark.skipif(shutil.which("strace") is None, reason="strace not available")
def test_open_count_halves(chunk_file):
    """Original path: 2 read opens/file. Byte-reuse path: 1 read open/file."""
    # Work on copies so the in-place rewrite never mutates the fixture between
    # the two runs (loudness normalization rewrites the file in place).
    legacy_copy = chunk_file.with_name("legacy_" + chunk_file.name)
    reuse_copy = chunk_file.with_name("reuse_" + chunk_file.name)
    legacy_copy.write_bytes(chunk_file.read_bytes())
    reuse_copy.write_bytes(chunk_file.read_bytes())

    opens_legacy = _count_read_opens(legacy_copy, reuse=False)
    opens_reuse = _count_read_opens(reuse_copy, reuse=True)

    assert opens_legacy == 2, f"expected 2 read opens in legacy path, got {opens_legacy}"
    assert opens_reuse == 1, f"expected 1 read open in byte-reuse path, got {opens_reuse}"
