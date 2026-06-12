"""Tests for benchmarking.warmup CLI flags and batch-size ladder logic.

No GPU required — tests cover argument parsing and the ladder-generation
function only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

# Make sure the repo root is on sys.path so the module can be imported.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.warmup import batch_size_ladder, denoising_probe_frames


# ---------------------------------------------------------------------------
# batch_size_ladder
# ---------------------------------------------------------------------------

def test_ladder_default_new_cap():
    """Default max_batch=512 → ladder reaches 512."""
    ladder = batch_size_ladder(512)
    assert ladder[-1] == 512
    assert ladder == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def test_ladder_stops_at_cap():
    ladder = batch_size_ladder(64)
    assert max(ladder) == 64
    assert ladder == [1, 2, 4, 8, 16, 32, 64]


def test_ladder_partial_cap():
    """A cap that is not a power-of-two stops the ladder just below."""
    ladder = batch_size_ladder(100)
    assert max(ladder) == 64


def test_ladder_cap_1():
    assert batch_size_ladder(1) == [1]


def test_ladder_cap_512():
    ladder = batch_size_ladder(512)
    assert 512 in ladder
    assert 1024 not in ladder


# ---------------------------------------------------------------------------
# denoising probe frame count (VRAM guard must see the runtime worst case)
# ---------------------------------------------------------------------------

def test_denoising_probe_frames_match_runtime_padded_len():
    """The denoising warmup must probe at the runtime worst case so the VRAM
    guard sees the activation memory the stage actually uses. The runtime
    stage pads every batch up to MODEL_MAX_PADDED_LEN
    (src/denoising/denoising.py); probing at the shorter --probe-seconds clip
    under-measures VRAM and lets batch_size:auto OOM on the first long shard."""
    from src.denoising.denoising import MODEL_MAX_PADDED_LEN

    args = argparse.Namespace(probe_seconds=10.0)
    assert denoising_probe_frames(args) == MODEL_MAX_PADDED_LEN


def test_denoising_probe_frames_ignores_short_probe_seconds():
    """Even a tiny --probe-seconds must still probe the full padded length."""
    from src.denoising.denoising import MODEL_MAX_PADDED_LEN

    args = argparse.Namespace(probe_seconds=1.0)
    assert denoising_probe_frames(args) == MODEL_MAX_PADDED_LEN


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse(argv: list[str]) -> argparse.Namespace:
    """Import the parser from warmup.main() without running the probe."""
    # We reconstruct the parser inline to avoid needing CUDA at import time.
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", type=str, default="configs/config.yaml")
    ap.add_argument("--models", type=str,
                    default="distillmos,antispoofing,denoising,transcription,punctuation,music_detect")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--max-batch", type=int, default=512)
    ap.add_argument("--probe-seconds", type=float, default=10.0)
    ap.add_argument("--audio-dir", type=str, default=None)
    ap.add_argument("--output", type=Path, default=Path("/tmp/node_profile.json"))
    return ap.parse_args(argv)


def test_max_batch_default_is_512():
    args = _parse([])
    assert args.max_batch == 512


def test_max_batch_override():
    args = _parse(["--max-batch", "1024"])
    assert args.max_batch == 1024


def test_models_default_contains_transcription():
    args = _parse([])
    wanted = {m.strip() for m in args.models.split(",") if m.strip()}
    assert "transcription" in wanted


def test_models_tone_only():
    args = _parse(["--models", "tone"])
    wanted = {m.strip() for m in args.models.split(",") if m.strip()}
    assert wanted == {"tone"}


def test_models_multiple():
    args = _parse(["--models", "tone,vosk"])
    wanted = {m.strip() for m in args.models.split(",") if m.strip()}
    assert wanted == {"tone", "vosk"}


def test_models_transcription_group():
    args = _parse(["--models", "transcription"])
    wanted = {m.strip() for m in args.models.split(",") if m.strip()}
    assert "transcription" in wanted
    assert "tone" not in wanted  # "tone" is inside transcription, not a group


# ---------------------------------------------------------------------------
# ASR model selection logic (mirrors main()'s set logic, no GPU needed)
# ---------------------------------------------------------------------------

def _asr_names_to_probe(models_str: str, all_asr_names: list[str]) -> list[str]:
    """Replicate the model-selection logic from warmup.main() for testing."""
    wanted = {m.strip() for m in models_str.split(",") if m.strip()}
    known_groups = {
        "distillmos", "antispoofing", "denoising",
        "transcription", "punctuation", "music_detect",
    }
    individual_asr_wanted = wanted - known_groups
    asr_name_filter = individual_asr_wanted & set(all_asr_names) if individual_asr_wanted else None

    probe_transcription = "transcription" in wanted or bool(asr_name_filter)
    if not probe_transcription:
        return []
    if "transcription" in wanted:
        return all_asr_names
    return [n for n in all_asr_names if n in (asr_name_filter or set())]


ALL_ASR = ["gigaam-v3-e2e-ctc", "giga_ctc", "giga_rnnt", "vosk", "tone"]


def test_transcription_group_probes_all():
    result = _asr_names_to_probe("transcription", ALL_ASR)
    assert result == ALL_ASR


def test_tone_only_selects_tone():
    result = _asr_names_to_probe("tone", ALL_ASR)
    assert result == ["tone"]


def test_vosk_only_selects_vosk():
    result = _asr_names_to_probe("vosk", ALL_ASR)
    assert result == ["vosk"]


def test_two_asr_models():
    result = _asr_names_to_probe("tone,vosk", ALL_ASR)
    # Order follows config order
    assert set(result) == {"tone", "vosk"}
    assert len(result) == 2


def test_unknown_asr_name_produces_empty():
    result = _asr_names_to_probe("nonexistent_model", ALL_ASR)
    assert result == []


def test_distillmos_only_no_asr():
    result = _asr_names_to_probe("distillmos", ALL_ASR)
    assert result == []


def test_transcription_and_tone_deduped():
    """'transcription' + individual name → all models (transcription wins)."""
    result = _asr_names_to_probe("transcription,tone", ALL_ASR)
    assert result == ALL_ASR
