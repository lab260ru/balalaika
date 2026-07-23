"""Tests for src.utils.node_profile.resolve_batch_size."""
from __future__ import annotations

import json

import pytest

from src.utils import node_profile
from src.utils.node_profile import resolve_batch_size


@pytest.fixture(autouse=True)
def clear_cache():
    node_profile._profile_cache.clear()
    yield
    node_profile._profile_cache.clear()


@pytest.fixture()
def profile(tmp_path):
    path = tmp_path / "node_profile.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "distillmos": {"best_batch_size": 32},
                    "transcription.giga_ctc": {"best_batch_size": 24},
                    "antispoofing": {"best_batch_size": None, "skipped": "x"},
                },
                "recommended_batch_sizes": {"transcription": 8},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_plain_int_passthrough(profile):
    assert resolve_batch_size("distillmos", 4, 16, profile_path=profile) == 4


def test_numeric_string(profile):
    assert resolve_batch_size("distillmos", "12", 16, profile_path=profile) == 12


def test_none_uses_default(profile):
    assert resolve_batch_size("distillmos", None, 16, profile_path=profile) == 16


def test_auto_resolves_from_profile(profile):
    assert resolve_batch_size("distillmos", "auto", 16, profile_path=profile) == 32


def test_auto_model_specific_transcription(profile):
    assert (
        resolve_batch_size("transcription.giga_ctc", "auto", 16, profile_path=profile)
        == 24
    )


def test_auto_falls_back_to_flat_recommendation(profile):
    assert (
        resolve_batch_size("transcription.vosk", "AUTO", 16, profile_path=profile) == 8
    )


def test_auto_skipped_model_uses_default(profile):
    assert resolve_batch_size("antispoofing", "auto", 8, profile_path=profile) == 8


def test_auto_without_profile_uses_default(tmp_path):
    missing = tmp_path / "missing.json"
    assert resolve_batch_size("distillmos", "auto", 16, profile_path=missing) == 16


def test_garbage_value_uses_default(profile):
    assert resolve_batch_size("distillmos", "huge", 16, profile_path=profile) == 16
