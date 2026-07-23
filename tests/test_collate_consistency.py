"""Collate reads ASR consistency from chunk JSON, not from exact ASR matches."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from src import collate
from src.utils.chunk_json import update_chunk_json

MODELS = ["m0", "m1"]


def _audio(tmp_path: Path, stem: str) -> Path:
    path = tmp_path / f"{stem}.flac"
    path.write_bytes(b"x")
    return path


def test_sidecar_specs_include_rover_consistency_field():
    specs = collate.sidecar_specs(MODELS)
    assert specs[collate.ASR_CONSISTENCY_COLUMN] == "asr_consistency"


def test_old_exact_match_calculator_removed_from_collate_hot_path():
    assert not hasattr(collate, "add_asr_consistency_column")
    assert not hasattr(collate, "asr_consistency_percent")


def test_process_audio_file_reads_numeric_consistency(tmp_path):
    audio = _audio(tmp_path, "a")
    update_chunk_json(
        audio,
        {
            "rover": "a b c",
            "asr_consistency": 88.5,
            "asr": {"m0": "a b c", "m1": "a b x"},
        },
    )

    row = collate.process_audio_file(str(audio), tmp_path, collate.sidecar_specs(MODELS), {})

    assert row["rover"] == "a b c"
    assert row[collate.ASR_CONSISTENCY_COLUMN] == 88.5


def test_process_audio_file_missing_or_empty_consistency_is_nan(tmp_path):
    missing = _audio(tmp_path, "missing")
    empty = _audio(tmp_path, "empty")
    update_chunk_json(empty, {"asr_consistency": ""})

    specs = collate.sidecar_specs(MODELS)
    missing_row = collate.process_audio_file(str(missing), tmp_path, specs, {})
    empty_row = collate.process_audio_file(str(empty), tmp_path, specs, {})

    assert math.isnan(missing_row[collate.ASR_CONSISTENCY_COLUMN])
    assert math.isnan(empty_row[collate.ASR_CONSISTENCY_COLUMN])


def test_build_slab_preserves_json_consistency_even_when_asr_texts_match(tmp_path):
    audio = _audio(tmp_path, "slab")
    update_chunk_json(
        audio,
        {
            "asr_consistency": 42.0,
            "asr": {"m0": "identical", "m1": "identical"},
        },
    )
    df = pd.DataFrame({"filepath": [str(audio)]})
    specs = collate.sidecar_specs(MODELS)

    class InlineExecutor:
        def map(self, fn, items):
            return [fn(item) for item in items]

    slab, errors = collate.build_slab_frame(
        df, specs, MODELS, tmp_path, {}, 1, InlineExecutor()
    )

    assert errors == []
    assert slab[collate.ASR_CONSISTENCY_COLUMN].iloc[0] == 42.0

