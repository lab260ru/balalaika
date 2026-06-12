"""Equivalence tests: vectorised ASR consistency vs the row-wise reference."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.collate import (
    ASR_CONSISTENCY_COLUMN,
    add_asr_consistency_column,
    asr_consistency_percent,
)

MODELS = ["giga_ctc", "giga_rnnt", "vosk", "tone"]


def reference(df: pd.DataFrame, asr_columns: list[str]) -> pd.Series:
    return df.apply(asr_consistency_percent, axis=1, asr_columns=asr_columns)


def assert_matches_reference(df: pd.DataFrame):
    expected = reference(df.copy(), MODELS)
    got = add_asr_consistency_column(df.copy(), MODELS)[ASR_CONSISTENCY_COLUMN]
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), equal_nan=True)


def test_edge_cases():
    df = pd.DataFrame(
        {
            "giga_ctc": ["привет мир", "a", np.nan, "", "x", "  Привет   МИР ", None],
            "giga_rnnt": ["привет мир", "b", np.nan, "", "x", "привет мир", ""],
            "vosk": ["привет мир", "c", np.nan, "", "x", "другое", np.nan],
            "tone": ["другой текст", np.nan, np.nan, "", "x", "привет мир", ""],
        }
    )
    assert_matches_reference(df)


def test_single_nonempty_is_nan():
    df = pd.DataFrame(
        {
            "giga_ctc": ["только один"],
            "giga_rnnt": [""],
            "vosk": [np.nan],
            "tone": [None],
        }
    )
    out = add_asr_consistency_column(df, MODELS)
    assert np.isnan(out[ASR_CONSISTENCY_COLUMN].iloc[0])


def test_full_agreement_is_100():
    df = pd.DataFrame({m: ["одно и то же"] for m in MODELS})
    out = add_asr_consistency_column(df, MODELS)
    assert out[ASR_CONSISTENCY_COLUMN].iloc[0] == pytest.approx(100.0)


def test_numeric_cells_match_reference():
    # all-NaN float columns / stray numeric cells go through str() in both paths
    df = pd.DataFrame(
        {
            "giga_ctc": [1.5, np.nan],
            "giga_rnnt": ["1.5", "y"],
            "vosk": [np.nan, "y"],
            "tone": ["1.5", np.nan],
        }
    )
    assert_matches_reference(df)


def test_fewer_than_two_columns():
    df = pd.DataFrame({"giga_ctc": ["a"], "other": ["b"]})
    out = add_asr_consistency_column(df, ["giga_ctc"])
    assert np.isnan(out[ASR_CONSISTENCY_COLUMN].iloc[0])


def test_vosk_suffix_dedup():
    # vosk and vosk_small share the 'vosk' sidecar suffix -> counted once
    df = pd.DataFrame(
        {
            "vosk": ["a", "b"],
            "giga_ctc": ["a", "b"],
        }
    )
    out = add_asr_consistency_column(df, ["vosk", "vosk_small", "giga_ctc"])
    assert out[ASR_CONSISTENCY_COLUMN].tolist() == [100.0, 100.0]


def test_random_matches_reference():
    rng = np.random.default_rng(7)
    phrases = ["раз два", "три", "", "четыре пять шесть", "раз  ДВА "]
    data = {}
    for m in MODELS:
        vals = [phrases[i] for i in rng.integers(0, len(phrases), size=500)]
        s = pd.Series(vals).mask(rng.random(500) < 0.15)
        data[m] = s
    df = pd.DataFrame(data)
    assert_matches_reference(df)
