"""Equivalence: vectorised metadata sanitisation vs. the old per-cell worker
loop. JSON bytes must be identical (None vs NaN, int unboxing, Timestamp str
format, numpy scalar unboxing, bool).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.to_webdataset import _sanitize_records


def _old_sanitize(record: dict) -> dict:
    """Verbatim copy of the old worker per-cell branch."""
    out = {}
    for k, v in record.items():
        k_str = str(k)
        if pd.isna(v) or (isinstance(v, float) and math.isnan(v)):
            out[k_str] = None
        elif isinstance(v, (pd.Timestamp, pd.Timedelta)):
            out[k_str] = str(v)
        elif hasattr(v, "item"):
            out[k_str] = v.item()
        else:
            out[k_str] = v
    return out


def _assert_records_match(df: pd.DataFrame):
    old_records = [_old_sanitize(r) for r in df.to_dict("records")]
    new_records = _sanitize_records(df)
    assert len(old_records) == len(new_records)
    for old, new in zip(old_records, new_records):
        # Same keys, same order
        assert list(old.keys()) == list(new.keys())
        # Same JSON bytes (None/NaN, int/float, str format all pinned here)
        assert json.dumps(old, ensure_ascii=False) == json.dumps(
            new, ensure_ascii=False
        )


def test_numeric_and_nan():
    df = pd.DataFrame(
        {
            "i": pd.array([1, 2, 3], dtype="int64"),
            "f": [2.5, np.nan, 0.0],
            "allnan": [np.nan, np.nan, np.nan],
        }
    )
    _assert_records_match(df)


def test_strings_and_bools():
    df = pd.DataFrame(
        {
            "s": ["hello", "мир", ""],
            "b": [True, False, True],
        }
    )
    _assert_records_match(df)


def test_object_mixed_column():
    # Object column with numpy scalars + python + NaN mixed
    df = pd.DataFrame(
        {
            "m": pd.Series([np.int64(5), "txt", np.float64(1.5), np.nan], dtype=object),
        }
    )
    _assert_records_match(df)


def test_timestamp_and_timedelta():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2021-01-01", "2021-02-02", pd.NaT]),
            "td": pd.to_timedelta(["1 days", pd.NaT, "3 days"]),
        }
    )
    _assert_records_match(df)


def test_from_fast_read_csv(tmp_path):
    """End-to-end through load_metadata's reader path (native python scalars)."""
    from src.utils.csv_manager import fast_read_csv

    p = tmp_path / "m.csv"
    p.write_text(
        "filepath,a_int,b_float,c_str,e_allnan\n"
        "/x/1.wav,5,2.5,hello,\n"
        "/x/2.wav,7,,world,\n",
        encoding="utf-8",
    )
    df = fast_read_csv(p).drop(columns=["filepath"])
    _assert_records_match(df)
