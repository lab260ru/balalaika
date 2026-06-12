"""State-aware late-stage reads (parquet mode).

In parquet pipeline-state mode (``csv.state_format: parquet``) the hot state
file is ``balalaika.parquet``; the CSV export is only refreshed at stage
completion (``absorb_partial_csvs``). Direct upserters — e.g. stage-7's
``ensure_audio_durations`` -> ``upsert_columns`` — never absorb, so the CSV
export goes **stale**. The late-stage consumers (``to_webdataset.load_metadata``
and ``collate``'s main-frame read) must therefore read the active *state* file,
not the hardcoded CSV.

This module pins:

* ``read_state_dataframe`` mirrors ``report.py``'s precedence: parquet state in
  parquet mode (when present), else the CSV; csv mode is unchanged.
* The parquet-loaded frame is normalized to CSV-equivalent dtypes (nullable
  Int64-with-nulls -> float64, nullable boolean -> object/None) so consumers
  see the same dtypes they always did.
* ``to_webdataset.load_metadata`` and ``collate``'s read both pick up the fresh
  parquet state even when ``balalaika.csv`` on disk is stale.

Run: .dev_venv/bin/python -m pytest tests/test_state_aware_reads.py -q
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils import csv_manager as cm


@contextmanager
def state_mode(mode: str):
    """Temporarily set BALALAIKA_STATE_FORMAT for the duration of the block."""
    prev = os.environ.get("BALALAIKA_STATE_FORMAT")
    if mode == "csv":
        os.environ.pop("BALALAIKA_STATE_FORMAT", None)
    else:
        os.environ["BALALAIKA_STATE_FORMAT"] = mode
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("BALALAIKA_STATE_FORMAT", None)
        else:
            os.environ["BALALAIKA_STATE_FORMAT"] = prev


def _write_fixture_csv(root: Path) -> None:
    pd.DataFrame(
        {
            "filepath": ["/d/a.wav", "/d/b.wav", "/d/c.wav"],
            "speaker_id": ["spk_0", "spk_1", "spk_2"],
            "crest_factor": [15.7, np.nan, 3.5],
        }
    ).to_csv(root / "balalaika.csv", index=False)


class TestReadStateDataframe:
    def test_parquet_mode_returns_fresh_state_when_csv_stale(self, tmp_path):
        """upsert WITHOUT absorb: loader sees the new column though CSV is stale."""
        _write_fixture_csv(tmp_path)
        with state_mode("parquet"):
            # migrate legacy CSV -> parquet, then a direct upsert (no absorb).
            cm.ensure_main_csv(tmp_path)
            cm.upsert_columns(
                tmp_path,
                pd.DataFrame(
                    {"filepath": ["/d/a.wav", "/d/b.wav", "/d/c.wav"],
                     "total_duration": [10.0, 11.0, 12.0]}
                ),
                ["total_duration"],
            )
            # The raw CSV export on disk has NOT been refreshed -> stale.
            stale_csv = cm.fast_read_csv(tmp_path / "balalaika.csv")
            assert "total_duration" not in stale_csv.columns

            df = cm.read_state_dataframe(tmp_path)
        assert "total_duration" in df.columns
        assert sorted(df["total_duration"].tolist()) == [10.0, 11.0, 12.0]

    def test_csv_mode_returns_csv_unchanged(self, tmp_path):
        """csv mode: loader returns the CSV (zero behavior change)."""
        _write_fixture_csv(tmp_path)
        with state_mode("csv"):
            df = cm.read_state_dataframe(tmp_path)
        expected = cm.fast_read_csv(tmp_path / "balalaika.csv")
        pd.testing.assert_frame_equal(
            df.reset_index(drop=True), expected.reset_index(drop=True)
        )

    def test_parquet_mode_falls_back_to_csv_when_no_parquet(self, tmp_path):
        """Parquet mode but no parquet on disk yet -> read the CSV export."""
        _write_fixture_csv(tmp_path)
        assert not (tmp_path / "balalaika.parquet").exists()
        with state_mode("parquet"):
            df = cm.read_state_dataframe(tmp_path)
        assert len(df) == 3
        assert "crest_factor" in df.columns

    def test_parquet_frame_normalized_to_csv_dtypes(self, tmp_path):
        """Nullable Int64/boolean from parquet -> float64 / object(None), like CSV."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        frame = pd.DataFrame(
            {
                "filepath": ["/d/a.wav", "/d/b.wav", "/d/c.wav"],
                "podcast_id": pd.array([10, pd.NA, 30], dtype="Int64"),
                "is_single_speaker": pd.array([True, pd.NA, False], dtype="boolean"),
                "crest_factor": [1.5, np.nan, 3.0],
            }
        )
        pq.write_table(
            pa.Table.from_pandas(frame, preserve_index=False),
            str(tmp_path / "balalaika.parquet"),
        )
        with state_mode("parquet"):
            df = cm.read_state_dataframe(tmp_path)
        # CSV-equivalent dtypes: int-with-nulls -> float64, bool -> object/None.
        assert df["podcast_id"].dtype == np.float64
        assert df["podcast_id"].tolist()[0] == 10.0
        assert pd.isna(df["podcast_id"].tolist()[1])
        assert df["is_single_speaker"].dtype == object
        assert df["is_single_speaker"].tolist() == [True, None, False]


class TestConsumersPickUpParquet:
    def test_to_webdataset_load_metadata_uses_parquet(self, tmp_path):
        from src import to_webdataset as twd

        _write_fixture_csv(tmp_path)
        with state_mode("parquet"):
            cm.ensure_main_csv(tmp_path)
            cm.upsert_columns(
                tmp_path,
                pd.DataFrame({"filepath": ["/d/a.wav"], "total_duration": [42.0]}),
                ["total_duration"],
            )
            meta = twd.load_metadata(tmp_path)
        # keyed by file stem; the fresh parquet duration must be visible.
        assert "a" in meta
        assert meta["a"]["total_duration"] == 42.0

    def test_collate_read_uses_parquet(self, tmp_path):
        from src import collate

        _write_fixture_csv(tmp_path)
        with state_mode("parquet"):
            cm.ensure_main_csv(tmp_path)
            cm.upsert_columns(
                tmp_path,
                pd.DataFrame({"filepath": ["/d/a.wav"], "total_duration": [42.0]}),
                ["total_duration"],
            )
            df = collate.read_state_for_collate(
                tmp_path, sidecar_columns=set(), config_path=None
            )
        assert "total_duration" in df.columns
        row = df[df["filepath"].astype(str).str.endswith("a.wav")]
        assert float(row["total_duration"].iloc[0]) == 42.0
