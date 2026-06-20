"""State-aware late-stage reads (parquet-only state).

The hot state file is ``balalaika.parquet`` (CSV state was removed). Direct
upserters — e.g. stage-7's ``ensure_audio_durations`` -> ``upsert_columns`` —
write the parquet state directly, and the late-stage consumers
(``to_webdataset.load_metadata`` and ``collate``'s main-frame read) read that
same state file.

This module pins:

* ``read_state_dataframe`` returns the parquet state, or an empty
  ``filepath``-only frame when no state file exists yet.
* The parquet-loaded frame is normalized to CSV-equivalent dtypes (nullable
  Int64-with-nulls -> float64, nullable boolean -> object/None) so consumers
  see the same dtypes they always did.
* ``to_webdataset.load_metadata`` and ``collate``'s read both pick up the
  parquet state, including columns written by a direct upsert.

Run: .dev_venv/bin/python -m pytest tests/test_state_aware_reads.py -q
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import csv_manager as cm


def _write_fixture_state(root: Path) -> None:
    pd.DataFrame(
        {
            "filepath": ["/d/a.wav", "/d/b.wav", "/d/c.wav"],
            "speaker_id": ["spk_0", "spk_1", "spk_2"],
            "crest_factor": [15.7, np.nan, 3.5],
        }
    ).to_parquet(root / "balalaika.parquet", index=False)


class TestReadStateDataframe:
    def test_direct_upsert_visible(self, tmp_path):
        """A direct upsert (no absorb) is visible to the state reader."""
        _write_fixture_state(tmp_path)
        cm.upsert_columns(
            tmp_path,
            pd.DataFrame(
                {"filepath": ["/d/a.wav", "/d/b.wav", "/d/c.wav"],
                 "total_duration": [10.0, 11.0, 12.0]}
            ),
            ["total_duration"],
        )
        df = cm.read_state_dataframe(tmp_path)
        assert "total_duration" in df.columns
        assert sorted(df["total_duration"].tolist()) == [10.0, 11.0, 12.0]

    def test_no_state_returns_empty(self, tmp_path):
        """No parquet on disk yet -> empty filepath-only frame."""
        assert not (tmp_path / "balalaika.parquet").exists()
        df = cm.read_state_dataframe(tmp_path)
        assert df["filepath"].tolist() == []

    def test_parquet_frame_normalized_to_csv_dtypes(self, tmp_path):
        """Nullable Int64/boolean from parquet -> float64 / object(None)."""
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
        df = cm.read_state_dataframe(tmp_path)
        assert df["podcast_id"].dtype == np.float64
        assert df["podcast_id"].tolist()[0] == 10.0
        assert pd.isna(df["podcast_id"].tolist()[1])
        assert df["is_single_speaker"].dtype == object
        assert df["is_single_speaker"].tolist() == [True, None, False]


class TestConsumersPickUpParquet:
    def test_to_webdataset_load_metadata_uses_parquet(self, tmp_path):
        from src import to_webdataset as twd

        _write_fixture_state(tmp_path)
        cm.upsert_columns(
            tmp_path,
            pd.DataFrame({"filepath": ["/d/a.wav"], "total_duration": [42.0]}),
            ["total_duration"],
        )
        meta = twd.load_metadata(tmp_path)
        assert "a" in meta
        assert meta["a"]["total_duration"] == 42.0

    def test_collate_read_uses_parquet(self, tmp_path):
        from src import collate

        _write_fixture_state(tmp_path)
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
