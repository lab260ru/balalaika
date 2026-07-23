"""Behavior tests for the parquet pipeline-state format.

CSV state was removed: the state layer (src.utils.csv_manager) keeps pipeline
state in ``balalaika.parquet`` only and never writes ``balalaika.csv``. These
tests pin that contract:

* A scripted sequence of state ops (ensure -> upsert -> flush -> absorb ->
  drop_missing -> reload) works on the parquet state across mixed
  int/float/str/NaN/unicode/quoted/empty-string columns.
* The parquet state file is written; no ``balalaika.csv`` is produced.
* ``state_format()`` is always ``"parquet"``.

Run: .dev_venv/bin/python -m pytest tests/test_csv_manager_parquet_state.py -q
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import csv_manager as cm


def _write_fixture_state(root: Path) -> None:
    """A production-shaped state exercising every dtype hazard.

    Mixed int-like / float / str / NaN / unicode-path / quoted / empty-string.
    """
    main = pd.DataFrame(
        {
            "filepath": [
                "/d/a.wav",
                "/d/b.wav",
                "/d/c.wav",
                "/данные/подкаст 1/файл.wav",  # unicode + space
                '/d/quote",comma.wav',
            ],
            "speaker_id": ["spk_0", "spk_1", "spk_2", "spk_3", "spk_4"],
            "crest_factor": [15.705164922563304, np.nan, 3.5, 7.0, 1.0],
            "loudness_normalized": ["True", "", "True", "", "True"],
            "total_duration": [10.0, 11.0, np.nan, 12.5, 0.0],
            "DistillMOS": [np.nan, 4.2, np.nan, 3.3, 2.1],
        }
    )
    main.to_parquet(root / "balalaika.parquet", index=False)


def _run_scripted_sequence(root: Path) -> pd.DataFrame:
    """ensure -> upsert -> partial flush -> absorb -> drop_missing -> reload."""
    cm.ensure_main_csv(root)

    # upsert: fill a hole, append a brand-new row
    inc = pd.DataFrame(
        {"filepath": ["/d/b.wav", "/d/new.wav"], "crest_factor": [9.9, 2.2]}
    )
    cm.upsert_columns(root, inc, ["crest_factor"], preserve_existing=True)

    # write a worker partial (transient .csv), fold it in via absorb
    pd.DataFrame(
        {
            "filepath": ["/d/c.wav", "/d/p2.wav"],
            "crest_factor": [5.5, 6.6],
            "total_duration": [3.0, 4.0],
        }
    ).to_csv(root / "crest_part_0.csv", index=False)
    cm.absorb_partial_csvs(
        root, "crest", ["crest_factor", "total_duration"], preserve_existing=True
    )

    # an upsert that overwrites a value (preserve False)
    cm.upsert_columns(
        root,
        pd.DataFrame({"filepath": ["/d/a.wav"], "DistillMOS": [np.nan]}),
        ["DistillMOS"],
        preserve_existing=False,
    )
    cm.absorb_partial_csvs(root, "crest", ["crest_factor"], preserve_existing=True)
    return cm.load_main_csv(root).reset_index(drop=True)


class TestScriptedSequence:
    def test_state_is_parquet_only(self, tmp_path):
        _write_fixture_state(tmp_path)
        _run_scripted_sequence(tmp_path)
        assert (tmp_path / "balalaika.parquet").exists()
        # CSV state is gone: only the transient worker partials may be .csv, and
        # those are deleted by absorb. No balalaika.csv export is produced.
        assert not (tmp_path / "balalaika.csv").exists()

    def test_merged_values(self, tmp_path):
        _write_fixture_state(tmp_path)
        df = _run_scripted_sequence(tmp_path)
        got = df.set_index("filepath")
        assert len(df) == 7  # a,b,c,unicode,quote,new,p2
        assert got.loc["/d/b.wav", "crest_factor"] == 9.9  # hole filled
        assert got.loc["/d/c.wav", "crest_factor"] == 5.5  # partial overwrote
        assert got.loc["/d/c.wav", "total_duration"] == 3.0
        assert got.loc["/d/new.wav", "crest_factor"] == 2.2  # appended
        assert got.loc["/d/p2.wav", "crest_factor"] == 6.6
        assert math.isnan(got.loc["/d/a.wav", "DistillMOS"])  # overwritten w/ NaN


class TestUnprocessedPathsParquet:
    def test_pending_set(self, tmp_path):
        pd.DataFrame(
            {
                "filepath": ["/d/done.wav", "/d/hole.wav"],
                "crest_factor": [1.0, np.nan],
            }
        ).to_parquet(tmp_path / "balalaika.parquet", index=False)
        pending = cm.unprocessed_paths(
            tmp_path, "crest_factor", ["/d/done.wav", "/d/hole.wav", "/d/new.wav"]
        )
        assert pending == ["/d/hole.wav", "/d/new.wav"]

    def test_object_blank_string_column_pending(self, tmp_path):
        pd.DataFrame(
            {"filepath": ["/d/a.wav", "/d/b.wav"], "loudness_normalized": ["True", "  "]}
        ).to_parquet(tmp_path / "balalaika.parquet", index=False)
        pending = cm.unprocessed_paths(
            tmp_path, "loudness_normalized", ["/d/a.wav", "/d/b.wav"]
        )
        assert pending == ["/d/b.wav"]


class TestStateFormat:
    def test_always_parquet(self):
        assert cm.state_format() == "parquet"

    def test_state_path_is_parquet(self, tmp_path):
        assert cm.state_path(tmp_path).name == "balalaika.parquet"
