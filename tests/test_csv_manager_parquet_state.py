"""Equivalence + behavior tests for the parquet pipeline-state format.

The state layer (src.utils.csv_manager) can keep pipeline state either as
``balalaika.csv`` (default) or ``balalaika.parquet`` (``csv.state_format:
parquet`` -> ``BALALAIKA_STATE_FORMAT=parquet``). The hard contract these tests
pin:

* A scripted sequence of state ops (ensure -> upsert -> flush -> absorb ->
  drop_missing -> final export) run in csv mode vs parquet mode on the same
  fixture must produce a **byte-identical** final ``balalaika.csv`` export and
  **value-identical** intermediate DataFrames — across mixed
  int/float/str/NaN/unicode/quoted/empty-string columns.
* Resume interop: parquet mode + only CSV on disk migrates on first load; csv
  mode + stale parquet warns and ignores.
* The csv-mode default behaves exactly as before (no parquet written).

Run: .dev_venv/bin/python -m pytest tests/test_csv_manager_parquet_state.py -q
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
    """A production-shaped CSV exercising every dtype hazard.

    Mixed int-like / float / str / NaN / unicode-path / quoted / empty-string.
    """
    main = pd.DataFrame(
        {
            "filepath": [
                "/d/a.wav",
                "/d/b.wav",
                "/d/c.wav",
                "/данные/подкаст 1/файл.wav",  # unicode + space
                '/d/quote",comma.wav',  # needs RFC-4180 quoting
            ],
            "speaker_id": ["spk_0", "spk_1", "spk_2", "spk_3", "spk_4"],
            "crest_factor": [15.705164922563304, np.nan, 3.5, 7.0, 1.0],
            "loudness_normalized": ["True", "", "True", "", "True"],
            "total_duration": [10.0, 11.0, np.nan, 12.5, 0.0],
            "DistillMOS": [np.nan, 4.2, np.nan, 3.3, 2.1],
        }
    )
    main.to_csv(root / "balalaika.csv", index=False)


def _run_scripted_sequence(root: Path) -> pd.DataFrame:
    """ensure -> upsert -> partial flush -> absorb -> drop_missing -> reload."""
    # ensure (no-op load; CSV already present / migrates in parquet mode)
    cm.ensure_main_csv(root)

    # upsert: fill a hole, append a brand-new row
    inc = pd.DataFrame(
        {
            "filepath": ["/d/b.wav", "/d/new.wav"],
            "crest_factor": [9.9, 2.2],
        }
    )
    cm.upsert_columns(root, inc, ["crest_factor"], preserve_existing=True)

    # write a worker partial, fold it in via absorb (the flush + final export)
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

    # an upsert that bootstraps a path and overwrites a value (preserve False)
    cm.upsert_columns(
        root,
        pd.DataFrame({"filepath": ["/d/a.wav"], "DistillMOS": [np.nan]}),
        ["DistillMOS"],
        preserve_existing=False,
    )
    cm.absorb_partial_csvs(root, "crest", ["crest_factor"], preserve_existing=True)
    return cm.load_main_csv(root).reset_index(drop=True)


def _run_both_modes(tmp_path):
    out = {}
    for mode in ("csv", "parquet"):
        root = tmp_path / mode
        root.mkdir()
        _write_fixture_csv(root)
        with state_mode(mode):
            df = _run_scripted_sequence(root)
        out[mode] = {
            "df": df,
            "csv_bytes": (root / "balalaika.csv").read_bytes(),
            "root": root,
        }
    return out


class TestScriptedEquivalence:
    def test_csv_export_byte_identical(self, tmp_path):
        res = _run_both_modes(tmp_path)
        assert res["csv"]["csv_bytes"] == res["parquet"]["csv_bytes"]

    def test_final_dataframes_value_identical(self, tmp_path):
        res = _run_both_modes(tmp_path)
        # Same columns, same order, same values (incl. NaN positions).
        pd.testing.assert_frame_equal(res["csv"]["df"], res["parquet"]["df"])

    def test_parquet_state_file_written(self, tmp_path):
        res = _run_both_modes(tmp_path)
        assert (res["parquet"]["root"] / "balalaika.parquet").exists()
        # csv mode never creates a parquet
        assert not (res["csv"]["root"] / "balalaika.parquet").exists()

    def test_csv_export_present_in_both_modes(self, tmp_path):
        res = _run_both_modes(tmp_path)
        assert (res["parquet"]["root"] / "balalaika.csv").exists()
        assert (res["csv"]["root"] / "balalaika.csv").exists()


class TestMigrationAndInterop:
    def test_parquet_mode_migrates_legacy_csv(self, tmp_path):
        _write_fixture_csv(tmp_path)
        assert not (tmp_path / "balalaika.parquet").exists()
        with state_mode("parquet"):
            df = cm.load_main_csv(tmp_path)
        # parquet now exists and holds the same rows
        assert (tmp_path / "balalaika.parquet").exists()
        assert len(df) == 5
        # CSV is left intact as the export
        assert (tmp_path / "balalaika.csv").exists()

    def test_csv_mode_ignores_stale_parquet(self, tmp_path):
        # Build a parquet that DISAGREES with the CSV; csv mode must read CSV.
        _write_fixture_csv(tmp_path)
        stale = pd.DataFrame({"filepath": ["/stale/only.wav"], "crest_factor": [42.0]})
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pandas(stale, preserve_index=False),
                       str(tmp_path / "balalaika.parquet"))
        with state_mode("csv"):
            df = cm.load_main_csv(tmp_path)
        assert "/stale/only.wav" not in set(df["filepath"])
        assert len(df) == 5

    def test_parquet_mode_uses_parquet_over_csv_after_migration(self, tmp_path):
        _write_fixture_csv(tmp_path)
        with state_mode("parquet"):
            cm.load_main_csv(tmp_path)  # migrate
            # Now mutate ONLY the CSV export behind the state's back.
            (tmp_path / "balalaika.csv").write_text("filepath\n/decoy/x.wav\n")
            df = cm.load_main_csv(tmp_path)
        # Parquet (the real state) wins, decoy ignored.
        assert "/decoy/x.wav" not in set(df["filepath"])
        assert len(df) == 5


class TestUnprocessedPathsEquivalence:
    def test_pending_sets_match_across_modes(self, tmp_path):
        audio = [
            "/d/a.wav",
            "/d/b.wav",
            "/d/c.wav",
            "/данные/подкаст 1/файл.wav",
            "/d/never_seen.wav",
        ]
        pending = {}
        for mode in ("csv", "parquet"):
            root = tmp_path / mode
            root.mkdir()
            _write_fixture_csv(root)
            with state_mode(mode):
                pending[mode] = cm.unprocessed_paths(root, "crest_factor", audio)
        assert pending["csv"] == pending["parquet"]

    def test_missing_column_all_pending_parquet(self, tmp_path):
        _write_fixture_csv(tmp_path)
        audio = ["/d/a.wav", "/d/b.wav"]
        with state_mode("parquet"):
            cm.load_main_csv(tmp_path)  # migrate to parquet
            pending = cm.unprocessed_paths(tmp_path, "nonexistent_col", audio)
        assert pending == audio

    def test_object_blank_string_column_pending_matches(self, tmp_path):
        # "True" -> done, "  " (whitespace) and NaN -> pending, in BOTH modes.
        pending = {}
        for mode in ("csv", "parquet"):
            root = tmp_path / mode
            root.mkdir()
            pd.DataFrame(
                {
                    "filepath": ["/d/a.wav", "/d/b.wav", "/d/c.wav"],
                    "loudness_normalized": ["True", "  ", np.nan],
                }
            ).to_csv(root / "balalaika.csv", index=False)
            with state_mode(mode):
                pending[mode] = cm.unprocessed_paths(
                    root, "loudness_normalized", ["/d/a.wav", "/d/b.wav", "/d/c.wav"]
                )
        assert pending["csv"] == pending["parquet"] == ["/d/b.wav", "/d/c.wav"]


class TestDefaultModeUnchanged:
    def test_default_is_csv(self):
        prev = os.environ.pop("BALALAIKA_STATE_FORMAT", None)
        try:
            assert cm.state_format() == "csv"
        finally:
            if prev is not None:
                os.environ["BALALAIKA_STATE_FORMAT"] = prev

    def test_unknown_value_falls_back_to_csv(self):
        with state_mode("nonsense"):
            assert cm.state_format() == "csv"
            assert cm.state_path("/x") == cm.csv_path("/x")
