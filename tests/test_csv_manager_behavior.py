"""Behavioral contract tests for src.utils.csv_manager and path discovery.

These tests pin down the OBSERVABLE semantics of the CSV state layer:
normalization rules, upsert merge rules, resume bookkeeping, and path
discovery filtering. They were written against the original implementation
(pre-optimization) and must keep passing after any performance rework.

Run: .dev_venv/bin/python -m pytest tests/test_csv_manager_behavior.py -q
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils import csv_manager as cm
from src.utils.utils import get_audio_paths


def _write_state(df: pd.DataFrame, directory) -> None:
    """Write the parquet pipeline state (the only state format)."""
    df.to_parquet(Path(directory) / "balalaika.parquet", index=False)


def _read_state(directory) -> pd.DataFrame:
    return pd.read_parquet(Path(directory) / "balalaika.parquet")


# ---------------------------------------------------------------------------
# normalize_path_string / normalize_path_values
# ---------------------------------------------------------------------------

class TestNormalizePathString:
    def test_absolute_path_returned_as_is(self):
        assert cm.normalize_path_string("/abs/file.wav") == "/abs/file.wav"

    def test_absolute_with_dotdot_not_resolved(self):
        # Historical behavior: absolute paths are trusted verbatim, even with '..'
        assert cm.normalize_path_string("/abs/../other/f.wav") == "/abs/../other/f.wav"

    def test_relative_path_resolved_against_cwd(self):
        out = cm.normalize_path_string("rel/file.wav")
        assert out == str(Path("rel/file.wav").resolve())
        assert os.path.isabs(out)

    def test_strip_whitespace(self):
        assert cm.normalize_path_string("  /abs/file.wav  ") == "/abs/file.wav"

    def test_empty_and_whitespace(self):
        assert cm.normalize_path_string("") == ""
        assert cm.normalize_path_string("   ") == ""

    def test_cyrillic_and_spaces(self):
        p = "/данные/подкаст 1/файл.wav"
        assert cm.normalize_path_string(p) == p

    def test_values_list_preserves_order_and_drop_empty(self):
        vals = ["/a/x.wav", "", "rel.wav", "  ", "/b/y.wav"]
        out = cm.normalize_path_values(vals, desc="t", drop_empty=True)
        assert out[0] == "/a/x.wav"
        assert out[-1] == "/b/y.wav"
        assert len(out) == 3
        out2 = cm.normalize_path_values(vals, desc="t", drop_empty=False)
        assert len(out2) == 5 and out2[1] == "" and out2[3] == ""


# ---------------------------------------------------------------------------
# DataFrame filepath normalization
# ---------------------------------------------------------------------------

class TestNormalizeFilepathColumn:
    def test_mixed_column(self):
        df = pd.DataFrame(
            {
                "filepath": ["/abs/a.wav", "rel/b.wav", "", np.nan, "/abs/../c.wav"],
                "v": [1, 2, 3, 4, 5],
            }
        )
        out = cm._normalize_filepath_column(df)
        assert out["filepath"].tolist()[0] == "/abs/a.wav"
        assert out["filepath"].tolist()[1] == str(Path("rel/b.wav").resolve())
        assert out["filepath"].tolist()[2] == ""
        # astype(str) turns NaN into the literal 'nan' -> resolved against cwd.
        assert out["filepath"].tolist()[3] == str(Path("nan").resolve())
        assert out["filepath"].tolist()[4] == "/abs/../c.wav"
        # untouched other columns, no row reorder
        assert out["v"].tolist() == [1, 2, 3, 4, 5]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"filepath": ["rel/b.wav"]})
        cm._normalize_filepath_column(df)
        assert df["filepath"].tolist() == ["rel/b.wav"]

    def test_empty_df_passthrough(self):
        df = pd.DataFrame()
        assert cm._normalize_filepath_column(df) is df


# ---------------------------------------------------------------------------
# upsert_columns semantics
# ---------------------------------------------------------------------------

@pytest.fixture()
def state_dir(tmp_path):
    main = pd.DataFrame(
        {
            "filepath": ["/d/a.wav", "/d/b.wav", "/d/c.wav"],
            "crest_factor": [1.5, np.nan, 3.5],
            "total_duration": [10.0, 11.0, 12.0],
        }
    )
    _write_state(main, tmp_path)
    return tmp_path


class TestUpsertColumns:
    def test_preserve_existing_true(self, state_dir):
        incoming = pd.DataFrame(
            {
                "filepath": ["/d/a.wav", "/d/b.wav", "/d/new.wav"],
                "crest_factor": [np.nan, 2.0, 9.0],
            }
        )
        out = cm.upsert_columns(state_dir, incoming, ["crest_factor"], preserve_existing=True)
        got = out.set_index("filepath")["crest_factor"]
        assert got["/d/a.wav"] == 1.5  # NaN incoming cannot erase existing
        assert got["/d/b.wav"] == 2.0  # incoming fills the hole
        assert got["/d/c.wav"] == 3.5  # untouched row
        assert got["/d/new.wav"] == 9.0  # appended
        assert len(out) == 4

    def test_preserve_existing_false_overwrites_with_nan(self, state_dir):
        incoming = pd.DataFrame(
            {"filepath": ["/d/a.wav"], "crest_factor": [np.nan]}
        )
        out = cm.upsert_columns(state_dir, incoming, ["crest_factor"], preserve_existing=False)
        got = out.set_index("filepath")["crest_factor"]
        assert math.isnan(got["/d/a.wav"])
        assert got["/d/c.wav"] == 3.5

    def test_duplicate_incoming_keeps_last(self, state_dir):
        incoming = pd.DataFrame(
            {"filepath": ["/d/a.wav", "/d/a.wav"], "crest_factor": [7.0, 8.0]}
        )
        out = cm.upsert_columns(state_dir, incoming, ["crest_factor"], preserve_existing=True)
        got = out.set_index("filepath")["crest_factor"]
        assert got["/d/a.wav"] == 8.0
        assert len(out) == 3

    def test_value_columns_filter(self, state_dir):
        incoming = pd.DataFrame(
            {
                "filepath": ["/d/a.wav"],
                "crest_factor": [7.0],
                "rogue_column": ["x"],
            }
        )
        out = cm.upsert_columns(state_dir, incoming, ["crest_factor"], preserve_existing=True)
        assert "rogue_column" not in out.columns

    def test_new_column_added(self, state_dir):
        incoming = pd.DataFrame({"filepath": ["/d/b.wav"], "music_prob": [0.25]})
        out = cm.upsert_columns(state_dir, incoming, ["music_prob"], preserve_existing=True)
        got = out.set_index("filepath")["music_prob"]
        assert got["/d/b.wav"] == 0.25
        assert math.isnan(got["/d/a.wav"])

    def test_bootstrap_adds_only_new_rows(self, state_dir):
        out = cm.upsert_columns(
            state_dir,
            pd.DataFrame(),
            ["crest_factor"],
            bootstrap_audio_paths=["/d/a.wav", "/d/z.wav"],
            preserve_existing=True,
        )
        assert len(out) == 4
        got = out.set_index("filepath")["crest_factor"]
        assert got["/d/a.wav"] == 1.5  # bootstrap must not clobber values

    def test_drop_missing_files(self, tmp_path):
        real = tmp_path / "real.wav"
        real.touch()
        main = pd.DataFrame(
            {
                "filepath": [str(real), str(tmp_path / "gone.wav")],
                "crest_factor": [1.0, 2.0],
            }
        )
        _write_state(main, tmp_path)
        out = cm.upsert_columns(
            tmp_path, pd.DataFrame(), ["crest_factor"], drop_missing_files=True
        )
        assert out["filepath"].tolist() == [str(real)]

    def test_column_ordering_follows_base_columns(self, state_dir):
        incoming = pd.DataFrame(
            {"filepath": ["/d/a.wav"], "DistillMOS": [4.2], "zz_extra": [1]}
        )
        out = cm.upsert_columns(
            state_dir, incoming, ["DistillMOS", "zz_extra"], preserve_existing=True
        )
        cols = list(out.columns)
        # canonical prefix order, extras appended at the end
        assert cols.index("filepath") < cols.index("crest_factor") < cols.index("DistillMOS")
        assert cols[-1] == "zz_extra"

    def test_atomic_write_keeps_backup(self, state_dir):
        before = _read_state(state_dir)
        cm.upsert_columns(
            state_dir,
            pd.DataFrame({"filepath": ["/d/a.wav"], "crest_factor": [9.9]}),
            ["crest_factor"],
        )
        bak = state_dir / "balalaika.parquet.bak"
        assert bak.exists()
        bak_df = pd.read_parquet(bak)
        # .bak holds the PREVIOUS generation
        assert bak_df.set_index("filepath")["crest_factor"]["/d/a.wav"] == before.set_index("filepath")["crest_factor"]["/d/a.wav"]


# ---------------------------------------------------------------------------
# unprocessed_paths
# ---------------------------------------------------------------------------

class TestUnprocessedPaths:
    def test_pending_logic(self, tmp_path):
        main = pd.DataFrame(
            {
                "filepath": ["/d/done.wav", "/d/empty.wav", "/d/nanrow.wav"],
                "crest_factor": [1.0, np.nan, np.nan],
            }
        )
        _write_state(main, tmp_path)
        audio = ["/d/done.wav", "/d/empty.wav", "/d/nanrow.wav", "/d/new.wav", ""]
        pending = cm.unprocessed_paths(tmp_path, "crest_factor", audio)
        assert pending == ["/d/empty.wav", "/d/nanrow.wav", "/d/new.wav"]

    def test_object_column_blank_strings_are_pending(self, tmp_path):
        main = pd.DataFrame(
            {
                "filepath": ["/d/a.wav", "/d/b.wav"],
                "loudness_normalized": ["True", "  "],
            }
        )
        _write_state(main, tmp_path)
        pending = cm.unprocessed_paths(
            tmp_path, "loudness_normalized", ["/d/a.wav", "/d/b.wav"]
        )
        assert pending == ["/d/b.wav"]

    def test_missing_column_returns_all(self, tmp_path):
        _write_state(pd.DataFrame({"filepath": ["/d/a.wav"]}), tmp_path)
        pending = cm.unprocessed_paths(tmp_path, "nope", ["/d/a.wav", "rel.wav"])
        assert pending == ["/d/a.wav", str(Path("rel.wav").resolve())]


# ---------------------------------------------------------------------------
# partials
# ---------------------------------------------------------------------------

class TestPartials:
    def test_read_partials_dedup_keep_last(self, tmp_path):
        pd.DataFrame(
            {"filepath": ["/d/a.wav", "/d/b.wav"], "crest_factor": [1.0, 2.0]}
        ).to_csv(tmp_path / "crest_part_0.csv", index=False)
        pd.DataFrame(
            {"filepath": ["/d/a.wav"], "crest_factor": [5.0]}
        ).to_csv(tmp_path / "crest_part_1.csv", index=False)
        merged = cm.read_partial_csvs(tmp_path, "crest")
        got = merged.set_index("filepath")["crest_factor"]
        assert got["/d/a.wav"] == 5.0  # later partial wins
        assert len(merged) == 2

    def test_empty_partial_skipped(self, tmp_path):
        (tmp_path / "crest_part_0.csv").touch()
        merged = cm.read_partial_csvs(tmp_path, "crest")
        assert merged.empty

    def test_count_partial_rows(self, tmp_path):
        pd.DataFrame({"filepath": ["/a", "/b", "/c"]}).to_csv(
            tmp_path / "crest_part_0.csv", index=False
        )
        (tmp_path / "crest_part_1.csv").touch()
        assert cm._count_partial_rows(tmp_path, "crest") == 3

    def test_already_processed_from_partials(self, tmp_path):
        pd.DataFrame(
            {"filepath": ["/d/a.wav", "/d/b.wav"], "crest_factor": [1.0, np.nan]}
        ).to_csv(tmp_path / "crest_part_0.csv", index=False)
        done = cm.already_processed_from_partials(tmp_path, "crest", "crest_factor")
        assert done == {"/d/a.wav"}

    def test_partial_writer_roundtrip(self, tmp_path):
        with cm.partial_writer(tmp_path, "crest", 0, fieldnames=cm.PARTIAL_FIELDS if hasattr(cm, "PARTIAL_FIELDS") else ("filepath", "crest_factor")) as w:
            w.write({"filepath": "/d/a.wav", "crest_factor": 1.25})
        with cm.partial_writer(tmp_path, "crest", 0) as w:
            assert w.already_done() == {"/d/a.wav"}
            w.write({"filepath": "/d/b.wav", "crest_factor": 2.5})
        df = pd.read_csv(tmp_path / "crest_part_0.csv")
        assert len(df) == 2  # append mode, single header


# ---------------------------------------------------------------------------
# audit + discovery
# ---------------------------------------------------------------------------

class TestAuditAndDiscovery:
    def test_audit_from_filter_partials(self):
        df = pd.DataFrame(
            {
                "filepath": ["/a", "/b", "/c", "/d"],
                "duration_s": [3600.0, 3600.0, 3600.0, np.nan],
                "deleted": [True, False, "true", ""],
            }
        )
        audit = cm.audit_from_filter_partials(df)
        assert audit["files_in"] == 4
        assert audit["files_deleted"] == 2
        assert audit["files_out"] == 2
        assert audit["hours_in"] == pytest.approx(3.0)
        assert audit["hours_out"] == pytest.approx(1.0)

    def test_dedupe_paths_filters_and_orders(self):
        vals = [
            "/d/a.wav",
            "/d/a.wav",  # duplicate
            "/d/b.txt",  # non-audio
            "/d/c.WAV",  # uppercase ext accepted
            None,
            "",
            "/d/e.opus",
        ]
        out = cm._dedupe_paths(vals)
        assert out == ["/d/a.wav", "/d/c.WAV", "/d/e.opus"]

    def test_audio_paths_from_csv(self, tmp_path):
        _write_state(
            pd.DataFrame({"filepath": ["/d/a.wav", "/d/b.txt", "/d/a.wav", np.nan]}),
            tmp_path,
        )
        out = cm._audio_paths_from_csv(tmp_path)
        assert out == ["/d/a.wav"]

    def test_get_audio_paths_extensions(self, tmp_path):
        names = ["x.mp3", "y.wav", "z.flac", "w.ogg", "v.opus", "no.txt", "X.MP3"]
        sub = tmp_path / "p1" / "p2"
        sub.mkdir(parents=True)
        for n in names:
            (sub / n).touch()
        out = sorted(str(p) for p in get_audio_paths(str(tmp_path)))
        expected = sorted(str(sub / n) for n in ["x.mp3", "y.wav", "z.flac", "w.ogg", "v.opus"])
        # case-sensitive match: X.MP3 excluded, parity with the original rglob
        assert out == expected

    def test_files_in_csv(self):
        df = pd.DataFrame({"filepath": ["/a.wav", "", "/b.wav"]})
        assert cm.files_in_csv(df) == {"/a.wav", "/b.wav"}


# ---------------------------------------------------------------------------
# ensure_main_csv recovery
# ---------------------------------------------------------------------------

class TestEnsureMainCsv:
    def test_bootstrap_from_audio_paths(self, tmp_path):
        df = cm.ensure_main_csv(tmp_path, audio_paths=["/d/b.wav", "/d/a.wav", "/d/a.wav"])
        assert df["filepath"].tolist() == ["/d/a.wav", "/d/b.wav"]  # sorted unique
        assert (tmp_path / "balalaika.parquet").exists()

    def test_existing_csv_loaded(self, tmp_path):
        _write_state(pd.DataFrame({"filepath": ["/d/a.wav"], "v": [1]}), tmp_path)
        df = cm.ensure_main_csv(tmp_path)
        assert df["v"].tolist() == [1]

    def test_corrupt_csv_restored_from_bak(self, tmp_path):
        good = pd.DataFrame({"filepath": ["/d/a.wav"], "v": [1]})
        good.to_parquet(tmp_path / "balalaika.parquet.bak", index=False)
        (tmp_path / "balalaika.parquet").write_text("not a parquet file")  # corrupt
        df = cm.ensure_main_csv(tmp_path)
        assert df["filepath"].tolist() == ["/d/a.wav"]
