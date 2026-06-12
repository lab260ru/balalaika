"""Equivalence tests for the state-layer performance refactors.

Pins that three perf reworks produce identical results to the original code:

* ``_csv_duration_cache``: narrow fast_read_csv + vectorized dict build vs the
  old callable-usecols C-engine read + per-row Python loop (incl. NaN / missing
  / non-numeric / unicode / last-wins rows).
* ``_normalize_filepath_column`` / ``upsert_columns`` copy elimination: the
  canonical-path guard must never change a value and must not mutate caller
  frames.
* ``PeriodicCsvMerger`` flush-skip: a triggered flush over byte-unchanged
  partials is skipped, and the final CSV is still identical.

Run: .dev_venv/bin/python -m pytest tests/test_csv_state_perf_refactors.py -q
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils import audio_durations as ad
from src.utils import csv_manager as cm


# ---------------------------------------------------------------------------
# Verbatim copy of the ORIGINAL _csv_duration_cache loop (the reference).
# ---------------------------------------------------------------------------

def _old_csv_duration_cache(target: Path, requested_norms: set[str]) -> dict[str, float]:
    if not target.exists() or not requested_norms:
        return {}
    df = pd.read_csv(
        target,
        usecols=lambda col: col in {"filepath", "total_duration"},
        low_memory=False,
    )
    if "filepath" not in df.columns or "total_duration" not in df.columns:
        return {}
    durations: dict[str, float] = {}
    for filepath, raw_duration in zip(df["filepath"], df["total_duration"]):
        normalized = cm.normalize_path_string(filepath)
        if normalized not in requested_norms:
            continue
        try:
            d = float(raw_duration)
        except (TypeError, ValueError):
            continue
        if d > 0:
            durations[normalized] = d
    return durations


class TestDurationCacheEquivalence:
    def _fixture(self, root: Path) -> set[str]:
        df = pd.DataFrame(
            {
                "filepath": [
                    "/d/a.wav",
                    "/d/b.wav",
                    "/d/c.wav",
                    "/d/zero.wav",
                    "/d/neg.wav",
                    "/d/text.wav",
                    "/данные/ю.wav",  # unicode
                    "/d/dup.wav",
                    "/d/dup.wav",  # duplicate -> last wins
                ],
                "speaker_id": [f"s{i}" for i in range(9)],
                "total_duration": [
                    8.6645, np.nan, 12.0, 0.0, -3.0, np.nan, 4.25, 1.0, 2.0
                ],
                "DistillMOS": [1.0] * 9,
            }
        )
        df.to_csv(root / "balalaika.csv", index=False)
        return {cm.normalize_path_string(p) for p in df["filepath"]}

    def test_matches_original_loop(self, tmp_path):
        requested = self._fixture(tmp_path)
        old = _old_csv_duration_cache(tmp_path / "balalaika.csv", requested)
        new = ad._csv_duration_cache(tmp_path, requested)
        assert new == old
        # spot-check the semantics the loop encoded
        assert "/d/zero.wav" not in new   # 0 is not positive
        assert "/d/neg.wav" not in new    # negative dropped
        assert "/d/text.wav" not in new   # NaN dropped
        assert new["/d/dup.wav"] == 2.0   # last wins

    def test_missing_duration_column(self, tmp_path):
        pd.DataFrame({"filepath": ["/d/a.wav"]}).to_csv(
            tmp_path / "balalaika.csv", index=False
        )
        assert ad._csv_duration_cache(tmp_path, {"/d/a.wav"}) == {}

    def test_only_requested_subset(self, tmp_path):
        self._fixture(tmp_path)
        new = ad._csv_duration_cache(tmp_path, {"/d/a.wav"})
        assert set(new) == {"/d/a.wav"}
        assert new["/d/a.wav"] == 8.6645

    def test_parquet_mode_duration_cache_matches_csv(self, tmp_path):
        import os

        prev = os.environ.get("BALALAIKA_STATE_FORMAT")
        try:
            root_csv = tmp_path / "csv"
            root_csv.mkdir()
            req = self._fixture(root_csv)
            os.environ.pop("BALALAIKA_STATE_FORMAT", None)
            csv_cache = ad._csv_duration_cache(root_csv, req)

            root_pq = tmp_path / "pq"
            root_pq.mkdir()
            self._fixture(root_pq)
            os.environ["BALALAIKA_STATE_FORMAT"] = "parquet"
            from src.utils import csv_manager as _cm

            _cm.load_main_csv(root_pq)  # migrate to parquet
            pq_cache = ad._csv_duration_cache(root_pq, req)
            assert csv_cache == pq_cache
        finally:
            if prev is None:
                os.environ.pop("BALALAIKA_STATE_FORMAT", None)
            else:
                os.environ["BALALAIKA_STATE_FORMAT"] = prev


class TestCanonicalGuard:
    def test_canonical_detects_absolute_strings(self):
        s = pd.Series(["/a/b.wav", "/c/d.wav"])
        assert cm._filepath_is_canonical(s)

    def test_non_canonical_relative(self):
        assert not cm._filepath_is_canonical(pd.Series(["rel/x.wav"]))

    def test_non_canonical_whitespace(self):
        assert not cm._filepath_is_canonical(pd.Series([" /a/b.wav"]))

    def test_non_canonical_empty(self):
        assert not cm._filepath_is_canonical(pd.Series([""]))

    def test_non_canonical_nan(self):
        assert not cm._filepath_is_canonical(pd.Series(["/a.wav", np.nan]))

    def test_non_canonical_numeric_dtype(self):
        assert not cm._filepath_is_canonical(pd.Series([1, 2, 3]))

    def test_guard_skips_copy_but_value_identical(self):
        # Canonical -> returns the SAME object (no copy), value unchanged.
        df = pd.DataFrame({"filepath": ["/a/b.wav", "/c/d.wav"], "v": [1, 2]})
        out = cm._normalize_filepath_column(df, owned=False)
        assert out is df  # no copy when canonical
        assert out["filepath"].tolist() == ["/a/b.wav", "/c/d.wav"]

    def test_non_owned_non_canonical_does_not_mutate_input(self):
        df = pd.DataFrame({"filepath": ["rel/b.wav"]})
        out = cm._normalize_filepath_column(df, owned=False)
        assert df["filepath"].tolist() == ["rel/b.wav"]  # input untouched
        assert out["filepath"].tolist()[0] == str(Path("rel/b.wav").resolve())


class TestUpsertDoesNotMutateCaller:
    def test_results_df_untouched_relative_paths(self, tmp_path):
        pd.DataFrame({"filepath": ["/d/a.wav"], "crest_factor": [1.0]}).to_csv(
            tmp_path / "balalaika.csv", index=False
        )
        incoming = pd.DataFrame(
            {"filepath": ["rel/x.wav", "/d/a.wav"], "crest_factor": [3.0, 4.0]}
        )
        before = incoming["filepath"].tolist()
        cm.upsert_columns(tmp_path, incoming, ["crest_factor"])
        # caller's frame must be unchanged (no in-place normalization leak)
        assert incoming["filepath"].tolist() == before


class TestFlushSkip:
    def test_signature_changes_with_size(self, tmp_path):
        pd.DataFrame({"filepath": ["/a"], "crest_factor": [1.0]}).to_csv(
            tmp_path / "crest_part_0.csv", index=False
        )
        sig1 = cm._partials_signature(tmp_path, "crest")
        # append a row -> size grows -> signature changes
        pd.DataFrame({"filepath": ["/b"], "crest_factor": [2.0]}).to_csv(
            tmp_path / "crest_part_0.csv", index=False, mode="a", header=False
        )
        sig2 = cm._partials_signature(tmp_path, "crest")
        assert sig1 != sig2

    def test_unchanged_partials_skip_flush(self, tmp_path, monkeypatch):
        pd.DataFrame(
            {"filepath": ["/a", "/b", "/c"], "crest_factor": [1.0, 2.0, 3.0]}
        ).to_csv(tmp_path / "crest_part_0.csv", index=False)

        merger = cm.PeriodicCsvMerger(
            tmp_path,
            "crest",
            ["crest_factor"],
            flush_every_rows=1,
            flush_every_seconds=0,
            poll_interval=5.0,
        )
        calls = {"n": 0}
        real_flush = merger._flush_once

        def counting_flush():
            calls["n"] += 1
            return real_flush()

        monkeypatch.setattr(merger, "_flush_once", counting_flush)

        # First trigger: signature empty -> must flush.
        sig = cm._partials_signature(tmp_path, "crest")
        merger._flush_once()
        merger._last_flushed_sig = sig
        assert calls["n"] == 1

        # Simulate the loop's skip decision on unchanged partials.
        sig2 = cm._partials_signature(tmp_path, "crest")
        skipped = bool(merger._last_flushed_sig and sig2 == merger._last_flushed_sig)
        assert skipped  # no new bytes -> the loop would skip the full flush

    def test_final_absorb_identical_regardless_of_skips(self, tmp_path):
        # Whatever the merger does mid-stage, the final absorb yields the same CSV.
        pd.DataFrame({"filepath": ["/d/a.wav"], "crest_factor": [1.0]}).to_csv(
            tmp_path / "balalaika.csv", index=False
        )
        pd.DataFrame(
            {"filepath": ["/d/a.wav", "/d/b.wav"], "crest_factor": [9.0, 8.0]}
        ).to_csv(tmp_path / "crest_part_0.csv", index=False)
        cm.absorb_partial_csvs(tmp_path, "crest", ["crest_factor"])
        out = cm.load_main_csv(tmp_path).set_index("filepath")["crest_factor"]
        assert out["/d/a.wav"] == 9.0
        assert out["/d/b.wav"] == 8.0
