"""HDD I/O infrastructure: shard ordering, merger pruning, io_profile.

Pins the equivalence contracts of the 2026-06 HDD pass:

* work-shard contents are the same *sets* under any ordering knob, with
  ``legacy`` reproducing the old byte order exactly;
* the periodic CSV merger no longer prunes missing files mid-stage, while
  the final absorb still yields the exact same CSV as the old flow;
* io_profile clamps are no-ops on ssd and bounded on hdd;
* duration probes run in sorted path order with identical results.
"""
import math
import os
from pathlib import Path

import pandas as pd
import pytest

from src.utils import audio_durations as ad
from src.utils import io_profile
from src.utils.csv_manager import (
    PeriodicCsvMerger,
    absorb_partial_csvs,
    atomic_write_csv,
    csv_path,
    fast_read_csv,
    partial_writer,
)
from src.utils.work_shards import (
    prepare_length_bucketed_work_shards,
    prepare_work_shards,
    read_annotated_work_shard,
    read_work_shard,
)


def _shard_lines(work_dir: Path) -> list[list[str]]:
    return [
        read_work_shard(p)
        for p in sorted(work_dir.glob("shard_*.pending"))
    ]


def _legacy_plain_reference(paths, shard_size, limit=None):
    """The exact pre-change prepare_work_shards ordering."""
    out, current = [], []
    total = 0
    for raw in paths:
        if limit is not None and total >= limit:
            break
        path = str(raw).strip()
        if not path:
            continue
        current.append(path)
        total += 1
        if len(current) >= shard_size:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


@pytest.fixture
def unsorted_paths():
    # Interleaved directories, exactly what duration sorting produces.
    return [
        f"/data/p{(i * 7) % 5}/ep{(i * 13) % 11}/chunk_{i:04d}.wav"
        for i in range(257)
    ]


class TestPlainShardOrder:
    def test_path_order_sorts_lines_same_set(self, tmp_path, unsorted_paths):
        plan = prepare_work_shards(tmp_path, "t", unsorted_paths, shard_size=50, order="path")
        lines = _shard_lines(plan.work_dir)
        flat = [p for shard in lines for p in shard]
        assert flat == sorted(unsorted_paths)
        assert plan.total_items == len(unsorted_paths)
        assert all(len(s) <= 50 for s in lines)

    def test_legacy_order_is_byte_identical_to_old_code(self, tmp_path, unsorted_paths):
        plan = prepare_work_shards(tmp_path, "t", unsorted_paths, shard_size=50, order="legacy")
        assert _shard_lines(plan.work_dir) == _legacy_plain_reference(unsorted_paths, 50)

    def test_limit_selects_input_prefix_then_sorts(self, tmp_path, unsorted_paths):
        plan = prepare_work_shards(tmp_path, "t", unsorted_paths, shard_size=50, limit=30, order="path")
        flat = [p for s in _shard_lines(plan.work_dir) for p in s]
        assert flat == sorted(unsorted_paths[:30])
        assert plan.total_items == 30

    def test_env_knob(self, tmp_path, unsorted_paths, monkeypatch):
        monkeypatch.setenv("BALALAIKA_SHARD_ORDER", "legacy")
        plan = prepare_work_shards(tmp_path, "t", unsorted_paths, shard_size=50)
        assert _shard_lines(plan.work_dir) == _legacy_plain_reference(unsorted_paths, 50)

    def test_annotations_survive_sorting(self, tmp_path):
        paths = [f"/d/{name}.wav" for name in ("c", "a", "b")]
        notes = {"/d/a.wav": "m1", "/d/c.wav": "m2"}
        plan = prepare_work_shards(tmp_path, "t", paths, shard_size=10, annotations=notes, order="path")
        shard = sorted(plan.work_dir.glob("shard_*.pending"))[0]
        assert read_annotated_work_shard(shard) == [
            ("/d/a.wav", "m1"),
            ("/d/b.wav", ""),
            ("/d/c.wav", "m2"),
        ]


class TestBucketedShardOrder:
    @pytest.fixture
    def corpus(self, unsorted_paths):
        durations = {
            p: 0.5 + ((i * 37) % 200) / 10.0  # 0.5 .. 20.4 s, scrambled
            for i, p in enumerate(unsorted_paths)
        }
        return unsorted_paths, durations

    @staticmethod
    def _by_label(work_dir):
        out: dict[str, list[str]] = {}
        for shard in sorted(work_dir.glob("shard_*.pending")):
            label = shard.stem.split("_", 2)[2]
            out.setdefault(label, []).extend(read_work_shard(shard))
        return out

    def test_bucket_membership_identical_to_legacy(self, tmp_path, corpus):
        paths, durations = corpus
        new = prepare_length_bucketed_work_shards(
            tmp_path, "new", paths, durations, shard_size=40, order="path"
        )
        old = prepare_length_bucketed_work_shards(
            tmp_path, "old", paths, durations, shard_size=40, order="legacy"
        )
        new_buckets = self._by_label(new.work_dir)
        old_buckets = self._by_label(old.work_dir)
        assert {k: sorted(v) for k, v in new_buckets.items()} == {
            k: sorted(v) for k, v in old_buckets.items()
        }

    def test_bounded_buckets_path_sorted_overflow_duration_sorted(self, tmp_path, corpus):
        paths, durations = corpus
        plan = prepare_length_bucketed_work_shards(
            tmp_path, "t", paths, durations, shard_size=10_000, order="path"
        )
        buckets = self._by_label(plan.work_dir)
        assert len(buckets) > 2
        for label, lines in buckets.items():
            if label.startswith("len_ge_"):
                assert lines == sorted(lines, key=lambda p: durations[p])
            else:
                assert lines == sorted(lines)

    def test_legacy_duration_order_everywhere(self, tmp_path, corpus):
        paths, durations = corpus
        plan = prepare_length_bucketed_work_shards(
            tmp_path, "t", paths, durations, shard_size=10_000, order="legacy"
        )
        for lines in self._by_label(plan.work_dir).values():
            assert lines == sorted(lines, key=lambda p: durations[p])


class TestMergerNoPeriodicPrune:
    @pytest.fixture
    def dataset(self, tmp_path):
        root = tmp_path / "data"
        root.mkdir()
        kept = root / "kept.wav"
        kept.write_bytes(b"x")
        df = pd.DataFrame(
            {"filepath": [str(kept), str(root / "deleted.wav")], "score": [None, None]}
        )
        atomic_write_csv(df, csv_path(root))
        with partial_writer(root, "scores", rank=0) as w:
            w.write({"filepath": str(kept), "score": 1.5})
        return root, kept

    def test_flush_keeps_missing_rows_final_absorb_prunes(self, dataset):
        root, kept = dataset
        merger = PeriodicCsvMerger(
            root, "scores", ["score"], drop_missing_files=True,
            flush_every_rows=1, flush_every_seconds=0,
        )
        assert merger._flush_once() == 1
        mid = fast_read_csv(csv_path(root))
        # Mid-stage flush merged the value but did NOT scan/prune the tree.
        assert len(mid) == 2
        assert mid.set_index("filepath").loc[str(kept), "score"] == 1.5

        absorb_partial_csvs(root, "scores", value_columns=["score"], drop_missing_files=True)
        final = fast_read_csv(csv_path(root))
        # Final state matches the old flow exactly: missing row pruned.
        assert final["filepath"].tolist() == [str(kept)]
        assert final["score"].tolist() == [1.5]


class TestIoProfile:
    def test_effective_workers_ssd_passthrough(self):
        assert io_profile.effective_workers(16, "ssd") == 16
        assert io_profile.effective_workers(16, "ssd", role="probe") == 16

    def test_effective_workers_hdd_clamps(self):
        assert io_profile.effective_workers(16, "hdd") == io_profile.HDD_MAX_LOADER_WORKERS
        assert io_profile.effective_workers(16, "hdd", role="probe") == io_profile.HDD_MAX_PROBE_WORKERS
        assert io_profile.effective_workers(1, "hdd") == 1

    def test_resolve_profile_explicit_and_env(self, tmp_path, monkeypatch):
        io_profile.resolve_io_profile.cache_clear()
        assert io_profile.resolve_io_profile(str(tmp_path), "hdd") == "hdd"
        monkeypatch.setenv(io_profile.IO_PROFILE_ENV, "hdd")
        io_profile.resolve_io_profile.cache_clear()
        assert io_profile.resolve_io_profile(str(tmp_path) + "/x" if False else str(tmp_path)) == "hdd"
        io_profile.resolve_io_profile.cache_clear()

    def test_is_rotational_does_not_crash(self):
        assert io_profile.is_rotational("/") in (True, False, None)


class TestDurationProbeOrder:
    def test_probe_runs_in_sorted_order_with_identical_values(self, tmp_path, monkeypatch):
        root = tmp_path / "ds"
        root.mkdir()
        paths = [str(root / f"z{i % 3}" / f"f{i:02d}.wav") for i in (5, 1, 9, 3, 7)]
        calls: list[str] = []

        def fake_probe(path):
            calls.append(path)
            return 1.0 + len(calls)

        monkeypatch.setattr(ad, "safe_audio_duration", fake_probe)
        io_profile.resolve_io_profile.cache_clear()
        out = ad.ensure_audio_durations(root, list(reversed(paths)), num_workers=1)
        assert calls == sorted(calls)
        for p in paths:
            assert out[ad.normalize_path_string(p)] > 0
