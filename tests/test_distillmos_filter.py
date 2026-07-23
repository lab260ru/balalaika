"""Behavior + equivalence tests for the DistillMOS deletion phase (stage 5.5).

The deletion phase was changed to shard only deletion *candidates*
(``DistillMOS`` < threshold) to the workers instead of every scored file, so
the kept >=95% of files are never touched on the (slow) HDD. These tests prove
that the externally observable contract is preserved:

  * the *set of deleted files* is identical to the old all-rows behaviour;
  * ``filter_summary.csv`` counts (files_in/out/deleted) are identical, and the
    hour figures are identical whenever ``total_duration`` is populated (the
    production-normal case, since durations are written by earlier stages);
  * the final ``balalaika.csv`` state is identical for those same rows.

They also measure the I/O win: how many per-file duration probes the workers
issue before vs. after the change.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import src.separation.distillmos_filter as dmf
from src.separation.distillmos_filter import deletion_candidates, run_deletion_workers


# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #
def _write_wav(path: Path, duration_s: float, sr: int = 16000) -> None:
    import numpy as _np
    import soundfile as _sf

    path.parent.mkdir(parents=True, exist_ok=True)
    _sf.write(str(path), _np.zeros(int(duration_s * sr), dtype="float32"), sr)


def make_dataset(tmp_path, rows, *, real_wavs=False):
    """Create balalaika.csv + chunk files.

    ``rows`` items are ``(mos, total_duration)`` or
    ``(mos, total_duration, exists_on_disk[, real_duration_s])``.
    With ``real_wavs=True`` real (silent) wavs are written so the duration
    probe returns a real value; otherwise tiny byte stubs are written.
    """
    paths = []
    csv_rows = []
    for i, row in enumerate(rows):
        mos, csv_dur = row[0], row[1]
        exists = row[2] if len(row) > 2 else True
        real_dur = row[3] if len(row) > 3 else (csv_dur if pd.notna(csv_dur) else 1.0)
        p = tmp_path / "pl" / "pod" / f"chunk_{i}.wav"
        if exists:
            if real_wavs:
                _write_wav(p, real_dur or 1.0)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"RIFF0000WAVE")
        paths.append(str(p.resolve()))
        csv_rows.append(
            {"filepath": str(p.resolve()), "DistillMOS": mos, "total_duration": csv_dur}
        )
    pd.DataFrame(csv_rows).to_parquet(tmp_path / "balalaika.parquet", index=False)
    return paths


def _write_config(tmp_path, threshold=3.0, num_workers=2) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "separation:\n"
        f"  podcasts_path: {tmp_path}\n"
        "  distillmos_filter:\n"
        f"    threshold: {threshold}\n"
        f"    num_workers: {num_workers}\n"
        "  runtime:\n"
        "    audio_paths_source: csv\n"
    )
    return cfg


def _run_main(tmp_path, threshold=3.0, num_workers=2):
    cfg = _write_config(tmp_path, threshold, num_workers)
    args = argparse.Namespace(
        config_path=str(cfg),
        log_dir=str(tmp_path / "logs"),
        manual=False,
        threshold=None,
    )
    dmf.main(args)


def _read_summary(tmp_path) -> dict:
    import json

    fs = pd.read_csv(tmp_path / "filter_summary.csv").iloc[-1]
    return {
        "files_in": int(fs.files_in),
        "files_out": int(fs.files_out),
        "hours_in": float(fs.hours_in),
        "hours_out": float(fs.hours_out),
        "files_deleted": int(json.loads(fs.params)["deleted"]),
    }


# --------------------------------------------------------------------------- #
# Reference: the OLD all-rows audit, recomputed by hand from the dataframe.
# --------------------------------------------------------------------------- #
def _old_reference(rows, threshold):
    """Replicate the old end-state for a fully-populated-duration dataset.

    Returns (deleted_indices, audit_dict). Files missing on disk are pruned
    before the (old) workers ran, so they are never candidates -- exactly as in
    the new code. With every ``total_duration`` present the old probe-after-
    delete quirk never fires, so this is a faithful reference for those cases.
    """
    surviving = [
        (i, row[0], row[1])
        for i, row in enumerate(rows)
        if (len(row) <= 2 or row[2])  # exists on disk
    ]
    scored = [(i, mos, dur) for (i, mos, dur) in surviving if pd.notna(mos)]
    deleted = [i for (i, mos, dur) in scored if mos < threshold]
    kept = [i for (i, mos, dur) in scored if not (mos < threshold)]
    dur = {i: (0.0 if pd.isna(d) else float(d)) for (i, m, d) in scored}
    files_in = len(scored)
    files_deleted = len(deleted)
    hours_in = sum(dur[i] for (i, m, d) in scored) / 3600.0
    hours_out = sum(dur[i] for i in kept) / 3600.0
    return set(deleted), {
        "files_in": files_in,
        "files_out": files_in - files_deleted,
        "hours_in": round(hours_in, 4),
        "hours_out": round(hours_out, 4),
        "files_deleted": files_deleted,
    }


# --------------------------------------------------------------------------- #
# Behaviour-preservation tests (candidate-only sharding).
# --------------------------------------------------------------------------- #
def test_deletion_below_threshold(tmp_path):
    rows = [
        (2.0, 5.0),     # below threshold -> deleted
        (3.5, 6.0),     # kept
        (np.nan, 7.0),  # unscored -> untouched, no partial row
        (2.9, 8.0),     # below -> deleted
        (4.9, 9.0),     # kept
    ]
    paths = make_dataset(tmp_path, rows)

    processed, deleted, errors = run_deletion_workers(
        tmp_path, threshold=3.0, num_workers=2, config_path=None
    )

    assert errors == 0
    assert deleted == 2
    # Only the two candidates are processed now (kept files are never touched).
    assert processed == 2

    assert not os.path.exists(paths[0])
    assert os.path.exists(paths[1])
    assert os.path.exists(paths[2])  # unscored survives
    assert not os.path.exists(paths[3])
    assert os.path.exists(paths[4])

    partials = sorted(tmp_path.glob("distillmos_filter_part_*.csv"))
    merged = pd.concat([pd.read_csv(p) for p in partials if p.stat().st_size > 0])
    # Candidate-only partial: just the two deleted files.
    assert len(merged) == 2
    by_path = merged.set_index("filepath")
    assert bool(by_path.loc[paths[0], "deleted"]) is True
    assert bool(by_path.loc[paths[3], "deleted"]) is True
    # Kept files never get a partial row.
    assert paths[1] not in by_path.index
    assert paths[4] not in by_path.index


def test_no_scored_files(tmp_path):
    rows = [(np.nan, 5.0), (np.nan, 5.0)]
    make_dataset(tmp_path, rows)
    processed, deleted, errors = run_deletion_workers(
        tmp_path, threshold=3.0, num_workers=2, config_path=None
    )
    assert (processed, deleted, errors) == (0, 0, 0)


def test_no_candidates(tmp_path):
    # All scored files are above threshold -> nothing to delete, no probes.
    rows = [(3.5, 5.0), (4.0, 6.0), (np.nan, 7.0)]
    paths = make_dataset(tmp_path, rows)
    processed, deleted, errors = run_deletion_workers(
        tmp_path, threshold=3.0, num_workers=2, config_path=None
    )
    assert (processed, deleted, errors) == (0, 0, 0)
    assert all(os.path.exists(p) for p in paths)
    assert not sorted(tmp_path.glob("distillmos_filter_part_*.csv"))


def test_threshold_boundary_is_kept(tmp_path):
    # mos == threshold must be KEPT (strict <).
    rows = [(3.0, 4.0), (2.999, 5.0)]
    paths = make_dataset(tmp_path, rows)
    _, deleted, _ = run_deletion_workers(
        tmp_path, threshold=3.0, num_workers=1, config_path=None
    )
    assert deleted == 1
    assert os.path.exists(paths[0])       # == threshold kept
    assert not os.path.exists(paths[1])   # below deleted


def test_deletion_candidates_helper(tmp_path):
    df = pd.DataFrame(
        {
            "filepath": ["a", "b", "c", "d"],
            "DistillMOS": [2.0, 3.0, np.nan, 2.9],
            "total_duration": [1, 2, 3, 4],
        }
    )
    cand = deletion_candidates(df, 3.0)
    assert set(cand["filepath"]) == {"a", "d"}  # 3.0 not < 3.0, NaN excluded


# --------------------------------------------------------------------------- #
# End-to-end equivalence: new main() vs the old all-rows reference.
# --------------------------------------------------------------------------- #
def test_end_to_end_matches_old_all_durations_present(tmp_path):
    """Production-normal case: every total_duration present, mixed edge cases.

    Asserts deleted set, audit counts AND audit hours, plus final CSV state.
    """
    rows = [
        (2.0, 5.0),     # 0 candidate -> deleted
        (3.5, 6.0),     # 1 kept
        (np.nan, 7.0),  # 2 unscored -> survives
        (2.9, 8.0),     # 3 candidate -> deleted
        (4.9, 9.0),     # 4 kept
        (3.0, 4.0),     # 5 == threshold -> kept
        (1.2, 2.0),     # 6 candidate -> deleted
    ]
    paths = make_dataset(tmp_path, rows, real_wavs=True)
    ref_deleted, ref_audit = _old_reference(rows, 3.0)

    _run_main(tmp_path, threshold=3.0, num_workers=2)

    got_deleted = {i for i, p in enumerate(paths) if not os.path.exists(p)}
    assert got_deleted == ref_deleted

    audit = _read_summary(tmp_path)
    assert audit["files_in"] == ref_audit["files_in"]
    assert audit["files_out"] == ref_audit["files_out"]
    assert audit["files_deleted"] == ref_audit["files_deleted"]
    # Hours are byte-identical to old when durations are populated.
    assert audit["hours_in"] == ref_audit["hours_in"]
    assert audit["hours_out"] == ref_audit["hours_out"]

    final = pd.read_parquet(tmp_path / "balalaika.parquet").set_index("filepath")
    # Deleted rows are pruned; kept/unscored rows survive with values intact.
    assert paths[0] not in final.index
    assert paths[3] not in final.index
    assert paths[6] not in final.index
    assert final.loc[paths[1], "DistillMOS"] == 3.5
    assert final.loc[paths[1], "total_duration"] == 6.0
    assert final.loc[paths[4], "total_duration"] == 9.0
    assert final.loc[paths[5], "DistillMOS"] == 3.0
    assert pd.isna(final.loc[paths[2], "DistillMOS"])  # unscored stays


def test_end_to_end_missing_on_disk_pruned_not_deleted_count(tmp_path):
    """A scored file missing on disk is pruned before sharding (old + new).

    It must not be counted as a deletion and must not appear in the final CSV.
    """
    rows = [
        (2.0, 5.0, True),    # 0 candidate -> deleted
        (4.0, 6.0, True),    # 1 kept
        (1.5, 3.0, False),   # 2 candidate but missing on disk -> pruned
    ]
    paths = make_dataset(tmp_path, rows, real_wavs=True)
    ref_deleted, ref_audit = _old_reference(rows, 3.0)

    _run_main(tmp_path, threshold=3.0, num_workers=2)

    got_deleted = {i for i, p in enumerate(paths) if not os.path.exists(p)}
    assert 0 in got_deleted
    audit = _read_summary(tmp_path)
    assert audit["files_in"] == ref_audit["files_in"] == 2
    assert audit["files_deleted"] == ref_audit["files_deleted"] == 1
    assert audit["files_out"] == ref_audit["files_out"] == 1

    final = set(pd.read_parquet(tmp_path / "balalaika.parquet")["filepath"])
    assert paths[0] not in final  # deleted candidate pruned
    assert paths[2] not in final  # missing-on-disk pruned
    assert paths[1] in final       # kept survives


def test_end_to_end_missing_duration_kept_file_not_probed(tmp_path):
    """Documented contract change: a KEPT file with no total_duration is NOT
    re-probed (that is the wasted I/O we removed). Counts stay identical to
    old; the kept file's missing duration is simply not backfilled.
    """
    rows = [
        (2.0, 5.0),       # 0 candidate -> deleted (has duration)
        (4.0, np.nan),    # 1 kept, missing duration -> NOT probed/backfilled
    ]
    paths = make_dataset(tmp_path, rows, real_wavs=True)

    _run_main(tmp_path, threshold=3.0, num_workers=1)

    assert not os.path.exists(paths[0])
    assert os.path.exists(paths[1])

    audit = _read_summary(tmp_path)
    assert audit["files_in"] == 2
    assert audit["files_deleted"] == 1
    assert audit["files_out"] == 1

    final = pd.read_parquet(tmp_path / "balalaika.parquet").set_index("filepath")
    # Kept file's duration stays missing (no re-probe).
    assert pd.isna(final.loc[paths[1], "total_duration"])


# --------------------------------------------------------------------------- #
# Symlinked dataset root: the deletion audit must count files from the
# candidate partials, not by matching resolve_path()-resolved partial paths
# against the verbatim (symlink-form) baseline paths. On a symlinked root the
# two path forms differ, so a baseline-mask audit reports 0 deleted / 0 hours.
# --------------------------------------------------------------------------- #
def make_symlinked_dataset(tmp_path, rows):
    """Like ``make_dataset`` but with a symlinked dataset root.

    Real wavs live under ``tmp_path/real_data``; ``tmp_path/data`` is a symlink
    to it and is used as ``podcasts_path``. balalaika.csv stores the
    *symlink-form* absolute paths (``.../data/...``) exactly as ``load_main_csv``
    would keep them (it trusts absolute paths verbatim). A worker deletes
    through the symlink but writes the partial with ``resolve_path`` (the
    ``.../real_data/...`` form), so the two path forms diverge.
    """
    real_root = tmp_path / "real_data"
    real_root.mkdir(parents=True, exist_ok=True)
    link_root = tmp_path / "data"
    link_root.symlink_to(real_root, target_is_directory=True)

    link_paths = []
    csv_rows = []
    for i, row in enumerate(rows):
        mos, csv_dur, real_dur = row[0], row[1], row[2]
        rel = Path("pl") / "pod" / f"chunk_{i}.wav"
        _write_wav(real_root / rel, real_dur)
        link_path = str(link_root / rel)  # symlink-form absolute path, verbatim
        link_paths.append(link_path)
        csv_rows.append(
            {"filepath": link_path, "DistillMOS": mos, "total_duration": csv_dur}
        )
    pd.DataFrame(csv_rows).to_parquet(link_root / "balalaika.parquet", index=False)
    return link_root, link_paths


def test_symlinked_root_audit_counts_deleted_files_and_hours(tmp_path):
    """Symlinked root: deleted/hours must come from the partials, not a
    baseline path-mask. One candidate is deleted; its duration is 7200 s = 2 h.
    A baseline-mask audit would report 0 deleted / 0.0 hours.
    """
    rows = [
        (2.0, 7200.0, 7200.0),  # 0 candidate -> deleted, 2.0 hours
        (4.0, 3600.0, 3600.0),  # 1 kept
    ]
    link_root, link_paths = make_symlinked_dataset(tmp_path, rows)

    _run_main(link_root, threshold=3.0, num_workers=1)

    assert not os.path.exists(link_paths[0])  # deleted through the symlink
    assert os.path.exists(link_paths[1])

    fs = pd.read_csv(link_root / "filter_summary.csv").iloc[-1]
    audit = _read_summary(link_root)
    assert audit["files_deleted"] == 1
    # hours_removed is recorded as hours_in - hours_out by record_stage_summary.
    assert float(fs.hours_removed) == 2.0
    assert audit["hours_in"] - audit["hours_out"] == 2.0


def test_deleted_candidate_missing_csv_duration_audit_stays_consistent(tmp_path):
    """A DELETED candidate with a missing (NaN) ``total_duration`` in the
    baseline CSV but a real probed ``duration_s`` must not drive ``hours_out``
    negative.

    ``hours_in`` is derived from the baseline's ``total_duration`` while
    ``hours_deleted`` is derived from the partials' probed ``duration_s``; when
    the deleted candidate's baseline duration is missing it contributes 0 to
    ``hours_in`` yet its probed 2.0 h is subtracted as ``hours_deleted`` ->
    ``hours_out`` would be -1.0 h with the mismatched-source audit. The audit
    must be internally consistent from a single duration source:
    ``hours_out >= 0`` and ``hours_in - hours_out == hours_deleted``.
    """
    rows = [
        # mos, csv_total_duration, exists, real_duration_s
        (2.0, np.nan, True, 7200.0),  # 0 candidate -> deleted; CSV dur missing, probed 2.0 h
        (4.0, 3600.0, True, 3600.0),  # 1 kept; 1.0 h
    ]
    paths = make_dataset(tmp_path, rows, real_wavs=True)

    _run_main(tmp_path, threshold=3.0, num_workers=1)

    assert not os.path.exists(paths[0])
    assert os.path.exists(paths[1])

    fs = pd.read_csv(tmp_path / "filter_summary.csv").iloc[-1]
    audit = _read_summary(tmp_path)
    assert audit["files_deleted"] == 1
    assert audit["files_in"] == 2
    assert audit["files_out"] == 1
    # Probed duration of the deleted candidate is 2.0 h.
    hours_removed = audit["hours_in"] - audit["hours_out"]
    assert hours_removed == 2.0
    assert float(fs.hours_removed) == 2.0
    # The crux: hours must stay internally consistent and never negative.
    assert audit["hours_out"] >= 0.0
    # The single kept file is 1.0 h; the audit must reflect that on the way out.
    assert audit["hours_out"] == 1.0


def test_non_symlinked_root_audit_unchanged(tmp_path):
    """Same scenario without a symlink: the audit numbers are identical to the
    symlinked case (1 deleted, 2.0 h removed). Guards against the fix changing
    the normal-root behaviour.
    """
    rows = [
        (2.0, 7200.0),  # 0 candidate -> deleted, 2.0 hours
        (4.0, 3600.0),  # 1 kept
    ]
    paths = make_dataset(tmp_path, rows, real_wavs=True)

    _run_main(tmp_path, threshold=3.0, num_workers=1)

    assert not os.path.exists(paths[0])
    assert os.path.exists(paths[1])

    fs = pd.read_csv(tmp_path / "filter_summary.csv").iloc[-1]
    audit = _read_summary(tmp_path)
    assert audit["files_deleted"] == 1
    assert float(fs.hours_removed) == 2.0
    assert audit["hours_in"] - audit["hours_out"] == 2.0


# --------------------------------------------------------------------------- #
# I/O win: count per-file duration probes before vs. after.
# --------------------------------------------------------------------------- #
def test_probe_count_scales_with_candidates(tmp_path, monkeypatch):
    """5k scored rows, 2% candidates, 30% missing durations.

    Old behaviour probed every scored file whose CSV row lacked a duration
    (~30% of 5000 = ~1500 probes). New behaviour probes only *candidates*
    lacking a duration. We instrument ``safe_audio_duration`` and assert the
    probe count collapses from the old ~1500 to a tiny candidate-only number.
    """
    rng = np.random.default_rng(0)
    n = 5000
    n_cand = int(n * 0.02)
    cand_idx = rng.choice(n, size=n_cand, replace=False)
    mos = np.full(n, 3.5)       # everything safely above threshold
    mos[cand_idx] = 2.0         # candidates below threshold 3.0
    missing = rng.random(n) < 0.30  # 30% of rows have a missing total_duration

    rows = [
        (float(mos[i]), (np.nan if missing[i] else 4.0))
        for i in range(n)
    ]
    make_dataset(tmp_path, rows, real_wavs=False)

    # OLD design would probe every scored file missing a duration.
    old_probes = int(missing.sum())
    # NEW design probes only candidates missing a duration.
    new_expected_probes = int(missing[cand_idx].sum())

    calls = {"n": 0}

    def counting_probe(path):
        calls["n"] += 1
        return 1.0

    monkeypatch.setattr(dmf, "safe_audio_duration", counting_probe)

    # Run the worker inline (no child process) so the patched symbol and the
    # in-process counter are observed. This is exactly what
    # run_deletion_workers does for one shard.
    from multiprocessing import Value

    df = dmf.load_main_csv(tmp_path)
    candidates = deletion_candidates(df, 3.0)
    # Same (path, mos, duration) tuples run_deletion_workers ships to workers.
    durs = pd.to_numeric(candidates["total_duration"], errors="coerce").fillna(0.0)
    items = list(
        zip(
            candidates["filepath"].astype(str),
            pd.to_numeric(candidates[dmf.COLUMN], errors="coerce").astype(float),
            durs.astype(float),
        )
    )
    assert len(items) == n_cand

    processed = Value("i", 0)
    deleted = Value("i", 0)
    errors = Value("i", 0)
    dmf.run_worker(0, items, 3.0, str(tmp_path), processed, deleted, errors)

    assert calls["n"] == new_expected_probes
    assert new_expected_probes <= n_cand
    assert old_probes > 10 * max(1, new_expected_probes)
    assert deleted.value == n_cand
    print(
        f"\n[probe-count] old would probe {old_probes} files; "
        f"new probes {calls['n']} (candidates missing duration); "
        f"candidates={n_cand}/{n}"
    )
