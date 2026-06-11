"""End-to-end behavior test for the DistillMOS deletion phase (stage 5.5)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.separation.distillmos_filter import run_deletion_workers


def make_dataset(tmp_path, rows):
    paths = []
    for i, _ in enumerate(rows):
        p = tmp_path / "pl" / "pod" / f"chunk_{i}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"RIFF0000WAVE")
        paths.append(str(p.resolve()))
    df = pd.DataFrame(
        {
            "filepath": paths,
            "DistillMOS": [r[0] for r in rows],
            "total_duration": [r[1] for r in rows],
        }
    )
    df.to_csv(tmp_path / "balalaika.csv", index=False)
    return paths


def test_deletion_below_threshold(tmp_path):
    rows = [
        (2.0, 5.0),   # below threshold -> deleted
        (3.5, 6.0),   # kept
        (np.nan, 7.0),  # unscored -> untouched, no partial row
        (2.9, 8.0),   # below -> deleted
        (4.9, 9.0),   # kept
    ]
    paths = make_dataset(tmp_path, rows)

    processed, deleted, errors = run_deletion_workers(
        tmp_path, threshold=3.0, num_workers=2, config_path=None
    )

    assert errors == 0
    assert deleted == 2
    assert processed == 4  # all scored files get an audit row

    import os

    assert not os.path.exists(paths[0])
    assert os.path.exists(paths[1])
    assert os.path.exists(paths[2])  # unscored survives
    assert not os.path.exists(paths[3])
    assert os.path.exists(paths[4])

    partials = sorted(tmp_path.glob("distillmos_filter_part_*.csv"))
    if not partials:  # prefix defined in module
        partials = sorted(tmp_path.glob("*_part_*.csv"))
    merged = pd.concat([pd.read_csv(p) for p in partials if p.stat().st_size > 0])
    assert len(merged) == 4
    by_path = merged.set_index("filepath")
    assert bool(by_path.loc[paths[0], "deleted"]) is True
    assert bool(by_path.loc[paths[1], "deleted"]) is False
    # duration carried from the CSV, not re-probed
    assert by_path.loc[paths[4], "total_duration"] == 9.0


def test_no_scored_files(tmp_path):
    rows = [(np.nan, 5.0), (np.nan, 5.0)]
    make_dataset(tmp_path, rows)
    processed, deleted, errors = run_deletion_workers(
        tmp_path, threshold=3.0, num_workers=2, config_path=None
    )
    assert (processed, deleted, errors) == (0, 0, 0)
