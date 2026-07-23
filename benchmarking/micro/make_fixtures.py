"""Generate synthetic fixtures for csv_manager micro-benchmarks.

Creates, under --root (default cache/bench_fixtures):

* ``tree/`` — ``n_real`` empty audio files spread over nested dirs
  (playlist_id/podcast_id layout like the real dataset).
* ``state/balalaika.csv`` — ``n_rows`` rows shaped like the production CSV
  (filepath + score columns, ~30% NaN, mixed-dtype loudness column). The
  first ``n_real`` rows point at files that really exist in ``tree/``.
* ``state/crest_part_{0..3}.csv`` — worker partials with ``part_rows`` rows
  each, overlapping the main CSV's filepaths.

Deterministic via --seed so before/after benchmark runs see identical data.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

AUDIO_EXT = ".wav"


def make_tree(tree_root: Path, n_real: int, files_per_dir: int = 500) -> list[str]:
    paths: list[str] = []
    made = 0
    playlist = 0
    while made < n_real:
        d = tree_root / f"playlist_{playlist:04d}" / f"podcast_{playlist:04d}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(min(files_per_dir, n_real - made)):
            p = d / f"chunk_{made + i:07d}{AUDIO_EXT}"
            p.touch()
            paths.append(str(p.resolve()))
        made += min(files_per_dir, n_real - made)
        playlist += 1
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("cache/bench_fixtures"))
    ap.add_argument("--n-rows", type=int, default=2_000_000)
    ap.add_argument("--n-real", type=int, default=50_000)
    ap.add_argument("--part-rows", type=int, default=50_000)
    ap.add_argument("--n-parts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = args.root.resolve()
    tree = root / "tree"
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)

    if tree.exists() and sum(1 for _ in tree.rglob(f"*{AUDIO_EXT}")) >= args.n_real:
        real_paths = sorted(str(p.resolve()) for p in tree.rglob(f"*{AUDIO_EXT}"))[: args.n_real]
        print(f"tree/ already has {len(real_paths)} files; reusing")
    else:
        real_paths = make_tree(tree, args.n_real)
        print(f"created {len(real_paths)} empty audio files under {tree}")

    n_fake = args.n_rows - len(real_paths)
    fake_paths = [
        str(tree / f"playlist_{9000 + i % 200:04d}" / f"podcast_{i % 200:04d}" / f"missing_{i:08d}.wav")
        for i in range(n_fake)
    ]
    filepaths = real_paths + fake_paths

    n = len(filepaths)
    crest = rng.uniform(1.0, 20.0, size=n)
    crest[rng.random(n) < 0.3] = np.nan
    duration = rng.uniform(0.5, 15.0, size=n).round(4)
    mos = rng.uniform(1.0, 5.0, size=n).round(4)
    mos[rng.random(n) < 0.5] = np.nan
    loud = np.where(rng.random(n) < 0.5, "True", "")

    df = pd.DataFrame(
        {
            "filepath": filepaths,
            "speaker_id": [f"spk_{i % 4000}" for i in range(n)],
            "start": rng.uniform(0, 600, size=n).round(3),
            "end": rng.uniform(600, 1200, size=n).round(3),
            "total_duration": duration,
            "playlist_id": [f"playlist_{i % 200:04d}" for i in range(n)],
            "podcast_id": [f"podcast_{i % 200:04d}" for i in range(n)],
            "crest_factor": crest,
            "loudness_normalized": loud,
            "DistillMOS": mos,
        }
    )
    df.to_csv(state / "balalaika.csv", index=False)
    print(f"wrote {n} rows to {state / 'balalaika.csv'}")

    # Partials overlap the tail of the main CSV plus some brand-new rows.
    for part in range(args.n_parts):
        lo = part * args.part_rows
        sel = [filepaths[(i * 7 + lo) % n] for i in range(args.part_rows)]
        pdf = pd.DataFrame(
            {
                "filepath": sel,
                "crest_factor": rng.uniform(1.0, 20.0, size=args.part_rows).round(4),
                "total_duration": rng.uniform(0.5, 15.0, size=args.part_rows).round(4),
                "duration_s": rng.uniform(0.5, 15.0, size=args.part_rows).round(4),
                "deleted": rng.random(args.part_rows) < 0.02,
            }
        )
        pdf.to_csv(state / f"crest_part_{part}.csv", index=False)
    print(f"wrote {args.n_parts} partials x {args.part_rows} rows")


if __name__ == "__main__":
    main()
