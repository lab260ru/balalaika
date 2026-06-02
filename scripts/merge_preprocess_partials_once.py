from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pandas as pd


BASE_COLUMNS = (
    "filepath",
    "speaker_id",
    "start",
    "end",
    "total_duration",
    "playlist_id",
    "podcast_id",
    "silence_percent",
    "max_silence_duration",
    "is_single_speaker",
    "crest_factor",
    "loudness_normalized",
    "music_prob",
    "DistillMOS",
    "antispoof_score",
    "antispoof_generated_prob",
    "denoised",
)


def resolve_path(value: object) -> str:
    return str(Path(str(value)).resolve())


def normalize_filepath(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty and "filepath" in df.columns:
        df = df.copy()
        df["filepath"] = df["filepath"].astype(str).map(resolve_path)
    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    base = [column for column in BASE_COLUMNS if column in df.columns]
    extras = [column for column in df.columns if column not in base]
    return df[base + extras]


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    bak = Path(str(path) + ".bak")
    if path.exists():
        shutil.copy2(path, bak)
    df.to_csv(tmp, index=False)
    with open(tmp, "rb") as file:
        os.fsync(file.fileno())
    os.replace(tmp, path)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: merge_preprocess_partials_once.py DATASET_DIR", file=sys.stderr)
        return 2

    dataset_dir = Path(sys.argv[1])
    main_csv = dataset_dir / "balalaika.csv"
    part_paths = [dataset_dir / f"preprocess_part_{idx}.csv" for idx in range(4)]

    missing = [str(path) for path in [main_csv, *part_paths] if not path.exists()]
    if missing:
        print("missing files:", *missing, sep="\n", file=sys.stderr)
        return 1

    print(f"reading main: {main_csv}")
    main_df = normalize_filepath(pd.read_csv(main_csv, low_memory=False))
    print(f"main rows: {len(main_df):,}; columns: {len(main_df.columns):,}")

    partial_frames = []
    for path in part_paths:
        print(f"reading partial: {path}")
        part_df = pd.read_csv(path, low_memory=False)
        print(f"  rows: {len(part_df):,}; columns: {len(part_df.columns):,}")
        if not part_df.empty:
            partial_frames.append(part_df)

    if not partial_frames:
        print("no partial rows to merge")
        return 0

    partials = normalize_filepath(pd.concat(partial_frames, ignore_index=True))
    if "filepath" not in partials.columns:
        print("partials do not contain filepath column", file=sys.stderr)
        return 1

    before_dedup = len(partials)
    partials = partials.drop_duplicates(subset="filepath", keep="last")
    value_columns = [column for column in partials.columns if column != "filepath"]
    print(
        f"partial rows: {before_dedup:,}; unique filepaths: {len(partials):,}; "
        f"merge columns: {', '.join(value_columns)}"
    )

    updated = main_df.drop(columns=value_columns, errors="ignore")
    updated = updated.merge(partials[["filepath", *value_columns]], on="filepath", how="outer")
    updated = reorder_columns(updated)

    print(f"merged rows: {len(updated):,}; columns: {len(updated.columns):,}")
    print(f"writing atomically: {main_csv}")
    atomic_write_csv(updated, main_csv)
    print(f"done; backup: {main_csv}.bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
