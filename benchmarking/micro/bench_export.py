"""Micro-benchmarks for the export/report hot loops touched in the §9.8 pass:

  * to_webdataset metadata sanitisation: old per-cell pd.isna/isinstance/hasattr
    loop vs. the vectorised _sanitize_records (CPU).
  * report.py missing-values scan: old pure-Python csv.DictReader loop vs. the
    chunked-pandas path; old decode-everything parquet scan vs. the
    statistics-projected path.

All compare measured wall (and prove identical counts / records).

    python -m benchmarking.micro.bench_export --label check --rows 200000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")


def _synth_meta_df(rows: int, extra_numeric: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    data = {
        "speaker_id": rng.integers(0, 8, size=rows),
        "total_duration": np.where(
            rng.random(rows) < 0.05, np.nan, rng.uniform(1, 15, size=rows).round(4)
        ),
        "crest_factor": rng.uniform(1, 5, size=rows).round(4),
        "DistillMOS": np.where(
            rng.random(rows) < 0.1, np.nan, rng.uniform(1, 5, size=rows).round(3)
        ),
        "is_single_speaker": rng.random(rows) < 0.7,
        "playlist_id": rng.integers(0, 1000, size=rows),
    }
    # Production metadata frames carry many numeric columns; these are the ones
    # the optimized parquet scan skips decoding via row-group statistics.
    for c in range(extra_numeric):
        data[f"num{c}"] = np.where(
            rng.random(rows) < 0.1, np.nan, rng.uniform(0, 5, size=rows)
        )
    return pd.DataFrame(data)


def _old_sanitize(record: dict) -> dict:
    out = {}
    for k, v in record.items():
        k_str = str(k)
        if pd.isna(v) or (isinstance(v, float) and math.isnan(v)):
            out[k_str] = None
        elif isinstance(v, (pd.Timestamp, pd.Timedelta)):
            out[k_str] = str(v)
        elif hasattr(v, "item"):
            out[k_str] = v.item()
        else:
            out[k_str] = v
    return out


def bench_sanitize(rows: int):
    from src.to_webdataset import _sanitize_records

    df = _synth_meta_df(rows)

    t0 = time.perf_counter()
    old = [_old_sanitize(r) for r in df.to_dict("records")]
    old_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    new = _sanitize_records(df)
    new_s = time.perf_counter() - t0

    # prove identical JSON bytes on a sample
    same = all(
        json.dumps(o, ensure_ascii=False) == json.dumps(n, ensure_ascii=False)
        for o, n in zip(old[: min(2000, rows)], new[: min(2000, rows)])
    )
    print(
        f"[wds sanitize] rows={rows}  OLD per-cell={old_s:.3f}s  "
        f"NEW vectorised={new_s:.3f}s  speedup={old_s / new_s:.1f}x  identical={same}"
    )


def _old_csv_missing(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        missing = {c: 0 for c in columns}
        total = 0
        for row in reader:
            total += 1
            for c in columns:
                v = row.get(c)
                if v is None or str(v).strip() == "":
                    missing[c] += 1
    return missing, total


def bench_report_csv(rows: int, tmp: Path):
    from src.report import _read_csv_missing_summary

    df = _synth_meta_df(rows).astype(object)
    df["filepath"] = [f"/d/{i}.wav" for i in range(rows)]
    df["transcript"] = ["слово раз два" if i % 4 else "" for i in range(rows)]
    p = tmp / "balalaika.csv"
    df.to_csv(p, index=False)

    t0 = time.perf_counter()
    old_missing, old_total = _old_csv_missing(p)
    old_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    rows_out, total = _read_csv_missing_summary(p)
    new_s = time.perf_counter() - t0

    new_missing = {r["column"]: int(r["missing"]) for r in rows_out}
    same = (new_missing == old_missing) and (total == old_total)
    print(
        f"[report CSV missing] rows={rows}  OLD csv.DictReader={old_s:.3f}s  "
        f"NEW chunked-pandas={new_s:.3f}s  speedup={old_s / new_s:.1f}x  identical={same}"
    )


def _old_parquet_missing(path: Path):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    missing = {n: 0 for n in schema.names}
    total = 0
    for batch in pf.iter_batches(batch_size=100_000):
        total += batch.num_rows
        for idx, col in enumerate(batch.schema.names):
            arr = batch.column(idx)
            m = int(arr.null_count)
            ft = batch.schema.field(idx).type
            if pa.types.is_string(ft) or pa.types.is_large_string(ft):
                em = pc.equal(arr, "")
                m += int(pc.sum(pc.fill_null(em, False)).as_py() or 0)
            missing[col] += m
    return missing, total


def bench_report_parquet(rows: int, tmp: Path):
    from src.report import _read_parquet_missing_summary

    df = _synth_meta_df(rows, extra_numeric=16)
    df["filepath"] = [f"/d/{i}.wav" for i in range(rows)]
    df["transcript"] = ["слово раз два" if i % 4 else "" for i in range(rows)]
    p = tmp / "balalaika.parquet"
    df.to_parquet(p, engine="pyarrow", index=False)

    t0 = time.perf_counter()
    old_missing, old_total = _old_parquet_missing(p)
    old_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    rows_out, total = _read_parquet_missing_summary(p)
    new_s = time.perf_counter() - t0

    new_missing = {r["column"]: int(r["missing"]) for r in rows_out}
    same = (new_missing == old_missing) and (total == old_total)
    print(
        f"[report parquet missing] rows={rows}  OLD decode-all={old_s:.3f}s  "
        f"NEW stats+projection={new_s:.3f}s  speedup={old_s / new_s:.1f}x  identical={same}"
    )


def main() -> None:
    import tempfile

    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--rows", type=int, default=200_000)
    args = ap.parse_args()

    bench_sanitize(args.rows)
    with tempfile.TemporaryDirectory() as td:
        bench_report_csv(args.rows, Path(td))
        bench_report_parquet(args.rows, Path(td))


if __name__ == "__main__":
    main()
