"""Equivalence: vectorised missing-value counts (CSV + parquet) vs. the old
pure-Python logic. Empty-string-vs-NaN distinctions, whitespace-only cells,
literal "NA"/"NaN"/"null" (NOT missing), short rows, and numeric nulls are all
pinned identical to the original implementation.
"""
from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.report import _read_csv_missing_summary, _read_parquet_missing_summary


def _old_csv_logic(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        missing_counts = {column: 0 for column in columns}
        total_rows = 0
        for row in reader:
            total_rows += 1
            for column in columns:
                value = row.get(column)
                if value is None or str(value).strip() == "":
                    missing_counts[column] += 1
    return missing_counts, total_rows


def _rows_to_counts(rows):
    return {r["column"]: int(r["missing"]) for r in rows}


def test_csv_tricky_values(tmp_path):
    p = tmp_path / "balalaika.csv"
    p.write_text(
        "a,b,c\n"
        "1,,x\n"          # b empty
        "2, ,NA\n"        # b whitespace-only -> missing; c literal NA -> NOT missing
        "3,hello,\n"      # c empty
        "4,world\n"       # short row -> c absent (None) -> missing
        "null,NaN,end\n"  # literal null/NaN -> NOT missing
        "  ,tab,\t\n",    # a whitespace -> missing; c whitespace(tab) -> missing
        encoding="utf-8",
    )
    old_counts, old_rows = _old_csv_logic(p)
    rows, total = _read_csv_missing_summary(p)
    assert total == old_rows
    assert _rows_to_counts(rows) == old_counts


def test_csv_empty_dataset(tmp_path):
    p = tmp_path / "balalaika.csv"
    p.write_text("a,b,c\n", encoding="utf-8")
    old_counts, old_rows = _old_csv_logic(p)
    rows, total = _read_csv_missing_summary(p)
    assert total == old_rows == 0
    assert _rows_to_counts(rows) == old_counts


def test_csv_chunk_boundary(tmp_path, monkeypatch):
    import src.report as report

    monkeypatch.setattr(report, "MISSING_VALUE_BATCH_SIZE", 2)
    p = tmp_path / "balalaika.csv"
    lines = ["s,n"]
    for i in range(7):
        lines.append(f"{'' if i % 3 == 0 else 'v'},{i}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    old_counts, old_rows = _old_csv_logic(p)
    rows, total = _read_csv_missing_summary(p)
    assert total == old_rows
    assert _rows_to_counts(rows) == old_counts


def _old_parquet_logic(path):
    """Verbatim copy of the pre-optimisation parquet scan (decode everything)."""
    import pyarrow.compute as pc

    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    missing_counts = {name: 0 for name in schema.names}
    total_rows = 0
    for batch in parquet_file.iter_batches(batch_size=100_000):
        total_rows += batch.num_rows
        for idx, column in enumerate(batch.schema.names):
            array = batch.column(idx)
            missing = int(array.null_count)
            field_type = batch.schema.field(idx).type
            if pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
                empty_mask = pc.equal(array, "")
                empty_count = pc.sum(pc.fill_null(empty_mask, False)).as_py()
                missing += int(empty_count or 0)
            missing_counts[column] += missing
    return missing_counts, total_rows


def test_parquet_string_and_numeric_nulls(tmp_path):
    df = pd.DataFrame(
        {
            "txt": ["a", "", "мир", None, "  ", "x"],
            "num": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0],
            "i": pd.array([1, 2, None, 4, 5, 6], dtype="Int64"),
            "b": [True, False, None, True, False, True],
        }
    )
    p = tmp_path / "balalaika.parquet"
    df.to_parquet(p, engine="pyarrow", index=False)

    old_counts, old_rows = _old_parquet_logic(p)
    rows, total = _read_parquet_missing_summary(p)
    assert total == old_rows
    assert _rows_to_counts(rows) == old_counts


def test_parquet_multi_rowgroup(tmp_path):
    df = pd.DataFrame(
        {
            "txt": ["a", "", "c", "", "e", "", "g", ""],
            "num": [1.0, np.nan, 3.0, 4.0, np.nan, 6.0, np.nan, 8.0],
        }
    )
    table = pa.Table.from_pandas(df, preserve_index=False)
    p = tmp_path / "balalaika.parquet"
    pq.write_table(table, p, row_group_size=3)  # forces multiple row groups

    old_counts, old_rows = _old_parquet_logic(p)
    rows, total = _read_parquet_missing_summary(p)
    assert total == old_rows
    assert _rows_to_counts(rows) == old_counts
