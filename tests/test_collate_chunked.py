"""Read-back equivalence: the chunked collate stream vs. the old single-shot
``merge -> add_asr_consistency_column -> to_parquet`` path.

The bar (per project discipline) is read-back equality: identical DataFrame
(columns, dtypes, values, row order) — parquet row-group layout/bytes may
differ. We reconstruct the OLD path verbatim here and compare it to the new
streamed output across edge cases (NaN metadata, unicode, missing sidecars,
duplicate filepaths in the source CSV, partial sidecar coverage, multiple
slabs).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import src.collate as collate

MODEL_NAMES = ["giga_ctc", "giga_rnnt", "vosk", "tone", "gigaam-v3-e2e-ctc"]


def _write_chunk_json(audio_dir: Path, stem: str, mapping: dict) -> None:
    """Translate old per-suffix sidecar contents into one ``<stem>.json``.

    ``_rover.txt``/``_punct.txt``/``_accent.txt``/``_rover_phonemes.txt`` ->
    top-level keys; ``_<model>.txt`` -> ``asr.<model>``; ``_<model>.tst`` ->
    ``asr_ts.<model>``.
    """
    data: dict = {}
    for suffix, content in mapping.items():
        name = suffix[1:]  # drop leading "_"
        if name.endswith(".tst"):
            data.setdefault("asr_ts", {})[name[:-4]] = content
        elif name.endswith(".txt"):
            base = name[:-4]
            if base in ("rover", "punct", "accent", "rover_phonemes"):
                data[base] = content
            else:
                data.setdefault("asr", {})[base] = content
    (audio_dir / f"{stem}.json").write_text(json.dumps(data), encoding="utf-8")


def _old_path(df: pd.DataFrame, base_path: Path, file_types, model_names) -> pd.DataFrame:
    """Verbatim reconstruction of the pre-chunked collate assembly."""
    df = df.copy()
    df.drop_duplicates(subset="filepath", inplace=True)
    df = collate.drop_csv_text_columns(
        df,
        extra_columns=set(file_types.keys())
        | collate.transcription_sidecar_columns(model_names),
    )
    audio_paths = df["filepath"].tolist()
    results = []
    for path in audio_paths:
        results.append(collate.process_audio_file(path, base_path, file_types, {}))
    extracted_df = pd.DataFrame(results)
    final_df = pd.merge(df, extracted_df, on="filepath", how="left")
    final_df = collate.add_asr_consistency_column(final_df, model_names)
    return final_df.reset_index(drop=True)


def _write_sidecars(audio_dir: Path, stem: str, mapping: dict) -> None:
    _write_chunk_json(audio_dir, stem, mapping)


def _build_fixture(tmp_path: Path):
    base = tmp_path / "data"
    d1 = base / "1" / "10"
    d2 = base / "2" / "20"
    d1.mkdir(parents=True)
    d2.mkdir(parents=True)

    file_types = collate.sidecar_specs(MODEL_NAMES)

    rows = []
    # row 0: full agreement across ASR + all sidecars, unicode
    p = d1 / "a.wav"
    p.write_bytes(b"x")
    _write_sidecars(
        d1,
        "a",
        {
            "_rover.txt": "привет мир",
            "_punct.txt": "Привет, мир.",
            "_accent.txt": "приве́т ми́р",
            "_giga_ctc.txt": "привет мир",
            "_giga_rnnt.txt": "  ПРИВЕТ   мир ",
            "_vosk.txt": "привет мир",
            "_tone.txt": "другое",
            "_gigaam-v3-e2e-ctc.txt": "привет мир",
            "_giga_ctc.tst": '{"t": 0.0}',
        },
    )
    rows.append({"filepath": str(p), "speaker_id": 0, "total_duration": 1.5, "crest_factor": 2.5})

    # row 1: missing ALL sidecars -> empty strings, consistency NaN
    p = d1 / "b.wav"
    p.write_bytes(b"x")
    rows.append({"filepath": str(p), "speaker_id": 1, "total_duration": 2.0, "crest_factor": np.nan})

    # row 2: only one ASR transcript -> consistency NaN
    p = d2 / "c.wav"
    p.write_bytes(b"x")
    _write_sidecars(d2, "c", {"_giga_ctc.txt": "только один"})
    rows.append({"filepath": str(p), "speaker_id": 2, "total_duration": np.nan, "crest_factor": 1.0})

    # row 3: partial agreement
    p = d2 / "d.wav"
    p.write_bytes(b"x")
    _write_sidecars(
        d2,
        "d",
        {
            "_giga_ctc.txt": "раз два три",
            "_giga_rnnt.txt": "раз два три",
            "_vosk.txt": "совсем другое",
            "_tone.txt": "раз два",
        },
    )
    rows.append({"filepath": str(p), "speaker_id": 3, "total_duration": 5.0, "crest_factor": 3.3})

    # row 4: stale text column present in CSV (must be dropped), unicode metadata-ish
    p = d2 / "e.wav"
    p.write_bytes(b"x")
    _write_sidecars(d2, "e", {"_giga_ctc.txt": "текст", "_vosk.txt": "текст"})
    rows.append(
        {
            "filepath": str(p),
            "speaker_id": 4,
            "total_duration": 4.0,
            "crest_factor": 2.0,
            "rover": "stale text that must be dropped",
        }
    )

    df = pd.DataFrame(rows)
    # duplicate filepath row (pre-dedup) -> old path drops it; new path too
    dup = df.iloc[[0]].copy()
    dup["speaker_id"] = 999
    df = pd.concat([df, dup], ignore_index=True)
    return df, base, file_types


@pytest.mark.parametrize("slab_rows", [1, 2, 3, 1000])
def test_chunked_matches_old_path(tmp_path, monkeypatch, slab_rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import concurrent.futures

    df, base, file_types = _build_fixture(tmp_path)

    expected = _old_path(df, base, file_types, MODEL_NAMES)

    # Drive the new streamed assembly directly (mirrors collate.main's loop).
    prepped = df.copy()
    prepped.drop_duplicates(subset="filepath", inplace=True)
    prepped = collate.drop_csv_text_columns(
        prepped,
        extra_columns=set(file_types.keys())
        | collate.transcription_sidecar_columns(MODEL_NAMES),
    )
    prepped = prepped.reset_index(drop=True)

    out_path = tmp_path / "out.parquet"
    dir_cache: dict = {}
    writer = None
    schema = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        for start in range(0, len(prepped), slab_rows):
            meta_slab = prepped.iloc[start : start + slab_rows]
            slab, errs = collate.build_slab_frame(
                meta_slab, file_types, MODEL_NAMES, base, dir_cache, 2, ex
            )
            assert errs == []
            table = pa.Table.from_pandas(slab, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(out_path, schema)
            else:
                table = table.cast(schema)
            writer.write_table(table)
    writer.close()

    got = pd.read_parquet(out_path)

    assert list(got.columns) == list(expected.columns)
    pdt.assert_frame_equal(
        got.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=True,
        check_like=False,
    )


def test_error_path_keeps_row_with_nan_sidecars(tmp_path, monkeypatch):
    """A file whose sidecar read raises must keep its metadata row with NaN
    sidecar columns (== the old left-merge with that path absent from
    extracted_df), not drop the row."""
    import concurrent.futures

    import pyarrow as pa
    import pyarrow.parquet as pq

    df, base, file_types = _build_fixture(tmp_path)
    prepped = df.copy()
    prepped.drop_duplicates(subset="filepath", inplace=True)
    prepped = collate.drop_csv_text_columns(
        prepped,
        extra_columns=set(file_types.keys())
        | collate.transcription_sidecar_columns(MODEL_NAMES),
    ).reset_index(drop=True)

    bad_path = prepped["filepath"].iloc[1]
    real = collate.process_audio_file

    def flaky(path, base_path, ft, cache=None):
        if path == bad_path:
            raise OSError("boom")
        return real(path, base_path, ft, cache)

    monkeypatch.setattr(collate, "process_audio_file", flaky)

    out_path = tmp_path / "out.parquet"
    writer = None
    schema = None
    n_errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        for start in range(0, len(prepped), 2):
            meta_slab = prepped.iloc[start : start + 2]
            slab, errs = collate.build_slab_frame(
                meta_slab, file_types, MODEL_NAMES, base, {}, 2, ex
            )
            n_errors += len(errs)
            table = pa.Table.from_pandas(slab, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(out_path, schema)
            else:
                table = table.cast(schema)
            writer.write_table(table)
    writer.close()

    assert n_errors == 1
    got = pd.read_parquet(out_path)
    # The bad row is still present (same total rows as deduped metadata)...
    assert len(got) == len(prepped)
    # ...with its sidecar columns NaN/None.
    bad_row = got[got["filepath"] == bad_path].iloc[0]
    for key in file_types:
        assert pd.isna(bad_row[key]) or bad_row[key] is None


def test_full_main_roundtrip(tmp_path, monkeypatch):
    """End-to-end through collate.main against a config, compared to old path."""
    import yaml

    df, base, file_types = _build_fixture(tmp_path)
    df.to_parquet(base / "balalaika.parquet", index=False)

    expected = _old_path(df, base, file_types, MODEL_NAMES)

    cfg = {
        "download": {"podcasts_path": str(base), "num_workers": 2},
        "transcription": {"model_names": MODEL_NAMES},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    class Args:
        config_path = str(cfg_path)
        log_dir = str(tmp_path / "logs")

    # Force tiny slabs via config so multiple row-groups are exercised.
    cfg["download"]["collate_slab_rows"] = 2
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    collate.main(Args())

    got = pd.read_parquet(base / "balalaika.parquet")
    assert list(got.columns) == list(expected.columns)
    pdt.assert_frame_equal(
        got.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=True
    )
