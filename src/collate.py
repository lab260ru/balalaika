import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import Dict, Iterable, Optional
import concurrent.futures
from loguru import logger

from src.utils.csv_manager import discover_audio_paths, fast_read_csv
from src.utils.logging_setup import setup_logging
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config, read_file_content

SUPPORTED_TIMESTAMP_MODELS = {'giga_ctc', 'giga_ctc_lm', 'tone', 'parakeet_v2', 'parakeet_v3', 'canary'}
ASR_CONSISTENCY_COLUMN = "asr_consistency_percent"

TEXT_COLUMNS = {
    "accent",
    "rover",
    "punct",
    "phonemes",
    "rover_phonemes",
    "text",
    "transcript",
    "transcription",
    "giga_ctc",
    "giga_rnnt",
    "giga_ctc_lm",
    "gigaam-v3-e2e-ctc",
    "gigaam_v3_e2e_ctc",
    "tone",
    "vosk",
    "vosk_small",
    "parakeet_v2",
    "parakeet_v3",
    "canary",
    "whisper_base",
    "whisper_turbo",
}


def output_suffix_for_model(model_name: str) -> str:
    """Match transcription.py sidecar naming."""
    return "vosk" if "vosk" in str(model_name) else str(model_name)


def transcription_sidecar_columns(model_names: Iterable[str]) -> set[str]:
    columns = set()
    for model_name in model_names:
        suffix = output_suffix_for_model(model_name)
        columns.add(suffix)
        columns.add(f"{suffix}_timestamps")
    columns.add(ASR_CONSISTENCY_COLUMN)
    return columns


def drop_csv_text_columns(df: pd.DataFrame, extra_columns: Optional[set[str]] = None) -> pd.DataFrame:
    """Keep balalaika.csv as metadata-only; sidecars feed final parquet text."""
    extra_columns = extra_columns or set()
    drop_cols = [
        col
        for col in df.columns
        if str(col).lower() in TEXT_COLUMNS
        or str(col) in extra_columns
        or str(col).lower().endswith(("_txt", "_text", "_transcript"))
        or str(col).lower().endswith(("_tst", "_timestamps"))
    ]
    if drop_cols:
        logger.info(f"Dropping text columns from CSV metadata: {drop_cols}")
    return df.drop(columns=drop_cols)


def sidecar_specs(model_names: Iterable[str]) -> Dict[str, str]:
    specs = {
        'accent': '_accent.txt',
        'rover': '_rover.txt',
        'punct': '_punct.txt',
        'phonemes': '_rover_phonemes.txt',
    }

    seen_suffixes = set()
    for model_name in model_names:
        suffix = output_suffix_for_model(model_name)
        if suffix in seen_suffixes:
            continue
        seen_suffixes.add(suffix)
        specs[suffix] = f"_{suffix}.txt"
        specs[f"{suffix}_timestamps"] = f"_{suffix}.tst"

    return specs


def normalize_transcript(text: object) -> str:
    if text is None:
        return ""
    if pd.isna(text):
        return ""
    return " ".join(str(text).lower().split())


def asr_consistency_percent(row: pd.Series, asr_columns: list[str]) -> float:
    transcripts = [
        normalized
        for col in asr_columns
        if col in row.index
        for normalized in [normalize_transcript(row[col])]
        if normalized
    ]
    if len(transcripts) < 2:
        return float("nan")

    best_matching_others = max(
        transcripts.count(transcript) - 1
        for transcript in set(transcripts)
    )
    return best_matching_others / (len(transcripts) - 1) * 100.0


def add_asr_consistency_column(df: pd.DataFrame, model_names: Iterable[str]) -> pd.DataFrame:
    """Vectorised equivalent of applying :func:`asr_consistency_percent` per row.

    Transcripts are normalised column-wise, factorised to integer codes, and
    agreement is computed from pairwise code comparisons — same semantics as
    the row-wise reference implementation (kept above for tests), orders of
    magnitude faster on multi-million-row frames.
    """
    import numpy as np

    asr_columns = []
    seen = set()
    for model_name in model_names:
        suffix = output_suffix_for_model(model_name)
        if suffix in seen:
            continue
        seen.add(suffix)
        if suffix in df.columns:
            asr_columns.append(suffix)

    out = df
    if len(asr_columns) < 2:
        out[ASR_CONSISTENCY_COLUMN] = float("nan")
        logger.info("ASR consistency skipped: fewer than two ASR text columns found.")
        return out

    normalized = [
        out[col].fillna("").astype(str).str.lower().str.split().str.join(" ")
        for col in asr_columns
    ]
    # One shared factorisation so equal transcripts share an integer code.
    stacked = pd.concat(normalized, ignore_index=True)
    codes_flat, uniques = pd.factorize(stacked)
    empty_code = -1
    for idx, value in enumerate(uniques):
        if value == "":
            empty_code = idx
            break

    n = len(out)
    k_cols = len(asr_columns)
    codes = codes_flat.reshape(k_cols, n)
    nonempty = codes != empty_code

    matches = np.zeros((k_cols, n), dtype=np.int16)
    for i in range(k_cols):
        for j in range(i + 1, k_cols):
            eq = (codes[i] == codes[j]) & nonempty[i] & nonempty[j]
            matches[i] += eq
            matches[j] += eq

    k = nonempty.sum(axis=0)
    best = np.where(nonempty, matches, -1).max(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        consistency = np.where(
            k >= 2, best / np.maximum(k - 1, 1) * 100.0, np.nan
        )
    out[ASR_CONSISTENCY_COLUMN] = consistency.astype(float)
    logger.info(
        f"Added {ASR_CONSISTENCY_COLUMN} from ASR columns: {asr_columns}"
    )
    return out


def process_audio_file(
    audio_path_str: str,
    base_path: Path,
    file_types: Dict[str, str],
    _dir_names: Optional[Dict[str, set]] = None,
) -> Dict[str, Optional[str]]:
    import os

    dirname, filename = os.path.split(audio_path_str)
    base_name = os.path.splitext(filename)[0]
    # os.path.join mirrors the original Path join: an absolute dirname wins
    # over base_path, a relative one is anchored under it.
    target_dir = os.path.join(str(base_path), dirname)

    names = None
    if _dir_names is not None:
        names = _dir_names.get(target_dir)
        if names is None:
            try:
                names = {entry.name for entry in os.scandir(target_dir)}
            except OSError:
                names = set()
            _dir_names[target_dir] = names

    results: Dict[str, Optional[str]] = {'filepath': audio_path_str}
    for key, suffix in file_types.items():
        name = f"{base_name}{suffix}"
        if names is not None and name not in names:
            # directory listing says the sidecar doesn't exist -> same result
            # as read_file_content's FileNotFoundError path, without the open
            results[key] = ''
        else:
            results[key] = read_file_content(os.path.join(target_dir, name))

    return results


def build_slab_frame(
    metadata_slab: pd.DataFrame,
    file_types: Dict[str, str],
    model_names: Iterable[str],
    base_path: Path,
    dir_names_cache: Dict[str, set],
    num_workers: int,
    executor: concurrent.futures.Executor,
) -> tuple[pd.DataFrame, list[tuple[str, Exception]]]:
    """Read sidecars for one slab of rows and assemble the merged frame.

    ``metadata_slab`` is a contiguous slice of the (deduplicated, text-column-
    dropped) metadata frame, in audio-path order. Its sidecar columns are read
    per-file, then column-aligned by position (paths are unique and the slab
    order is preserved), which is exactly equivalent to
    ``pd.merge(df, extracted_df, on='filepath', how='left')`` on the unique
    join key — same values, same row order, same column order
    (metadata columns first, then sidecar columns in ``file_types`` order, then
    the appended consistency column) — but it only ever materialises one slab's
    worth of sidecar text instead of the whole dataset's.
    """
    paths = metadata_slab['filepath'].tolist()

    # Split into sub-slabs so per-future dispatch overhead stays amortised while
    # we still parallelise the (GIL-released) os.scandir / open work.
    sub = max(1, len(paths) // max(1, num_workers))
    sub_slabs = [paths[i:i + sub] for i in range(0, len(paths), sub)]

    def process_sub(sub_paths):
        sub_results, sub_errors = [], []
        for path in sub_paths:
            try:
                sub_results.append(
                    process_audio_file(path, base_path, file_types, dir_names_cache)
                )
            except Exception as exc:  # keep per-file error attribution
                sub_results.append(None)
                sub_errors.append((path, exc))
        return sub_results, sub_errors

    import numpy as np

    # Per-column accumulators preserve the input order (executor.map is ordered),
    # so the slab aligns 1:1 with metadata_slab without building one dict per row.
    # A file that raised keeps its metadata row but gets NaN sidecar values —
    # exactly what the old `pd.merge(df, extracted_df, how='left')` produced when
    # that path was missing from extracted_df (the row stayed, sidecars NaN).
    columns: Dict[str, list] = {key: [] for key in file_types}
    errors: list[tuple[str, Exception]] = []
    for sub_results, sub_errors in executor.map(process_sub, sub_slabs):
        errors.extend(sub_errors)
        for res in sub_results:
            if res is None:
                for key in file_types:
                    columns[key].append(np.nan)
                continue
            for key in file_types:
                columns[key].append(res[key])

    kept_meta = metadata_slab.reset_index(drop=True)
    sidecar_df = pd.DataFrame(columns, index=kept_meta.index)
    slab = pd.concat([kept_meta, sidecar_df], axis=1)
    slab = add_asr_consistency_column(slab, model_names)
    return slab, errors


def main(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    processed = 0
    errors = 0
    error_details: list[dict] = []

    setup_logging("collate", log_dir=args.log_dir)
    config = load_config(args.config_path, 'download')
    transcription_config = load_config(args.config_path, 'transcription')
    model_names = transcription_config.get('model_names', [])
    configured_timestamp_models = [
        name for name in model_names if name in SUPPORTED_TIMESTAMP_MODELS
    ]
    base_path = Path(config.get('podcasts_path', '../../balalaika'))
    num_workers = config.get('num_workers', 32)
    # Rows per streamed slab. Caps peak RAM at ~O(slab) instead of O(dataset):
    # only one slab's sidecar text is resident at a time. Lower it on very
    # low-RAM nodes; raise it to trade RAM for fewer parquet row-groups.
    slab_rows = int(config.get('collate_slab_rows', 200_000))
    # Parquet compression for balalaika.parquet (zstd ~2x smaller than the
    # pandas/pyarrow default snappy on text-heavy frames = fewer HDD bytes).
    parquet_compression = config.get('collate_parquet_compression', 'zstd')
    file_types = sidecar_specs(model_names)
    sidecar_columns = set(file_types.keys()) | transcription_sidecar_columns(model_names)
    logger.info(
        f"Collating {len(file_types)} sidecar columns "
        f"({len(model_names)} ASR model(s), {len(configured_timestamp_models)} timestamp-capable)."
    )

    df_path = Path(base_path) / "balalaika.csv"
    if df_path.exists():
        logger.info(f"Loading existing dataframe from {df_path}")
        df = fast_read_csv(df_path)
        df.drop_duplicates(subset='filepath', inplace=True)
        df = drop_csv_text_columns(df, extra_columns=sidecar_columns)
    else:
        logger.info(f"No existing dataframe found. Creating new one from audio paths.")
        audio_paths = discover_audio_paths(base_path, config_path=args.config_path)
        df = pd.DataFrame({'filepath': audio_paths})

    df = df.reset_index(drop=True)
    n_rows = len(df)
    logger.info(f"Starting chunked processing of {n_rows} rows with {num_workers} workers")

    if n_rows == 0:
        logger.info("No data was processed. Exiting.")
        return

    output_path = base_path / "balalaika.parquet"
    dir_names_cache: Dict[str, set] = {}
    writer: Optional[pq.ParquetWriter] = None
    schema: Optional[pa.Schema] = None

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            with tqdm(total=n_rows, desc="Processing files") as bar:
                for start in range(0, n_rows, slab_rows):
                    metadata_slab = df.iloc[start:start + slab_rows]
                    slab, slab_errors = build_slab_frame(
                        metadata_slab,
                        file_types,
                        model_names,
                        base_path,
                        dir_names_cache,
                        num_workers,
                        executor,
                    )
                    # `processed` counts successfully-read files (old semantics:
                    # only rows that made it into extracted_df), errors counted
                    # separately. Both rows still land in the parquet (the row
                    # itself is kept with NaN sidecars, as the old left-merge did).
                    processed += len(slab) - len(slab_errors)
                    for path, exc in slab_errors:
                        logger.error(f'{path} generated an exception: {exc}')
                        errors += 1
                        error_details.append({"file": str(path), "reason": str(exc)})

                    table = pa.Table.from_pandas(slab, preserve_index=False)
                    if writer is None:
                        # Lock the schema from the first slab; every metadata
                        # column comes from one CSV read so dtypes are stable,
                        # and sidecar columns are always strings.
                        schema = table.schema
                        writer = pq.ParquetWriter(
                            output_path, schema, compression=parquet_compression
                        )
                    else:
                        table = table.cast(schema)
                    writer.write_table(table)
                    bar.update(len(metadata_slab))
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        logger.info("No data was processed. Exiting.")
        return

    logger.info(f"Successfully saved data to {output_path}")

    write_stage_status(
        stage=12,
        stage_name="collate",
        log_dir=args.log_dir or "./logs",
        processed=processed,
        skipped=0,
        errors=errors,
        error_details=error_details,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collate information from podcast files.")
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to config file",
    )
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")

    args = parser.parse_args()
    main(args)
