import argparse
import math
import json
import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict

import webdataset as wds
from tqdm import tqdm
from loguru import logger

from src.utils.chunk_json import chunk_json_path, read_chunk_json
from src.utils.csv_manager import discover_audio_paths
from src.utils.logging_setup import setup_logging
from src.utils.utils import load_config
from src.utils.stage_status import write_stage_status

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


def _open_shard_exclusive(path: str):
    """Never truncate an existing shard when a start index is reused."""
    return open(path, "xb")


def resolve_output_dir(podcasts_path: Path, output_path: str | None) -> Path:
    """Resolve the exact directory that will contain WebDataset tar shards."""
    if output_path and str(output_path).strip():
        return Path(output_path).expanduser()
    return podcasts_path.parent / f"{podcasts_path.name}_webdataset" / "train"


def load_metadata(podcasts_path: Path) -> Dict[str, dict]:
    """Load legacy parquet metadata keyed by audio stem.

    New pipeline runs write metadata into each chunk's ``<stem>.json`` sidecar,
    so the main export path no longer needs this full-state read. This remains
    as a fallback for older datasets that do not have metadata sidecars yet.
    """
    import os

    from src.utils.csv_manager import read_state_dataframe

    df = read_state_dataframe(podcasts_path)
    if df.empty:
        legacy_csv = podcasts_path / "balalaika.csv"
        if legacy_csv.exists():
            logger.info(f"Loading legacy CSV metadata from {legacy_csv}")
            df = pd.read_csv(legacy_csv, low_memory=False)
        else:
            logger.warning(f"No pipeline state found under {podcasts_path}!")
            return {}

    drop_cols = [
        col
        for col in df.columns
        if str(col).lower() in TEXT_COLUMNS
        or str(col).lower().endswith(("_txt", "_text", "_transcript"))
    ]
    if drop_cols:
        logger.info(f"Dropping text columns from CSV metadata: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # to_dict('records') + zip is ~30x faster than iterrows on large frames.
    stems = [
        os.path.splitext(os.path.basename(p))[0] for p in df["filepath"].astype(str)
    ]
    records = _sanitize_records(df.drop(columns=["filepath"]))
    metadata_dict = dict(zip(stems, records))

    logger.info(f"Loaded metadata for {len(metadata_dict)} files.")
    return metadata_dict


TEXT_SIDECAR_KEYS = TEXT_COLUMNS | {"asr", "asr_ts", "asr_consistency"}


def has_metadata_sidecars(audio_paths: List[str]) -> bool:
    """Return True when chunk JSONs contain flat state metadata, not only text."""
    for path in audio_paths:
        json_path = chunk_json_path(path)
        if not json_path.exists():
            continue
        data = read_chunk_json(json_path)
        if any(str(key) not in TEXT_SIDECAR_KEYS for key in data):
            return True
    return False


def load_sidecar_metadata(audio_path: Path) -> dict:
    data = read_chunk_json(chunk_json_path(audio_path))
    return dict(data) if isinstance(data, dict) else {}


def _sanitize_record_value(v):
    """Exact per-cell sanitisation used by the old worker hot loop.

    Kept as the column-fallback path for object columns that may hold a mix
    of types (datetime scalars, numpy scalars, strings); identical result to
    the inline branch the worker used to run per cell.
    """
    if pd.isna(v) or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (pd.Timestamp, pd.Timedelta)):
        return str(v)
    if hasattr(v, "item"):
        return v.item()
    return v


def _sanitize_records(df: pd.DataFrame) -> List[dict]:
    """Vectorised, JSON-ready records.

    Reproduces the old worker's per-cell branch (NaN/NaT -> None,
    Timestamp/Timedelta -> str, numpy scalar -> .item(), else passthrough)
    but computes the null mask and per-column conversion once instead of
    dispatching ``pd.isna``/``isinstance``/``hasattr`` per value per file.
    Column keys are ``str(col)`` to match the old ``str(k)``.
    """
    import numpy as np
    import pandas.api.types as ptypes

    n = len(df)
    columns = list(df.columns)
    per_col_values: Dict[str, list] = {}
    for col in columns:
        s = df[col]
        null_mask = s.isna().to_numpy()
        if ptypes.is_datetime64_any_dtype(s) or ptypes.is_timedelta64_dtype(s):
            # Timestamp/Timedelta -> str(v); NaT -> None.
            strs = s.astype(object).astype(str)
            vals = [None if null_mask[i] else strs.iloc[i] for i in range(n)]
        elif ptypes.is_integer_dtype(s):
            # Native int64 holds no NaN, but pandas' nullable Int64 (e.g. from a
            # parquet read) can carry pd.NA -> honour the mask like the float
            # branch. to_numpy() upcasts nullable-with-null to float64, so guard
            # int() on the present cells only.
            arr = s.to_numpy()
            pylist = arr.tolist()
            vals = [None if null_mask[i] else int(pylist[i]) for i in range(n)]
        elif ptypes.is_float_dtype(s):
            arr = s.to_numpy()
            pylist = arr.tolist()  # numpy float64 -> python float
            vals = [None if null_mask[i] else pylist[i] for i in range(n)]
        elif ptypes.is_bool_dtype(s):
            # Native bool holds no NA, but nullable 'boolean' (parquet) can; its
            # to_numpy() is an object array with pd.NA -> bool(pd.NA) raises, so
            # honour the mask and only coerce present cells.
            arr = s.to_numpy()
            pylist = arr.tolist()
            vals = [None if null_mask[i] else bool(pylist[i]) for i in range(n)]
        else:
            # Object/mixed column: fall back to the exact per-cell transform.
            raw = s.tolist()
            vals = [_sanitize_record_value(raw[i]) for i in range(n)]
        per_col_values[str(col)] = vals

    str_cols = [str(c) for c in columns]
    return [
        {c: per_col_values[c][i] for c in str_cols}
        for i in range(n)
    ]

def worker_fn(
    worker_id: int,
    audio_paths: List[str],
    output_dir: Path,
    metadata_dict: Dict[str, dict],
    max_shard_size: int,
    max_shard_count: int,
    shard_start_index: int = 0,
):
    if not audio_paths:
        return 0, 0

    pattern = str(output_dir / f"shard_{worker_id:03d}_%04d.tar")
    samples_processed = 0
    errors_count = 0
    dir_cache: Dict[str, set] = {}

    with wds.ShardWriter(
        pattern,
        maxsize=max_shard_size,
        maxcount=max_shard_count,
        start_shard=shard_start_index,
        opener=_open_shard_exclusive,
    ) as sink:
        for audio_str in tqdm(audio_paths, desc=f"Worker {worker_id}", position=worker_id):
            audio_path = Path(audio_str)
            
            key = audio_path.stem
            ext = audio_path.suffix.lstrip('.')

            safe_key = key.replace('.', '_')

            # Read directly and let the exception path handle a missing file,
            # avoiding a redundant exists() stat right before the open. A
            # missing file is skipped silently (same as the old exists() guard);
            # any other read error is logged and counted as before.
            try:
                audio_bytes = audio_path.read_bytes()
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.warning(f"Error reading {audio_path}: {e}")
                errors_count += 1
                continue

            # Metadata cells were sanitised once, vectorised, in load_metadata
            # (NaN -> None, Timestamp/Timedelta -> str, numpy scalar -> .item()),
            # so the worker just copies its chunk's record — no per-cell pd.isna /
            # isinstance / hasattr dispatch in the hot loop.
            meta = metadata_dict.get(key)
            json_data = dict(meta) if meta is not None else {}
            json_data.update(load_sidecar_metadata(audio_path))

            # One cached scandir per directory instead of two globs per FILE:
            # directories hold hundreds of chunks, so the old pattern rescanned
            # every directory listing 2x per chunk inside it.
            parent_dir = audio_path.parent
            parent_str = str(parent_dir)
            dir_files = dir_cache.get(parent_str)
            if dir_files is None:
                if len(dir_cache) > 64:
                    dir_cache.clear()  # keep worker memory bounded
                try:
                    with os.scandir(parent_str) as it:
                        dir_files = {e.name for e in it if e.is_file()}
                except OSError:
                    dir_files = set()
                dir_cache[parent_str] = dir_files

            prefix_us = f"{key}_"
            prefix_dot = f"{key}."
            sibling_names = [
                n
                for n in dir_files
                if (n.startswith(prefix_us) or n.startswith(prefix_dot))
                and n != audio_path.name
                and n != f"{key}.json"
            ]

            for sibling_name in sibling_names:
                sibling = parent_dir / sibling_name
                postfix_name = sibling_name[len(key):].lstrip('_.')

                try:
                    text_content = sibling.read_text(encoding='utf-8').strip()
                    json_data[str(postfix_name)] = text_content
                except UnicodeDecodeError:
                    pass
                except Exception as e:
                    logger.warning(f"Error reading {sibling}: {e}")
                    errors_count += 1

            try:
                json_bytes = json.dumps(json_data, ensure_ascii=False).encode('utf-8')
            except Exception as e:
                logger.error(f"Failed to serialize JSON for {key}: {e}")
                errors_count += 1
                continue

            sample = {
                "__key__": safe_key,
                ext: audio_bytes,
                "json": json_bytes
            }
            
            try:
                sink.write(sample)
                samples_processed += 1
            except Exception as e:
                logger.error(f"Failed to write sample {key} to tar: {e}")
                errors_count += 1

    return samples_processed, errors_count

def main(config, config_path: str | None = None):
    podcasts_path_str = config.get('podcasts_path')
    if not podcasts_path_str:
        logger.error("podcasts_path is not defined in the config!")
        return

    max_shard_size = config.get('max_shard_size', 512 * 1024 * 1024)
    max_shard_count = config.get('max_shard_count', 10000)
    shard_start_index = int(config.get('shard_start_index', 0))
    if shard_start_index < 0:
        raise ValueError("export.shard_start_index must be >= 0")
        
    podcasts_path = Path(podcasts_path_str)

    wds_output_dir = resolve_output_dir(podcasts_path, config.get('output_path'))
    wds_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"WebDataset shards will be saved to: {wds_output_dir}")

    num_workers = config.get('num_workers', 4)
    num_workers = max(1, num_workers)

    all_audio_paths = discover_audio_paths(podcasts_path_str, config_path=config_path)
    if not all_audio_paths:
        logger.warning("No audio data to process.")
        return

    if has_metadata_sidecars(all_audio_paths):
        logger.info("Using per-audio JSON sidecars for WebDataset metadata.")
        metadata_dict = {}
    else:
        logger.info("No metadata sidecars found; falling back to full parquet metadata read.")
        metadata_dict = load_metadata(podcasts_path)

    chunk_size = len(all_audio_paths) // num_workers + 1
    chunks = [all_audio_paths[i:i + chunk_size] for i in range(0, len(all_audio_paths), chunk_size)]

    logger.info(f"Starting {len(chunks)} workers to build WebDataset from {len(all_audio_paths)} audio files...")
    logger.info(
        f"Shard numbering starts at {shard_start_index}; existing shard files will not be overwritten."
    )

    def chunk_metadata(chunk: List[str]) -> Dict[str, dict]:
        # Ship each worker only its own chunk's records — the full dict is
        # GBs at production row counts and gets pickled once per worker.
        stems = {os.path.splitext(os.path.basename(p))[0] for p in chunk}
        return {s: metadata_dict[s] for s in stems if s in metadata_dict}

    total_processed = 0
    total_errors = 0
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(
                worker_fn,
                worker_id,
                chunk,
                wds_output_dir,
                chunk_metadata(chunk),
                max_shard_size,
                max_shard_count,
                shard_start_index,
            )
            for worker_id, chunk in enumerate(chunks)
        ]

        for future in as_completed(futures):
            try:
                chunk_processed, chunk_errors = future.result()
                total_processed += chunk_processed
                total_errors += chunk_errors
            except Exception as e:
                logger.error(f"Worker failed with error: {e}")
                total_errors += 1

    logger.success(f"WebDataset creation completed! Total samples packed: {total_processed}")
    logger.success(f"Output directory: {wds_output_dir}")

    write_stage_status(
        stage=14,
        stage_name="to_webdataset",
        log_dir=config.get("log_dir", "./logs"),
        processed=total_processed,
        skipped=0,
        errors=total_errors,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    args = parser.parse_args()

    setup_logging("to_webdataset", log_dir=args.log_dir)
    config = load_config(args.config_path, process_name='export')
    main(config, args.config_path)
