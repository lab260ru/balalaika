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


def load_metadata(csv_path: Path) -> Dict[str, dict]:
    """Загружает balalaika.csv и делает словарь с ключом по базовому имени файла."""
    import os

    if not csv_path.exists():
        logger.warning(f"Metadata file {csv_path} not found!")
        return {}

    from src.utils.csv_manager import fast_read_csv

    df = fast_read_csv(csv_path)
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
    records = df.drop(columns=["filepath"]).to_dict("records")
    metadata_dict = dict(zip(stems, records))

    logger.info(f"Loaded metadata for {len(metadata_dict)} files.")
    return metadata_dict

def worker_fn(worker_id: int, audio_paths: List[str], output_dir: Path, metadata_dict: Dict[str, dict], max_shard_size: int, max_shard_count: int):
    if not audio_paths:
        return 0, 0

    pattern = str(output_dir / f"shard_{worker_id:03d}_%04d.tar")
    samples_processed = 0
    errors_count = 0
    dir_cache: Dict[str, set] = {}

    with wds.ShardWriter(pattern, maxsize=max_shard_size, maxcount=max_shard_count) as sink:
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

            json_data = {}

            if key in metadata_dict:
                for k, v in metadata_dict[key].items():
                    k_str = str(k)
                    if pd.isna(v) or (isinstance(v, float) and math.isnan(v)):
                        json_data[k_str] = None
                    elif isinstance(v, (pd.Timestamp, pd.Timedelta)):
                        json_data[k_str] = str(v)
                    elif hasattr(v, 'item'):
                        json_data[k_str] = v.item()
                    else:
                        json_data[k_str] = v

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
        
    podcasts_path = Path(podcasts_path_str)
    csv_path = podcasts_path / 'balalaika.csv'
    
    wds_output_dir = podcasts_path.parent / f"{podcasts_path.name}_webdataset" / "train"
    wds_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"WebDataset shards will be saved to: {wds_output_dir}")

    num_workers = config.get('num_workers', 4)
    num_workers = max(1, num_workers)

    all_audio_paths = discover_audio_paths(podcasts_path_str, config_path=config_path)
    if not all_audio_paths:
        logger.warning("No audio data to process.")
        return

    metadata_dict = load_metadata(csv_path)

    chunk_size = len(all_audio_paths) // num_workers + 1
    chunks = [all_audio_paths[i:i + chunk_size] for i in range(0, len(all_audio_paths), chunk_size)]

    logger.info(f"Starting {len(chunks)} workers to build WebDataset from {len(all_audio_paths)} audio files...")

    def chunk_metadata(chunk: List[str]) -> Dict[str, dict]:
        # Ship each worker only its own chunk's records — the full dict is
        # GBs at production row counts and gets pickled once per worker.
        stems = {os.path.splitext(os.path.basename(p))[0] for p in chunk}
        return {s: metadata_dict[s] for s in stems if s in metadata_dict}

    total_processed = 0
    total_errors = 0
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(worker_fn, worker_id, chunk, wds_output_dir, chunk_metadata(chunk), max_shard_size, max_shard_count)
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
        stage=13,
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
