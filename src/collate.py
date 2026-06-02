import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import Dict, Iterable, Optional
import concurrent.futures
from loguru import logger

from src.utils.csv_manager import discover_audio_paths
from src.utils.logging_setup import setup_logging
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config, read_file_content

SUPPORTED_TIMESTAMP_MODELS = {'giga_ctc', 'giga_ctc_lm', 'tone', 'parakeet_v2', 'parakeet_v3', 'canary'}

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


def process_audio_file(audio_path_str: str, base_path: Path, file_types: Dict[str, str]) -> Dict[str, Optional[str]]:

    audio_path = Path(audio_path_str)
    dir_path = audio_path.parent
    base_name = audio_path.stem

    results = {'filepath': audio_path_str}
    for key, suffix in file_types.items():
        file_path = base_path / dir_path / f"{base_name}{suffix}"
        results[key] = read_file_content(file_path)

    return results


def main(args):
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
    file_types = sidecar_specs(model_names)
    sidecar_columns = set(file_types.keys()) | transcription_sidecar_columns(model_names)
    logger.info(
        f"Collating {len(file_types)} sidecar columns "
        f"({len(model_names)} ASR model(s), {len(configured_timestamp_models)} timestamp-capable)."
    )

    df_path = Path(base_path) / "balalaika.csv"
    if df_path.exists():
        logger.info(f"Loading existing dataframe from {df_path}")
        df = pd.read_csv(df_path)
        df.drop_duplicates(subset='filepath', inplace=True)
        df = drop_csv_text_columns(df, extra_columns=sidecar_columns)
    else:
        logger.info(f"No existing dataframe found. Creating new one from audio paths.")
        audio_paths = discover_audio_paths(base_path, config_path=args.config_path)
        df = pd.DataFrame({'filepath': audio_paths})
    
    audio_paths = df['filepath'].tolist()
    results = []

    logger.info(f"Starting processing with {num_workers} workers")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {executor.submit(process_audio_file, path, base_path, file_types): path for path in audio_paths}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(audio_paths), desc="Processing files"):
            try:
                data = future.result()
                if data:
                    results.append(data)
                    processed += 1
            except Exception as exc:
                path = future_to_path[future]
                logger.error(f'{path} generated an exception: {exc}')
                errors += 1
                error_details.append({"file": str(path), "reason": str(exc)})

    if not results:
        logger.info("No data was processed. Exiting.")
        return
        
    extracted_df = pd.DataFrame(results)

    final_df = pd.merge(df, extracted_df, on='filepath', how='left')

    output_path = base_path / "balalaika.parquet"
    final_df.to_parquet(output_path, engine='pyarrow', index=False)
    logger.info(f"Successfully saved data to {output_path}")

    write_stage_status(
        stage=11,
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
