import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import Dict, Optional
import concurrent.futures
from loguru import logger

from src.utils.logging_setup import setup_logging
from src.utils.stage_status import write_stage_status
from src.utils.utils import get_audio_paths, load_config, read_file_content

def process_audio_file(audio_path_str: str, base_path: Path) -> Dict[str, Optional[str]]:

    audio_path = Path(audio_path_str)
    dir_path = audio_path.parent
    base_name = audio_path.stem

    file_types = {
        'accent': '_accent.txt',
        'rover': '_rover.txt',
        'punct': '_punct.txt',
        'phonemes': '_rover_phonemes.txt',
    }

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
    base_path = Path(config.get('podcasts_path', '../../balalaika'))
    num_workers = config.get('num_workers', 32)

    df_path = Path(base_path) / "balalaika.csv"
    if df_path.exists():
        logger.info(f"Loading existing dataframe from {df_path}")
        df = pd.read_csv(df_path)
        df.drop_duplicates(subset='filepath', inplace=True)
    else:
        logger.info(f"No existing dataframe found. Creating new one from audio paths.")
        audio_paths = [str(path) for path in get_audio_paths(base_path)]
        df = pd.DataFrame({'filepath': audio_paths})
    
    audio_paths = df['filepath'].tolist()
    results = []

    logger.info(f"Starting processing with {num_workers} workers")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {executor.submit(process_audio_file, path, base_path): path for path in audio_paths}
        
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