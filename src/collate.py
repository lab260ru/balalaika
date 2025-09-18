import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import Dict, Optional
import concurrent.futures
from loguru import logger

from src.utils import load_config, read_file_content

def process_audio_file(audio_path_str: str, base_path: Path) -> Dict[str, Optional[str]]:

    audio_path = Path(audio_path_str)
    dir_path = audio_path.parent
    base_name = audio_path.stem

    file_types = {
        'accent': '_accent.txt',
        'rover': '_rover.txt',
        'punct': '_punct.txt',
        'phonemes': '_rover_phonemes.txt'
    }

    results = {'filepath': audio_path_str}
    for key, suffix in file_types.items():
        file_path = base_path / dir_path / f"{base_name}{suffix}"
        results[key] = read_file_content(file_path)

    return results


def main(args):

    base_path = Path(
        load_config(args.config_path, 'download').get('podcasts_path', '../../balalaika')
        if args.config_path else args.podcasts_path
    )


    df = pd.read_csv(base_path / "balalaika.csv")
    df.drop_duplicates(subset='filepath', inplace=True)
    
    audio_paths = df['filepath'].tolist()
    results = []

    num_workers = 32
    logger.info(f"Starting processing with {num_workers} workers")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {executor.submit(process_audio_file, path, base_path): path for path in audio_paths}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(audio_paths), desc="Processing files"):
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception as exc:
                path = future_to_path[future]
                logger.error(f'{path} generated an exception: {exc}')

    if not results:
        logger.info("No data was processed. Exiting.")
        return
        
    extracted_df = pd.DataFrame(results)

    final_df = pd.merge(df, extracted_df, on='filepath', how='left')

    output_path = base_path / "balalaika.parquet"
    final_df.to_parquet(output_path, engine='pyarrow', index=False)
    logger.info(f"Successfully saved data to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collate information from podcast files.")
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to config file"
    )
    parser.add_argument(
        "--podcasts_path",
        type=str,
        default='../../balalaika', 
        help="Path to dataset directory"
    )
    
    args = parser.parse_args()
    main(args)