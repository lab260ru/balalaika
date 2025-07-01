import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import Dict

from src.utils import load_config, read_file_content


def process_audio_row(row: pd.Series, base_path: Path) -> Dict[str, str]:
    audio_path = Path(row['audio_path'])
    dir_path = audio_path.parent
    base_name = audio_path.stem

    file_types = {
        'accent': '_accent.txt',
        'phonemes': '_e_phonemes.txt',
        'giga': '_giga.txt',
        'punct': '_punct.txt',
        'whisper': '_whisper.txt',
        'e': '_e.txt'
    }

    results = {}
    for key, suffix in file_types.items():
        file_path = base_path / dir_path / f"{base_name}{suffix}"
        results[key] = read_file_content(file_path)

    return results


def main(args):
    print(args.config_path)
    base_path = Path(
        load_config(args.config_path, 'download').get('podcasts_path', '../../podcasts')
        if args.config_path else args.podcasts_path
    )

    df = pd.read_csv(base_path / "results.csv")

    columns_to_add = ['accent', 'phonemes', 'giga', 'punct', 'whisper', 'e']
    for col in columns_to_add:
        df[col] = ''

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
        extracted_data = process_audio_row(row, base_path)
        for col in columns_to_add:
            df.at[idx, col] = extracted_data[col]

    df.drop_duplicates(subset='audio_path', inplace=True)
    output_path = base_path / "balalaika.parquet"
    df.to_parquet(output_path, engine='pyarrow', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collate information from podcast files.")
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        help="Path to config file"
    )
    parser.add_argument(
        "--podcasts_path",
        "-p",
        type=str,
        help="Path to dataset directory"
    )
    args = parser.parse_args()
    main(args)