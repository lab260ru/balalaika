import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import os

from src.utils import get_txt_paths, load_config

def make_txt_with_e(path: Path):

    new_path = path.with_name(path.name.replace('_accent.txt', '_e.txt'))

    if new_path.exists():
        return

    try:
        with open(path, encoding='utf-8', mode='r') as f:
            text = f.readline().strip().lower()

        new_text = re.sub(r'[^а-яё ]', '', text)
        new_text = re.sub(r'\s+', ' ', new_text).strip()

        with open(new_path, encoding='utf-8', mode='w') as f:
            f.write(new_text)

    except Exception as e:
        logger.error(f"Error {path}: {e}")

def get_valit_txt_paths(podcasts_path: str) -> List[Path]:
    valis_paths = []
    accent_paths = get_txt_paths(podcasts_path, '_accent.txt')
    for accent_path in accent_paths:
        new_path = accent_path.with_name(accent_path.name.replace('_accent.txt', '_e.txt'))
        if not os.path.exists(new_path):
            valis_paths.append(accent_path)
    return valis_paths

            

def main(args):
    config = load_config(args.config_path, 'yofication')
    num_workers = args.num_workers if args.num_workers else config.get('num_workers', 4)
    podcasts_path = args.podcasts_path if args.podcasts_path else config.get('podcasts_path', '../../../podcasts')

    logger.info(
        f"""
        Using parms 
        podcast_path:{podcasts_path} 
        num_workers:{num_workers} 
        """)
    with ProcessPoolExecutor(
        max_workers=num_workers,
    ) as executor:
        
        futures = [
            executor.submit(make_txt_with_e, path)
            for path in get_valit_txt_paths(podcasts_path)
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Processing failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="restoring the letters ё.")
    parser.add_argument(
        "--podcasts_path",
        type=str,
        help="Path to the dataset directory."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of worker processes."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Config path."
        )
    args = parser.parse_args()
    main(args)