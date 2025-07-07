import argparse
import pandas as pd
import os
from pydub import AudioSegment
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from src.utils import load_config

def process_audio_file(
    audio_path: str,
    podcasts_path: str,
    metadata_grouped: pd.core.groupby.generic.DataFrameGroupBy
):
    try:
        parts = audio_path.split(os.sep)
        playlist_id = int(parts[-2])
        podcast_id = int(parts[-1].replace('.mp3', ''))

        key = (playlist_id, podcast_id)
        if key not in metadata_grouped.groups:
            logger.info(f"Meta not found for {playlist_id}/{podcast_id}")
            return

        episode_meta = metadata_grouped.get_group(key)
        dest_dir = os.path.join(podcasts_path, str(playlist_id), str(podcast_id))
        os.makedirs(dest_dir, exist_ok=True)

        audio = AudioSegment.from_mp3(audio_path)

        saved = False
        for _, row in episode_meta.iterrows():
            start_sec = float(row['start'])
            end_sec = float(row['end'])
            
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)
            
            segment = audio[start_ms:end_ms]
            
            new_name = f"{start_sec:.2f}_{end_sec:.2f}_{playlist_id}_{podcast_id}.mp3"
            dest_path = os.path.join(dest_dir, new_name)
            
            if os.path.exists(dest_path):
                continue

            segment.export(dest_path, format="mp3")
            saved = True

        if saved and os.listdir(dest_dir):
            os.remove(audio_path)

    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")


def main(args):
    config = load_config(args.config_path, 'download')
    parquet_path = args.parquet_path or config.get('parquet_path', '../../balalaika.parquet')
    podcasts_path = args.podcasts_path or config.get('podcasts_path', '../../../balalaika')
    num_workers = args.num_workers or config.get('num_workers', 2)

    logger.info("Loading metadata...")
    df = pd.read_parquet(parquet_path)

    logger.info("Grouping metadata by podcast...")
    metadata_grouped = df.groupby(['playlist_id', 'podcast_id'])

    logger.info("Collecting audio files...")
    audio_paths = []
    for root, _, files in os.walk(podcasts_path):
        for file in files:
            if file.endswith('.mp3') and len(file.split('_')) < 4:
                full_path = os.path.join(root, file)
                audio_paths.append(full_path)

    logger.info(f"Found {len(audio_paths)} audio files to process")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_audio_file,
                audio,
                podcasts_path,
                metadata_grouped
            )
            for audio in audio_paths
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in thread: {e}")

    logger.info("All files processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Dataset using meta")
    parser.add_argument("--parquet_path", "-p", type=str, help="Path to parquet file")
    parser.add_argument("--config_path", "-c", type=str, help="Path to config file")
    parser.add_argument("--podcasts_path", "-d", type=str, help="Path to podcasts directory")
    parser.add_argument("--num_workers", "-n", type=int, help="Number of parallel workers")
    args = parser.parse_args()
    main(args)