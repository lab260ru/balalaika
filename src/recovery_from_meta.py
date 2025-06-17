import argparse
import pandas as pd
import os
import torchaudio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
from loguru import logger

from src.utils import load_config

# TODO: fix code duplicate
def get_audio_paths(directory: str) -> List[str]:
    audio_paths = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if len(os.path.basename(full_path).split('_')) == 4:
            continue
        if os.path.isdir(full_path):
            audio_paths.extend(get_audio_paths(full_path))
        elif entry.endswith(".mp3") :
            audio_paths.append(full_path)
    return audio_paths

def process_audio_file(audio: str, podcasts_path: str, metadata: pd.DataFrame) -> None:
    parts = audio.split(os.sep)
    playlist_id = int(parts[-2])
    podcast_id = int(parts[-1].replace('.mp3', ''))

    episode_meta = metadata[
        (metadata['playlist_id'] == playlist_id) &
        (metadata['podcast_id'] == podcast_id)
    ]

    if episode_meta.empty:
        logger.info(f"Meta not found for {playlist_id}/{podcast_id}")
        return
    
    for index, row in episode_meta.iterrows():
        start = float(row['start'])
        end = float(row['end'])

        src_audio, sr = torchaudio.load(audio)

        dest_dir = os.path.join(podcasts_path, str(playlist_id), str(podcast_id))
        os.makedirs(dest_dir, exist_ok=True)

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        new_audio = src_audio[:, start_sample:end_sample]

        new_name = f"{start:.2f}_{end:.2f}_{playlist_id}_{podcast_id}.mp3"
        dest_path = os.path.join(dest_dir, new_name)

        torchaudio.save(
            dest_path,
            new_audio,
            sr
        )
        
    if len(os.listdir(dest_dir)) != 0 :
        os.remove(audio)

def main(args):
    config = load_config(args.config_path, 'download')
    parquet_path = config.get('parquet_path', '../../balalaika.parquet') if args.parquet_path is None else args.parquet_path
    podcasts_path = config.get('podcasts_path','../../../podcasts') if args.podcasts_path is None else args.podcasts_path
    num_workers = config.get('num_workers', 8) if args.num_workers is None else args.num_workers

    df = pd.read_parquet(parquet_path)
    audio_paths = get_audio_paths(podcasts_path)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_audio_file,
                audio,
                podcasts_path,
                df
            )
            for audio in audio_paths
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Error in process: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Dataset using meta")
    parser.add_argument(
        "--parquet_path",
        "-p",
        type=str,
        help="Path to parquet file"
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        help="Path to config file"
    )
    parser.add_argument(
        "--podcasts_path",
        "-d",
        type=str,
        help="Path to config file"
    )
    parser.add_argument(
        "--num_workers",
        "-n",
        type=int,
        help="Path to config file"
    )
    args = parser.parse_args()
    main(args)