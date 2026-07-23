import argparse
import pandas as pd
import os
from pydub import AudioSegment
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

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

        # Compute every output segment's path FIRST (names depend only on the
        # metadata, not the audio). If all segments already exist, return before
        # decoding the whole episode into RAM — on resume this turns a full
        # re-decode of every completed episode (eternal, since `saved` would stay
        # False and the source was kept) into a metadata-only no-op. Same files
        # produced/skipped and same source-removal semantics as before.
        pending = []  # (start_ms, end_ms, dest_path) for segments not yet written
        for _, row in episode_meta.iterrows():
            start_sec = float(row['start'])
            end_sec = float(row['end'])

            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)

            new_name = f"{start_sec:.2f}_{end_sec:.2f}_{playlist_id}_{podcast_id}.mp3"
            dest_path = os.path.join(dest_dir, new_name)

            if os.path.exists(dest_path):
                continue
            pending.append((start_ms, end_ms, dest_path))

        if not pending:
            # Nothing to export -> original kept the source (saved == False).
            return

        audio = AudioSegment.from_mp3(audio_path)

        saved = False
        for start_ms, end_ms, dest_path in pending:
            segment = audio[start_ms:end_ms]
            segment.export(dest_path, format="mp3")
            saved = True

        if saved and os.listdir(dest_dir):
            os.remove(audio_path)

    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")


def main(args):
    parquet_path = args.parquet_path
    podcasts_path = args.podcasts_path
    num_workers = args.num_workers

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
    parser.add_argument(
        "--podcasts_path",
        type=str,
        help="Path to podcasts directory"
        )
    parser.add_argument(
        "--parquet_path",
        type=str,
        help="Path to parquet file"
        )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of parallel workers"
        )
    args = parser.parse_args()
    main(args)