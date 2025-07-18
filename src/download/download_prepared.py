import argparse
import os
import re
import pickle
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from src.download.download import init_client, download_podcast, extract_podcast_id

def extract_playlist_id(url):
    match = re.search(r'(?:album|playlist)/\d+/(\d+)', url)
    if match:
        return match.group(1)
    raise ValueError("Invalid URL: Unable to extract playlist ID")

def load_urls_from_pickle(pkl_path):
    try:
        with open(pkl_path, 'rb') as f:
            urls = pickle.load(f)
        logger.info(f"Loaded {len(urls)} podcast URLs from {pkl_path}")
        return urls
    except Exception as e:
        logger.error(f"Failed to load pickle file: {e}")
        return []

def group_episodes_by_podcast(urls):
    podcast_episode_map = {}
    for url in urls:
        url = url.strip()
        podcast_id = extract_podcast_id(url)
        episode_id = extract_playlist_id(url)

        if podcast_id not in podcast_episode_map:
            podcast_episode_map[podcast_id] = set()
        podcast_episode_map[podcast_id].add(episode_id)
    return podcast_episode_map

def main(args):
    load_dotenv()

    client_key = os.getenv("YANDEX_KEY")
    client = init_client(client_key)
    if not client:
        logger.error("Failed to initialize Yandex Music client.")
        return

    urls = load_urls_from_pickle(args.pickle_path)
    if not urls:
        logger.warning("No URLs loaded from pickle file.")
        return

    podcast_episode_map = group_episodes_by_podcast(urls)

    podcasts_path = args.podcasts_path if args.podcasts_path else "../../../balalaika"
    Path(podcasts_path).mkdir(parents=True, exist_ok=True)

    num_workers = args.num_workers
    logger.info(f"Using {num_workers} workers")

    for podcast_id, episode_ids in podcast_episode_map.items():
        dummy_url = f"https://music.yandex.ru/album/{podcast_id}"
        logger.info(f"Downloading episodes {episode_ids} for podcast ID: {podcast_id}")
        try:
            result = download_podcast(
                client=client,
                url=dummy_url,
                podcasts_path=podcasts_path,
                num_workers=num_workers,
                episode_ids=episode_ids
            )
            logger.info(result)
        except Exception as e:
            logger.error(f"Error downloading podcast {podcast_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download specific episodes from Yandex Music using .pkl")
    parser.add_argument("--pickle_path", required=True, help="Path to the .pkl file with URLs")
    parser.add_argument("--podcasts_path", default=None, help="Path for saving podcasts")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for parallel downloading")

    args = parser.parse_args()
    main(args)