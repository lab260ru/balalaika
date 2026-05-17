import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
from pathlib import Path
import pickle

from dotenv import load_dotenv
from loguru import logger

from src.utils.logging_setup import setup_logging
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config

def init_client(client_key):
    try:
        from yandex_music import Client

        client = Client(client_key).init()
        return client
    except Exception as e:
        logger.error(f"Client initialization error: {e}")
        return None


def extract_podcast_id(url):
    match = re.search(r'/album/(\d+)', url)
    if match:
        return match.group(1)
    raise ValueError("Invalid URL: Unable to extract podcast ID")


def download_episode(client, part, info_podcast, folder_podcast):
    import music_tag
    import requests

    track_info = client.tracks_download_info(
        track_id=part['id'], 
        get_direct_links=True
    )
    track_info.sort(reverse=True, key=lambda k: k['bitrate_in_kbps'])
    part_download_link = track_info[0]['direct_link']

    track_file = Path(folder_podcast) / f"{part['id']}.mp3"
    track_folder = Path(folder_podcast) / f"{part['id']}"

    if track_folder.is_dir():
        logger.info(
            f"Episode '{part['title']}' already exists. Skipping download."
        )
        return

    if track_file.exists():
        logger.info(
            f"Episode '{part['title']}' already exists. Skipping download."
        )
        return

    with open(track_file, 'wb') as f:
        response = requests.get(part_download_link)
        f.write(response.content)

    logger.info(
        f"Episode '{part['title']}' downloaded from "
        f"podcast '{info_podcast['title']}'."
    )

    mp3 = music_tag.load_file(track_file)
    mp3['tracktitle'] = part['title']
    mp3['discnumber'] = part['albums'][0]['track_position']['volume']
    mp3['tracknumber'] = part['albums'][0]['track_position']['index']
    mp3['totaltracks'] = info_podcast['tracks']
    mp3['artist'] = info_podcast['title']
    mp3['album_artist'] = info_podcast['title']
    mp3['comment'] = part['short_description']
    mp3.save()

    return (
        f"Successfully downloaded episode: {part['title']} "
        f"from podcast: {info_podcast['title']}"
    )


def download_podcast(client, url, podcasts_path, episodes_limit=None, num_workers=1, episode_ids=None):
    podcast_id = extract_podcast_id(url)

    s = client.albumsWithTracks(album_id=podcast_id)
    info_podcast = {
        'id': podcast_id,
        'title': s['title'],
        'cover_url': f"https://{s['cover_uri'].replace('%%', '1000x1000')}",
        'tracks': s['track_count'],
        'short_description': s['short_description'],
        'description': s['description']
    }

    logger.info(f"Podcast: {info_podcast['title']}")

    folder_podcast = Path(podcasts_path) / str(info_podcast['id'])
    folder_podcast.mkdir(parents=True, exist_ok=True)
    volumes = s['volumes']
    episode_counter = 0
    all_parts = []

    for volume in volumes:
        for part in volume:
            if episodes_limit and episode_counter >= episodes_limit:
                break

            if episode_ids is None:
                all_parts.append(part)
                episode_counter += 1
            elif part['id'] not in episode_ids:
                continue
            else:
                all_parts.append(part)

        if episodes_limit and episode_counter >= episodes_limit:
            break

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                download_episode, 
                client, 
                part, 
                info_podcast, 
                folder_podcast
            )
            for part in all_parts
        ]
        for future in as_completed(futures):
            result = future.result()
            if result:
                logger.info(result)

    return (
        f"Successfully downloaded {len(all_parts)} episodes "
        f"from podcast: {info_podcast['title']}"
    )


def main(args):
    setup_logging("download", log_dir=args.log_dir)
    load_dotenv()
    
    client_key = os.getenv("YANDEX_KEY")
    client = init_client(client_key)

    config = load_config(args.config_path, 'download')

    if not client:
        return

    podcasts_path = config.get('podcasts_path','../../../balalaika')
    episodes_limit = config.get('episodes_limit',1)
    urls_pickle_path = config.get('podcasts_urls_file','alboms.pkl')

    try:
        with open(urls_pickle_path, 'rb') as file:
                podcasts_urls = pickle.load(file)
    except:
        logger.info('URLS not found')
        podcasts_urls = []

    logger.info(f"{len(podcasts_urls)} number of podcasts downloaded")

    num_workers = args.num_workers if args.num_workers else config.get('num_workers')
    num_workers = min(os.cpu_count(), num_workers)

    logger.info(
    f"""
    Using parms 
    podcasts_path:{podcasts_path} 
    episodes_limit:{episodes_limit} 
    num_workers:{num_workers}
    """)

    processed = 0
    errors = 0
    error_details: list[dict] = []

    for url in podcasts_urls:
        try:
            result = download_podcast(
                client=client,
                url=url,
                podcasts_path=podcasts_path,
                episodes_limit=episodes_limit,
                num_workers=num_workers
            )
            logger.info(result)
            processed += 1
        except Exception as e:
            logger.error(f"Error when downloading a podcast {url}: {e}")
            errors += 1
            error_details.append({"podcast": str(url), "reason": str(e)})

    write_stage_status(
        stage=0,
        stage_name="download",
        log_dir=args.log_dir or "./logs",
        processed=processed,
        skipped=0,
        errors=errors,
        error_details=error_details,
    )

    


    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download podcasts from Yandex Music"
    )
    parser.add_argument(
        "--config_path",
        default="./configs/config.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    parser.add_argument("--num_workers", type=int, default=None, help="Override download worker count")

    args = parser.parse_args()
    main(args)