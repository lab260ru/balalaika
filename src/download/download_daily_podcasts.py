import argparse
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

from src.download.download import download_episode, get_track_id, get_yandex_token, init_client
from src.utils.logging_setup import setup_logging


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_album_id(url: str) -> str:
    match = re.search(r"/album/(\d+)", url)
    if not match:
        raise ValueError(f"Unable to extract album ID from URL: {url}")
    return match.group(1)


def extract_playlist_ref(url: str) -> tuple[str, str]:
    patterns = (
        r"/users/([^/]+)/playlists/(\d+)",
        r"/playlists/([^/]+)/(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1), match.group(2)
    raise ValueError(f"Unable to extract playlist ref from URL: {url}")


def is_playlist_url(url: str) -> bool:
    return "/playlists/" in url


def load_pickle(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as file:
        pickle.dump(data, file)
    tmp_path.replace(path)


def load_podcast_urls(path: Path) -> list[str]:
    data = load_pickle(path, default=[])
    urls = []

    for item in data:
        if isinstance(item, str):
            urls.append(item)
            continue

        url = get_value(item, "url")
        album_id = get_value(item, "id")
        if url:
            urls.append(url)
        elif album_id:
            urls.append(f"https://music.yandex.ru/album/{album_id}")

    return sorted(set(urls))


def load_source_urls(path: Path | None, inline_urls: list[str] | None) -> list[str]:
    urls = list(inline_urls or [])
    if path is not None:
        urls.extend(load_podcast_urls(path))
    return sorted(set(urls))


def albums_with_tracks(client: Any, album_id: str) -> Any:
    if hasattr(client, "albumsWithTracks"):
        return client.albumsWithTracks(album_id=album_id)
    return client.albums_with_tracks(album_id)


def users_playlist(client: Any, user_id: str, kind: str) -> Any:
    return client.users_playlists(kind=kind, user_id=user_id)


def podcast_info(album: Any, album_id: str) -> dict[str, Any]:
    cover_uri = get_value(album, "cover_uri")
    cover_url = None
    if cover_uri:
        cover_url = f"https://{str(cover_uri).replace('%%', '1000x1000')}"

    return {
        "id": album_id,
        "title": get_value(album, "title", album_id),
        "cover_url": cover_url,
        "tracks": get_value(album, "track_count", 0),
        "short_description": get_value(album, "short_description", ""),
        "description": get_value(album, "description", ""),
    }


def playlist_info(playlist: Any, user_id: str, kind: str) -> dict[str, Any]:
    owner = get_value(playlist, "owner")
    owner_name = get_value(owner, "name") or get_value(owner, "login") or user_id
    title = get_value(playlist, "title", f"playlist_{kind}")
    track_count = get_value(playlist, "track_count", 0)
    if not track_count:
        track_count = len(get_value(playlist, "tracks", []) or [])

    return {
        "id": f"{user_id}_{kind}",
        "title": f"{title} ({owner_name})",
        "cover_url": None,
        "tracks": track_count,
        "short_description": "",
        "description": "",
    }


def collect_episodes(album: Any, limit: int | None) -> list[Any]:
    episodes = []
    for volume in get_value(album, "volumes", []) or []:
        for part in volume:
            episodes.append(part)
            if limit and len(episodes) >= limit:
                return episodes
    return episodes


def collect_playlist_episodes(playlist: Any, limit: int | None) -> list[Any]:
    episodes = []
    tracks = get_value(playlist, "tracks", []) or []
    for item in tracks:
        track = get_value(item, "track")
        if track is None and hasattr(item, "fetch_track"):
            track = item.fetch_track()
        if track is None:
            track = item
        episodes.append(track)
        if limit and len(episodes) >= limit:
            return episodes
    return episodes


def download_new_episodes(
    client: Any,
    urls: list[str],
    output_root: Path,
    state_path: Path,
    episodes_per_podcast: int | None,
    num_workers: int,
) -> int:
    state = load_pickle(state_path, default={"downloaded_episode_ids": set()})
    downloaded_ids = set(state.get("downloaded_episode_ids", set()))
    day_dir = output_root / datetime.now().strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for url in urls:
        try:
            if is_playlist_url(url):
                user_id, kind = extract_playlist_ref(url)
                playlist = users_playlist(client, user_id=user_id, kind=kind)
                info = playlist_info(playlist, user_id=user_id, kind=kind)
                source_dir = day_dir / f"playlist_{user_id}_{kind}"
                episodes = collect_playlist_episodes(playlist, episodes_per_podcast)
            else:
                album_id = extract_album_id(url)
                album = albums_with_tracks(client, album_id)
                info = podcast_info(album, album_id)
                source_dir = day_dir / album_id
                episodes = collect_episodes(album, episodes_per_podcast)

            source_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Prepared {len(episodes)} episodes from {url}")

            for episode in episodes:
                current_episode_id = get_track_id(episode)
                if current_episode_id in downloaded_ids:
                    continue
                jobs.append((episode, info, source_dir, current_episode_id))
        except Exception as exc:
            logger.error(f"Unable to prepare source {url}: {exc}")

    if not jobs:
        logger.info("No new episodes to download.")
        return 0

    completed_ids = set()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_episode, client, episode, info, podcast_dir): item_id
            for episode, info, podcast_dir, item_id in jobs
        }
        for future in as_completed(futures):
            item_id = futures[future]
            try:
                result = future.result()
                if result:
                    logger.info(result)
                completed_ids.add(item_id)
            except Exception as exc:
                logger.error(f"Unable to download episode {item_id}: {exc}")

    if completed_ids:
        downloaded_ids.update(completed_ids)
        state["downloaded_episode_ids"] = downloaded_ids
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        save_pickle(state_path, state)

    logger.info(f"Downloaded {len(completed_ids)} new episodes into {day_dir}")
    return len(completed_ids)


def main(args: argparse.Namespace) -> None:
    setup_logging("download_daily_podcasts", log_dir=args.log_dir)
    load_dotenv(args.env_path)

    client_key = get_yandex_token()
    client = init_client(client_key)
    if not client:
        logger.error("Failed to initialize Yandex Music client.")
        return

    pickle_path = Path(args.podcasts_pickle) if args.podcasts_pickle else None
    urls = load_source_urls(pickle_path, args.playlist_url)
    if not urls:
        logger.error("No playlist or podcast URLs provided.")
        return

    output_root = Path(args.output_root)
    state_path = Path(args.state_path) if args.state_path else output_root / ".yandex_music_download_state.pkl"
    num_workers = min(os.cpu_count() or 1, args.num_workers)

    logger.info(
        f"Loaded {len(urls)} sources, output_root={output_root}, "
        f"state_path={state_path}, interval={args.interval_seconds}s"
    )

    while True:
        download_new_episodes(
            client=client,
            urls=urls,
            output_root=output_root,
            state_path=state_path,
            episodes_per_podcast=args.episodes_per_podcast,
            num_workers=num_workers,
        )

        if args.once:
            break

        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continuously download new Yandex Music podcast episodes by day."
    )
    parser.add_argument(
        "--podcasts_pickle",
        default=None,
        help="Pickle with podcast/playlist records or URLs.",
    )
    parser.add_argument(
        "--playlist_url",
        action="append",
        default=[],
        help="Yandex Music playlist URL. Can be passed multiple times.",
    )
    parser.add_argument("--output_root", default="/mnt/ssd_1tb_2/youtube_data")
    parser.add_argument("--state_path", default=None)
    parser.add_argument("--episodes_per_podcast", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--interval_seconds", type=int, default=3600)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--env_path", default=".env")
    parser.add_argument("--log_dir", type=str, default=None)

    main(parser.parse_args())
