import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from loguru import logger

from src.download.download import get_track_id, get_yandex_token, init_client
from src.download.download_daily_podcasts import (
    albums_with_tracks,
    collect_episodes,
    extract_playlist_ref,
    get_value,
    load_pickle,
    save_pickle,
    users_playlist,
)
from src.utils.logging_setup import setup_logging


DEFAULT_PODCAST_CHART_URL = "https://music.yandex.ru/entities/editorial-compilation/editorial_audiobooks_CHART"


def iter_nested(obj: Any) -> Iterable[Any]:
    if obj is None:
        return
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from iter_nested(value)
        return
    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            yield from iter_nested(value)
        return

    yield obj
    for attr in ("items", "albums", "results", "entities", "blocks", "data", "podcasts"):
        value = get_value(obj, attr)
        if value is not None and value is not obj:
            yield from iter_nested(value)


def album_id_from_object(item: Any) -> str | None:
    album_id = get_value(item, "id") or get_value(item, "album_id")
    if album_id is None:
        return None
    return str(album_id)


def safe_page_name(url: str) -> str:
    slug = re.sub(r"^https?://", "", url)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", slug).strip("_")
    return slug[:160] or "page"


def discover_podcast_album_ids_from_page(
    url: str,
    limit: int | None,
    html_cache_path: Path | None = None,
) -> list[str]:
    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    }
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    html = response.text

    if html_cache_path is not None:
        html_cache_path.parent.mkdir(parents=True, exist_ok=True)
        html_cache_path.write_text(html, encoding="utf-8")
        logger.info(f"Saved chart HTML to {html_cache_path}")

    patterns = (
        r"/album/(\d+)",
        r"\\u002Falbum\\u002F(\d+)",
        r"\\/album\\/(\d+)",
    )
    ids_by_value = {}
    for pattern in patterns:
        for match in re.finditer(pattern, html):
            album_id = match.group(1)
            ids_by_value.setdefault(album_id, album_id)
            if limit and len(ids_by_value) >= limit:
                logger.info(f"Discovered {len(ids_by_value)} albums from {url}")
                return list(ids_by_value.values())

    logger.info(f"Discovered {len(ids_by_value)} albums from {url}")
    if not ids_by_value:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
        if title_match:
            title = re.sub(r"\s+", " ", title_match.group(1)).strip()
            logger.warning(f"No album links found. Page title: {title}")
    return list(ids_by_value.values())


def discover_podcast_album_ids(client: Any, limit: int | None) -> list[str]:
    ids_by_value = {}
    sources = []

    for name, loader in (
        ("podcasts", lambda: client.podcasts()),
        ("landing_podcasts", lambda: client.landing(["podcasts"])),
    ):
        try:
            sources.append((name, loader()))
        except Exception as exc:
            logger.warning(f"Unable to discover podcasts from {name}: {exc}")

    for source_name, source in sources:
        for item in iter_nested(source):
            album_id = album_id_from_object(item)
            if album_id is None:
                continue
            ids_by_value.setdefault(album_id, album_id)
            if limit and len(ids_by_value) >= limit:
                logger.info(f"Discovered {len(ids_by_value)} podcast albums from {source_name}")
                return list(ids_by_value.values())

    logger.info(f"Discovered {len(ids_by_value)} podcast albums")
    return list(ids_by_value.values())


def artist_names(track: Any) -> list[str]:
    names = []
    for artist in get_value(track, "artists", []) or []:
        name = get_value(artist, "name")
        if name:
            names.append(name)
    return names


def album_record(track: Any) -> dict[str, Any] | None:
    albums = get_value(track, "albums", []) or []
    if not albums:
        return None

    album = albums[0]
    album_id = get_value(album, "id")
    return {
        "id": str(album_id) if album_id is not None else None,
        "title": get_value(album, "title", ""),
        "url": f"https://music.yandex.ru/album/{album_id}" if album_id else None,
    }


def track_record(
    track: Any,
    source_url: str,
    source_type: str,
    source_id: str,
    source_title: str,
) -> dict[str, Any]:
    track_id = get_track_id(track)
    album = album_record(track)
    album_id = album["id"] if album else get_value(track, "album_id")
    audio_id = f"{track_id}:{album_id}" if album_id is not None else track_id

    return {
        "audio_id": audio_id,
        "track_id": track_id,
        "album_id": str(album_id) if album_id is not None else None,
        "title": get_value(track, "title", track_id),
        "artists": artist_names(track),
        "duration_ms": get_value(track, "duration_ms"),
        "short_description": get_value(track, "short_description", ""),
        "source_type": source_type,
        "source_id": source_id,
        "source_title": source_title,
        "source_url": source_url,
        "album": album,
        "track_url": (
            f"https://music.yandex.ru/album/{album_id}/track/{track_id}"
            if album_id is not None
            else None
        ),
        "parsed_at": datetime.now().isoformat(timespec="seconds"),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)
    logger.info(f"Saved {path}")


def load_manual_playlist_source(client: Any, playlist_url: str, limit: int | None) -> dict[str, Any] | None:
    try:
        user_id, kind = extract_playlist_ref(playlist_url)
        playlist = users_playlist(client, user_id=user_id, kind=kind)
    except Exception as exc:
        logger.error(f"Unable to parse manual playlist {playlist_url}: {exc}")
        return None

    tracks = []
    for item in get_value(playlist, "tracks", []) or []:
        track = get_value(item, "track")
        if track is None and hasattr(item, "fetch_track"):
            track = item.fetch_track()
        if track is not None:
            tracks.append(track)
        if limit and len(tracks) >= limit:
            break

    return {
        "source_type": "manual_playlist",
        "source_id": f"{user_id}:{kind}",
        "source_title": get_value(playlist, "title", playlist_url),
        "source_url": playlist_url,
        "tracks": tracks,
    }


def build_sources(
    client: Any,
    podcast_album_ids: list[str],
    playlist_urls: list[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    sources = []

    for album_id in sorted(set(podcast_album_ids)):
        try:
            album = albums_with_tracks(client, album_id)
            tracks = collect_episodes(album, limit=limit)
            title = get_value(album, "title", album_id)
            sources.append(
                {
                    "source_type": "podcast_album",
                    "source_id": album_id,
                    "source_title": title,
                    "source_url": f"https://music.yandex.ru/album/{album_id}",
                    "tracks": tracks,
                }
            )
        except Exception as exc:
            logger.error(f"Unable to parse podcast album {album_id}: {exc}")

    for playlist_url in playlist_urls:
        logger.warning(
            f"Manual playlist source may include music, not only podcasts: {playlist_url}"
        )
        source = load_manual_playlist_source(client, playlist_url, limit=limit)
        if source:
            sources.append(source)

    return sources


def parse_sources(
    sources: list[dict[str, Any]],
    output_dir: Path,
    state_path: Path,
    include_seen: bool,
) -> list[dict[str, Any]]:
    state = load_pickle(state_path, default={"parsed_audio_ids": set()})
    parsed_ids = set(state.get("parsed_audio_ids", set()))
    seen_this_run = set()
    records = []

    for source in sources:
        logger.info(
            f"Parsed {source['source_type']} {source['source_url']}: "
            f"{len(source['tracks'])} tracks"
        )
        for track in source["tracks"]:
            record = track_record(
                track=track,
                source_url=source["source_url"],
                source_type=source["source_type"],
                source_id=source["source_id"],
                source_title=source["source_title"],
            )
            audio_id = record["audio_id"]
            if audio_id in seen_this_run:
                continue
            seen_this_run.add(audio_id)
            if not include_seen and audio_id in parsed_ids:
                continue
            records.append(record)

    if records:
        parsed_ids.update(record["audio_id"] for record in records)
        state["parsed_audio_ids"] = parsed_ids
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        save_pickle(state_path, state)

    day = datetime.now().strftime("%Y%m%d")
    save_pickle(output_dir / f"podcast_audio_{day}.pkl", records)
    write_jsonl(output_dir / f"podcast_audio_{day}.jsonl", records)

    logger.info(
        f"Saved {len(records)} new unique podcast tracks. "
        f"Known parsed tracks in state: {len(parsed_ids)}"
    )
    return records


def main(args: argparse.Namespace) -> None:
    setup_logging("parse_playlist_audio", log_dir=args.log_dir)
    load_dotenv(args.env_path)

    client = init_client(get_yandex_token())
    if not client:
        logger.error("Failed to initialize Yandex Music client.")
        return

    output_dir = Path(args.output_dir)
    podcast_album_ids = list(args.podcast_album_id)
    page_urls = list(args.page_url)
    if args.chart_url:
        page_urls.append(args.chart_url)

    for page_url in sorted(set(page_urls)):
        html_cache_path = None
        if args.save_page_html:
            html_cache_path = output_dir / f"{safe_page_name(page_url)}.html"
        podcast_album_ids.extend(
            discover_podcast_album_ids_from_page(
                url=page_url,
                limit=args.podcast_limit,
                html_cache_path=html_cache_path,
            )
        )
    if args.discover_podcasts:
        podcast_album_ids.extend(
            discover_podcast_album_ids(client, limit=args.podcast_limit)
        )

    sources = build_sources(
        client=client,
        podcast_album_ids=podcast_album_ids,
        playlist_urls=args.playlist_url,
        limit=args.limit,
    )
    if not sources:
        logger.error("No podcast sources found.")
        return

    state_path = (
        Path(args.state_path)
        if args.state_path
        else output_dir / ".yandex_music_parse_state.pkl"
    )
    parse_sources(
        sources=sources,
        output_dir=output_dir,
        state_path=state_path,
        include_seen=args.include_seen,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse Yandex Music podcast episodes without downloading."
    )
    parser.add_argument(
        "--chart_url",
        default=DEFAULT_PODCAST_CHART_URL,
        help="Backward-compatible single page URL. Defaults to the podcast/audiobook chart.",
    )
    parser.add_argument(
        "--page_url",
        action="append",
        default=[],
        help="Yandex Music page URL to parse for /album/<id>. Can be passed multiple times.",
    )
    parser.add_argument(
        "--save_page_html",
        action="store_true",
        help="Save downloaded HTML pages into output_dir for debugging.",
    )
    parser.add_argument(
        "--discover_podcasts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also try Yandex Music API podcasts landing. Currently less reliable.",
    )
    parser.add_argument("--podcast_limit", type=int, default=1000)
    parser.add_argument(
        "--podcast_album_id",
        action="append",
        default=[],
        help="Podcast album ID. Can be passed multiple times.",
    )
    parser.add_argument(
        "--playlist_url",
        action="append",
        default=[],
        help="Manual playlist URL. Use only if it is known to contain podcasts.",
    )
    parser.add_argument("--output_dir", default="./data/yandex_music")
    parser.add_argument("--state_path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include_seen",
        action="store_true",
        help="Include tracks already stored in parse state.",
    )
    parser.add_argument("--env_path", default=".env")
    parser.add_argument("--log_dir", type=str, default=None)

    main(parser.parse_args())
