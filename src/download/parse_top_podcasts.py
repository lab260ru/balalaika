import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from loguru import logger

from src.download.download import get_yandex_token, init_client
from src.utils.logging_setup import setup_logging


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


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

    for attr in (
        "items",
        "albums",
        "results",
        "entities",
        "blocks",
        "data",
        "podcasts",
    ):
        value = get_value(obj, attr)
        if value is not None and value is not obj:
            yield from iter_nested(value)


def looks_like_podcast(album: Any) -> bool:
    album_type = str(get_value(album, "type", "") or "").lower()
    meta_type = str(get_value(album, "meta_type", "") or "").lower()
    kind = str(get_value(album, "kind", "") or "").lower()
    title = str(get_value(album, "title", "") or "")
    description = str(get_value(album, "description", "") or "")
    short_description = str(get_value(album, "short_description", "") or "")

    haystack = " ".join(
        [album_type, meta_type, kind, title, description, short_description]
    ).lower()
    return "podcast" in haystack or "подкаст" in haystack


def album_to_record(album: Any) -> dict[str, Any] | None:
    album_id = get_value(album, "id")
    if album_id is None:
        return None

    artists = []
    for artist in get_value(album, "artists", []) or []:
        name = get_value(artist, "name")
        if name:
            artists.append(name)

    cover_uri = get_value(album, "cover_uri")
    cover_url = None
    if cover_uri:
        cover_url = f"https://{str(cover_uri).replace('%%', '1000x1000')}"

    return {
        "id": str(album_id),
        "title": get_value(album, "title", ""),
        "url": f"https://music.yandex.ru/album/{album_id}",
        "track_count": get_value(album, "track_count", 0),
        "artists": artists,
        "short_description": get_value(album, "short_description", ""),
        "description": get_value(album, "description", ""),
        "cover_url": cover_url,
    }


def add_records(
    records_by_id: dict[str, dict[str, Any]],
    source: Any,
    limit: int | None,
    require_podcast_hint: bool,
) -> bool:
    for item in iter_nested(source):
        record = album_to_record(item)
        if record is None:
            continue
        if require_podcast_hint and not looks_like_podcast(item):
            continue
        records_by_id.setdefault(record["id"], record)
        if limit and len(records_by_id) >= limit:
            return True
    return False


def collect_podcasts(client: Any, limit: int | None = None) -> list[dict[str, Any]]:
    records_by_id: dict[str, dict[str, Any]] = {}
    try:
        source = client.podcasts()
        if add_records(records_by_id, source, limit, require_podcast_hint=False):
            return list(records_by_id.values())
    except Exception as exc:
        logger.warning(f"Unable to load podcasts landing: {exc}")

    try:
        source = client.landing(["podcasts", "albums", "chart"])
        add_records(records_by_id, source, limit, require_podcast_hint=True)
    except Exception as exc:
        logger.warning(f"Unable to load landing fallback: {exc}")

    return list(records_by_id.values())


def save_pickle(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(data, file)
    logger.info(f"Saved {path}")


def main(args: argparse.Namespace) -> None:
    setup_logging("parse_top_podcasts", log_dir=args.log_dir)
    load_dotenv(args.env_path)

    client_key = get_yandex_token()
    client = init_client(client_key)
    if not client:
        logger.error("Failed to initialize Yandex Music client.")
        return

    podcasts = collect_podcasts(client, limit=args.limit)
    if not podcasts:
        logger.error("No podcasts found in Yandex Music landing.")
        return

    output_dir = Path(args.output_dir)
    urls = [podcast["url"] for podcast in podcasts]

    save_pickle(output_dir / args.catalog_name, podcasts)
    save_pickle(output_dir / args.urls_name, urls)

    logger.info(f"Collected {len(podcasts)} podcasts")
    for podcast in podcasts[:10]:
        logger.info(f"{podcast['id']}: {podcast['title']} ({podcast['url']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse top podcasts from Yandex Music and save pickle files."
    )
    parser.add_argument("--output_dir", default="./data/yandex_music")
    parser.add_argument("--catalog_name", default="top_podcasts.pkl")
    parser.add_argument("--urls_name", default="top_podcasts_urls.pkl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--env_path", default=".env")
    parser.add_argument("--log_dir", type=str, default=None)

    main(parser.parse_args())
