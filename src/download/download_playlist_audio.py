import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

from src.download.download import get_yandex_token, init_client
from src.download.download_daily_podcasts import load_pickle, save_pickle
from src.utils.logging_setup import setup_logging


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".pkl":
        return load_pickle(path, default=[])

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"JSON manifest must contain a list: {path}")
        return data

    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL line {line_number} in {path}: {exc}"
                    ) from exc
    return records


def record_audio_id(record: dict[str, Any]) -> str:
    audio_id = record.get("audio_id")
    if audio_id:
        return str(audio_id)

    track_id = str(record["track_id"])
    album_id = record.get("album_id")
    return f"{track_id}:{album_id}" if album_id else track_id


def download_record(client: Any, record: dict[str, Any], output_dir: Path) -> Path:
    import requests

    audio_id = record_audio_id(record)
    track_id = str(record["track_id"])
    safe_audio_id = audio_id.replace(":", "_")
    target = output_dir / f"{safe_audio_id}.mp3"

    if target.exists():
        logger.info(f"Already exists, skipping: {target}")
        return target

    track_info = client.tracks_download_info(
        track_id=audio_id,
        get_direct_links=True,
    )
    if not track_info and audio_id != track_id:
        track_info = client.tracks_download_info(
            track_id=track_id,
            get_direct_links=True,
        )

    track_info.sort(reverse=True, key=lambda item: item["bitrate_in_kbps"])
    direct_link = track_info[0]["direct_link"]

    tmp_target = target.with_suffix(target.suffix + ".tmp")
    response = requests.get(direct_link, timeout=120)
    response.raise_for_status()
    with open(tmp_target, "wb") as file:
        file.write(response.content)
    tmp_target.replace(target)

    logger.info(f"Downloaded {record.get('title', track_id)} -> {target}")
    return target


def download_manifest(
    client: Any,
    manifest_path: Path,
    output_root: Path,
    state_path: Path,
    num_workers: int,
    dry_run: bool,
) -> int:
    records = load_manifest(manifest_path)
    state = load_pickle(state_path, default={"downloaded_audio_ids": set()})
    downloaded_ids = set(state.get("downloaded_audio_ids", set()))

    seen = set()
    pending = []
    for record in records:
        audio_id = record_audio_id(record)
        if audio_id in seen or audio_id in downloaded_ids:
            continue
        seen.add(audio_id)
        pending.append(record)

    day_dir = output_root / datetime.now().strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Loaded {len(records)} manifest rows, "
        f"{len(pending)} new unique audios to download into {day_dir}"
    )

    if dry_run:
        for record in pending[:10]:
            logger.info(
                f"DRY RUN: would download {record_audio_id(record)} "
                f"{record.get('title', '')}"
            )
        logger.info(f"DRY RUN: {len(pending)} new unique audios, no files downloaded")
        return 0

    completed = set()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_record, client, record, day_dir): record_audio_id(record)
            for record in pending
        }
        for future in as_completed(futures):
            audio_id = futures[future]
            try:
                future.result()
                completed.add(audio_id)
            except Exception as exc:
                logger.error(f"Unable to download {audio_id}: {exc}")

    if completed:
        downloaded_ids.update(completed)
        state["downloaded_audio_ids"] = downloaded_ids
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        save_pickle(state_path, state)

    logger.info(f"Downloaded {len(completed)} new audios")
    return len(completed)


def main(args: argparse.Namespace) -> None:
    setup_logging("download_playlist_audio", log_dir=args.log_dir)
    load_dotenv(args.env_path)

    client = init_client(get_yandex_token())
    if not client:
        logger.error("Failed to initialize Yandex Music client.")
        return

    manifest_path = Path(args.manifest)
    output_root = Path(args.output_root)
    state_path = (
        Path(args.state_path)
        if args.state_path
        else output_root / ".yandex_music_download_state.pkl"
    )
    num_workers = min(os.cpu_count() or 1, args.num_workers)

    while True:
        download_manifest(
            client=client,
            manifest_path=manifest_path,
            output_root=output_root,
            state_path=state_path,
            num_workers=num_workers,
            dry_run=args.dry_run,
        )

        if args.once:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Yandex Music audio from parsed playlist manifest."
    )
    parser.add_argument("--manifest", required=True, help="Path to .pkl or .jsonl manifest.")
    parser.add_argument("--output_root", default="/mnt/ssd_1tb_2/youtube_data")
    parser.add_argument("--state_path", default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--interval_seconds", type=int, default=3600)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--env_path", default=".env")
    parser.add_argument("--log_dir", type=str, default=None)

    main(parser.parse_args())
