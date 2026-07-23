"""Preprocess metadata backfill for datasets that are already chunked.

The default path does not cut or rewrite audio. It diarizes each existing
audio file and writes the same metadata columns that the normal preprocess
chunking stage produces. When ``fuse_audio_preprocessing`` is enabled, it also
does one native-rate read for crest filtering plus LUFS normalization and
rewrites each kept file at most once.

* speaker_id
* start / end
* total_duration
* playlist_id / podcast_id
* silence_percent / max_silence_duration
* is_single_speaker
"""
from __future__ import annotations

import argparse
import io
import multiprocessing
import re
import time
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Set

import pandas as pd
import torch
import torch.multiprocessing as mp
import torchaudio
from loguru import logger
from tqdm import tqdm

from src.preprocess.audio_postprocessing import (
    fused_audio_preprocessing_enabled,
    postprocess_audio_tensor,
    save_audio_atomic,
)
from src.preprocess.preprocess import (
    DEFAULT_CHUNK_DURATION_S,
    FUSED_PARTIAL_FIELDS,
    PARTIAL_FIELDS,
    PARTIAL_PREFIX,
    diarize_audio,
    get_chunk_metrics,
    init_models,
    single_speaker_only_enabled,
)
from src.utils.audit import record_stage_summary
from src.utils.csv_manager import (
    PartialCsvWriter,
    PeriodicCsvMerger,
    absorb_partial_csvs,
    discover_audio_paths,
    ensure_main_csv,
    load_csv_settings,
    load_main_csv,
    normalize_path_string,
)
from src.utils.datasets.preprocess import create_diarization_dataloader
from src.utils.logging_setup import setup_logging
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config
from src.utils.work_shards import (
    claim_work_shard,
    load_work_shard_size,
    mark_work_shard_done,
    prepare_work_shards,
    read_work_shard,
)

METADATA_VALUE_COLUMNS = [column for column in PARTIAL_FIELDS if column != "filepath"]
VALUE_COLUMNS = [column for column in FUSED_PARTIAL_FIELDS if column != "filepath"]
DEFAULT_AUDIO_PATHS_SOURCE = "auto"

CHUNK_STEM_PREFIX_RE = re.compile(
    r"^(?P<start>\d+(?:\.\d+)?)_(?P<end>\d+(?:\.\d+)?)(?:_|$)"
)
CHUNK_STEM_IDS_RE = re.compile(
    r"^(?P<start>\d+(?:\.\d+)?)_"
    r"(?P<end>\d+(?:\.\d+)?)_"
    r"(?P<playlist>[^_]+)_"
    r"(?P<podcast>.+)$"
)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _falsey(value: object) -> bool:
    return str(value).strip().lower() in {"0", "false", "no", "n", "off"}



def _metadata_complete_mask(df: pd.DataFrame, required_columns: Sequence[str]) -> pd.Series:
    if df.empty or "filepath" not in df.columns:
        return pd.Series(False, index=df.index)

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        return pd.Series(False, index=df.index)

    mask = df["filepath"].notna() & df["filepath"].astype(str).str.strip().ne("")
    for column in required_columns:
        column_mask = df[column].notna()
        if df[column].dtype == object:
            column_mask &= df[column].astype(str).str.strip().ne("")
        mask &= column_mask
    return mask


def pending_metadata_paths(
    podcasts_path: Path,
    audio_paths: Iterable[str],
    *,
    overwrite: bool = False,
    required_columns: Sequence[str] = METADATA_VALUE_COLUMNS,
    single_speaker_only: bool = False,
) -> List[str]:
    paths = [normalize_path_string(path) for path in audio_paths if str(path).strip()]
    if overwrite:
        return paths

    df = load_main_csv(podcasts_path)
    if df.empty:
        return paths

    done_mask = _metadata_complete_mask(df, required_columns)
    done = set(df.loc[done_mask, "filepath"].astype(str).map(normalize_path_string))
    known_multi_speaker = set()
    if single_speaker_only and "is_single_speaker" in df.columns:
        normalized_paths = df["filepath"].astype(str).map(normalize_path_string)
        false_mask = df["is_single_speaker"].map(_falsey)
        known_multi_speaker = set(normalized_paths[false_mask])
    pending = [path for path in paths if path not in done or path in known_multi_speaker]
    logger.info(
        f"Existing-chunk metadata: {len(done)} complete row(s), "
        f"{len(pending)} pending out of {len(paths)} audio file(s)."
    )
    return pending


def _ids_from_path(audio_path: Path, podcasts_path: Path, duration_s: float) -> dict:
    """Derive preprocess-compatible row identity from a chunk path."""
    try:
        relative = audio_path.resolve().relative_to(podcasts_path.resolve())
        parts = relative.parts
    except Exception:
        parts = audio_path.parts

    if len(parts) >= 3:
        playlist_id = parts[-3]
        podcast_id = parts[-2]
    else:
        legacy_match = CHUNK_STEM_IDS_RE.match(audio_path.stem)
        if legacy_match:
            playlist_id = legacy_match.group("playlist")
            podcast_id = legacy_match.group("podcast")
        elif len(parts) >= 2:
            playlist_id = parts[-2]
            podcast_id = audio_path.stem
        else:
            playlist_id = audio_path.parent.name
            podcast_id = audio_path.stem

    range_match = CHUNK_STEM_PREFIX_RE.match(audio_path.stem)
    if range_match:
        start = float(range_match.group("start"))
        end = float(range_match.group("end"))
    else:
        start = 0.0
        end = duration_s
    if end <= start:
        start, end = 0.0, duration_s

    return {
        "start": round(start, 2),
        "end": round(end, 2),
        "playlist_id": playlist_id,
        "podcast_id": podcast_id,
    }


def metadata_for_chunk(
    path_audio: str,
    audio: torch.Tensor,
    sr: int,
    podcasts_path: Path,
    config: Mapping[str, object],
) -> dict:
    duration_s = float(audio.shape[-1]) / float(sr) if sr else 0.0
    audio_path = Path(path_audio)
    identity = _ids_from_path(audio_path, podcasts_path, duration_s)

    if duration_s <= 0:
        return {
            "filepath": normalize_path_string(path_audio),
            "speaker_id": -1,
            "total_duration": 0.0,
            "silence_percent": 100.0,
            "max_silence_duration": 0.0,
            "is_single_speaker": False,
            **identity,
        }

    chunk_duration = float(config.get("chunk_duration", DEFAULT_CHUNK_DURATION_S))
    raw_segments = diarize_audio(audio, sr, chunk_duration)
    if raw_segments:
        silence_percent, max_silence_duration, speaker_count = get_chunk_metrics(
            0.0,
            duration_s,
            raw_segments,
        )
        speaker_id = int(raw_segments[0][2])
        is_single_speaker = speaker_count == 1
    else:
        silence_percent = 100.0
        max_silence_duration = round(duration_s, 2)
        speaker_id = -1
        is_single_speaker = False

    return {
        "filepath": normalize_path_string(path_audio),
        "speaker_id": speaker_id,
        "total_duration": round(duration_s, 2),
        "silence_percent": silence_percent,
        "max_silence_duration": max_silence_duration,
        "is_single_speaker": is_single_speaker,
        **identity,
    }


def _increment(counter, amount: float = 1) -> None:
    if counter is None:
        return
    try:
        with counter.get_lock():
            counter.value += amount
    except AttributeError:
        counter.value += amount


def _postprocess_existing_chunk(
    path_audio: str,
    config: Mapping[str, object],
    raw_bytes: bytes | None = None,
) -> tuple[bool, float, bool, float, bool]:
    load_started_at = time.perf_counter()
    # Reuse the bytes the DataLoader already read from disk when available so the
    # native-rate decode does not trigger a second cold-cache HDD read. torchcodec
    # decodes a ``bytes`` source bit-identically to a path source.
    decode_source = io.BytesIO(raw_bytes) if raw_bytes is not None else path_audio
    native_audio, native_sr = torchaudio.load_with_torchcodec(decode_source)
    native_audio = native_audio.to(dtype=torch.float32).contiguous()
    logger.debug(
        f"perf audio_load stage=preprocess_existing_chunks path={path_audio} "
        f"seconds={time.perf_counter() - load_started_at:.6f} "
        f"sample_rate={int(native_sr)} frames={int(native_audio.shape[-1])} "
        f"source={'bytes' if raw_bytes is not None else 'path'}"
    )

    result = postprocess_audio_tensor(
        native_audio,
        int(native_sr),
        crest_threshold=float(
            config.get("crest_threshold", config.get("crest_treshold", 10.0))
        ),
        peak=float(config.get("peak", -1.0)),
        loudness=float(config.get("loudness", -23.0)),
        block_size=float(config.get("block_size", 0.400)),
    )
    duration_s = float(native_audio.shape[-1]) / float(native_sr)

    if not result.keep:
        Path(path_audio).unlink()
        logger.debug(
            f"Deleted {path_audio} before rewrite "
            f"(crest_factor={result.crest_factor:.2f})"
        )
        return False, result.crest_factor, False, duration_s, False

    if result.loudness_normalized:
        try:
            save_started_at = time.perf_counter()
            # Atomic tmp+os.replace in the same dir: a crash mid-encode can no
            # longer truncate the source file (bytes identical to direct save).
            save_audio_atomic(path_audio, result.samples, int(native_sr))
            logger.debug(
                f"perf audio_save stage=preprocess_existing_chunks path={path_audio} "
                f"seconds={time.perf_counter() - save_started_at:.6f} "
                f"sample_rate={int(native_sr)} frames={int(result.samples.shape[-1])}"
            )
        except Exception as exc:
            logger.error(f"Fused audio write failed for {path_audio}: {exc}")
            return True, result.crest_factor, False, duration_s, True
    elif result.loudness_error:
        logger.error(
            f"Fused loudness normalization failed for {path_audio}: "
            f"{result.loudness_error}; leaving the original audio unchanged."
        )

    return (
        True,
        result.crest_factor,
        result.loudness_normalized,
        duration_s,
        bool(result.loudness_error),
    )


def _process_files(
    rank: int,
    files: List[str],
    config: Mapping[str, object],
    podcasts_path: Path,
    writer: PartialCsvWriter,
    already_done: Set[str],
    processed_counter,
    skipped_counter,
    errors_counter,
    single_speaker_dropped,
    crest_files_in,
    crest_files_out,
    crest_duration_in,
    crest_duration_out,
) -> None:
    fuse_audio = fused_audio_preprocessing_enabled(config)
    single_speaker_only = single_speaker_only_enabled(config)
    partial_fields = FUSED_PARTIAL_FIELDS if fuse_audio else PARTIAL_FIELDS
    pending_files = []
    for path in files:
        resolved = normalize_path_string(path)
        if resolved in already_done:
            _increment(skipped_counter)
            continue
        pending_files.append(path)

    if not pending_files:
        return

    # Pre-chunked inputs are short clips (the prior chunking stage caps them at
    # ``duration``, default 15 s ≈ 0.5-2 MB each). When fusing crest/loudness we
    # read each file's bytes once in the loader and reuse them for the native-rate
    # decode, so each chunk leaves the HDD once. The cap (``existing_chunks_raw_bytes_max_s``,
    # default 4x ``duration`` for headroom) keeps an unexpectedly long file from
    # ballooning prefetch RAM; oversized files fall back to a second path decode.
    if fuse_audio:
        default_cap = 4.0 * float(config.get("duration", 15))
        raw_bytes_max_duration_s = float(
            config.get("existing_chunks_raw_bytes_max_s", default_cap)
        )
    else:
        raw_bytes_max_duration_s = None
    dataloader = create_diarization_dataloader(
        pending_files,
        batch_size=int(config.get("diarization_batch_size", 1)),
        num_workers=int(config.get("diarization_loader_workers", 0)),
        prefetch_factor=int(config.get("diarization_prefetch_factor", 2)),
        raw_bytes_max_duration_s=raw_bytes_max_duration_s,
    )

    batch_wait_started_at = time.perf_counter()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Chunks-{rank}", position=rank)):
        batch_received_at = time.perf_counter()
        logger.debug(
            f"perf dataloader_wait stage=preprocess_existing_chunks rank={rank} "
            f"batch={batch_idx} seconds={batch_received_at - batch_wait_started_at:.6f} "
            f"items={len(batch)}"
        )
        for path_audio, audio, sr, error, raw_bytes in batch:
            resolved = normalize_path_string(path_audio)
            if resolved in already_done:
                _increment(skipped_counter)
                continue
            if error:
                logger.error(f"Broken chunk {path_audio}: {error}")
                _increment(errors_counter)
                continue

            try:
                row = metadata_for_chunk(str(path_audio), audio, sr, podcasts_path, config)
                if single_speaker_only and not _truthy(row.get("is_single_speaker")):
                    try:
                        Path(path_audio).unlink(missing_ok=True)
                    except OSError as exc:
                        logger.error(f"Failed to delete multi-speaker chunk {path_audio}: {exc}")
                        _increment(errors_counter)
                        continue
                    already_done.add(resolved)
                    _increment(processed_counter)
                    _increment(single_speaker_dropped)
                    logger.debug(f"Deleted multi-speaker chunk due to single_speaker_only: {path_audio}")
                    continue

                if fuse_audio:
                    keep, crest_factor, normalized, duration_s, postprocess_error = (
                        _postprocess_existing_chunk(str(path_audio), config, raw_bytes)
                    )
                    if postprocess_error:
                        _increment(errors_counter)
                    _increment(crest_files_in)
                    _increment(crest_duration_in, duration_s)
                    if not keep:
                        already_done.add(resolved)
                        _increment(processed_counter)
                        continue
                    _increment(crest_files_out)
                    _increment(crest_duration_out, duration_s)
                    row["crest_factor"] = round(crest_factor, 4)
                    row["loudness_normalized"] = True if normalized else ""

                writer.write({column: row.get(column, "") for column in partial_fields})
                already_done.add(resolved)
                _increment(processed_counter)
            except Exception as exc:
                logger.error(f"Metadata extraction failed for {path_audio}: {exc}")
                _increment(errors_counter)
        batch_wait_started_at = time.perf_counter()


def run_worker(
    rank: int,
    world_size: int,
    work_dir: str,
    config: Mapping[str, object],
    config_path: str,
    podcasts_path_str: str,
    processed_counter,
    skipped_counter,
    errors_counter,
    single_speaker_dropped,
    crest_files_in,
    crest_files_out,
    crest_duration_in,
    crest_duration_out,
) -> None:
    podcasts_path = Path(podcasts_path_str)
    init_models(rank, dict(config), config_path)

    claimed = 0
    partial_fields = (
        FUSED_PARTIAL_FIELDS
        if fused_audio_preprocessing_enabled(config)
        else PARTIAL_FIELDS
    )
    with PartialCsvWriter(
        podcasts_path, PARTIAL_PREFIX, rank, fieldnames=partial_fields
    ) as writer:
        already_done = writer.already_done()
        if already_done:
            logger.info(
                f"Worker {rank}: {len(already_done)} row(s) already in partial; skipping."
            )

        while True:
            shard_path = claim_work_shard(work_dir, rank)
            if shard_path is None:
                break
            shard_files = read_work_shard(shard_path)
            claimed += 1
            logger.info(
                f"Worker {rank}/{world_size}: processing {len(shard_files)} "
                f"pre-chunked file(s) from {shard_path.name}"
            )
            _process_files(
                rank,
                shard_files,
                config,
                podcasts_path,
                writer,
                already_done,
                processed_counter,
                skipped_counter,
                errors_counter,
                single_speaker_dropped,
                crest_files_in,
                crest_files_out,
                crest_duration_in,
                crest_duration_out,
            )
            mark_work_shard_done(shard_path)

    logger.info(f"Worker {rank} finished after {claimed} claimed shard(s).")


def main(args, *, config: Mapping[str, object] | None = None, logging_configured: bool = False) -> None:
    if not logging_configured:
        setup_logging("preprocess_existing_chunks", log_dir=args.log_dir)

    config = dict(config or load_config(args.config_path, "preprocess"))
    podcasts_path = Path(config.get("podcasts_path", "../../../podcasts"))
    overwrite = _truthy(config.get("existing_chunks_overwrite", False))
    source = str(config.get("existing_chunks_audio_paths_source", DEFAULT_AUDIO_PATHS_SOURCE))
    fuse_audio = fused_audio_preprocessing_enabled(config)
    single_speaker_only = single_speaker_only_enabled(config)
    logger.info(f"Fused crest/loudness preprocessing: {fuse_audio}")

    audio_paths = discover_audio_paths(
        podcasts_path,
        config_path=args.config_path,
        source=source,
    )
    if not audio_paths:
        logger.info("No audio files found for pre-chunked metadata extraction.")
        write_stage_status(
            stage=1,
            stage_name="preprocess_existing_chunks",
            log_dir=args.log_dir or "./logs",
            processed=0,
            skipped=0,
            errors=0,
        )
        return

    ensure_main_csv(podcasts_path, audio_paths=audio_paths)
    leftover_partials, absorbed = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=VALUE_COLUMNS,
        bootstrap_audio_paths=audio_paths,
        drop_missing_files=(fuse_audio or single_speaker_only),
        preserve_existing=not overwrite,
    )
    if absorbed:
        logger.info(
            f"Absorbed {absorbed} leftover {PARTIAL_PREFIX}_part_*.csv row(s) "
            "before scheduling existing chunks."
        )

    pending = pending_metadata_paths(
        podcasts_path,
        audio_paths,
        overwrite=overwrite,
        required_columns=(VALUE_COLUMNS if fuse_audio else METADATA_VALUE_COLUMNS),
        single_speaker_only=single_speaker_only,
    )
    skipped_initial = len(audio_paths) - len(pending)
    if not pending:
        logger.info("All existing chunks already have preprocess metadata.")
        write_stage_status(
            stage=1,
            stage_name="preprocess_existing_chunks",
            log_dir=args.log_dir or "./logs",
            processed=0,
            skipped=skipped_initial,
            errors=0,
        )
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        logger.error("No GPU detected; existing-chunk preprocess requires Sortformer.")
        write_stage_status(
            stage=1,
            stage_name="preprocess_existing_chunks",
            log_dir=args.log_dir or "./logs",
            processed=0,
            skipped=skipped_initial,
            errors=1,
            error_details=[{"reason": "No GPU detected"}],
        )
        return

    shard_size = load_work_shard_size(args.config_path)
    work_plan = prepare_work_shards(
        podcasts_path,
        "preprocess_existing_chunks",
        pending,
        shard_size=shard_size,
    )
    del pending

    processed = mp.Value("i", 0)
    skipped = mp.Value("i", skipped_initial)
    errors = mp.Value("i", 0)
    single_speaker_dropped = mp.Value("i", 0)
    crest_files_in = mp.Value("i", 0)
    crest_files_out = mp.Value("i", 0)
    crest_duration_in = mp.Value("d", 0.0)
    crest_duration_out = mp.Value("d", 0.0)
    csv_settings = load_csv_settings(args.config_path)

    try:
        with PeriodicCsvMerger(
            podcasts_path,
            prefix=PARTIAL_PREFIX,
            value_columns=VALUE_COLUMNS,
            bootstrap_audio_paths=audio_paths,
            drop_missing_files=(fuse_audio or single_speaker_only),
            preserve_existing=not overwrite,
            **csv_settings,
        ):
            mp.spawn(
                run_worker,
                args=(
                    num_gpus,
                    str(work_plan.work_dir),
                    config,
                    args.config_path,
                    str(podcasts_path),
                    processed,
                    skipped,
                    errors,
                    single_speaker_dropped,
                    crest_files_in,
                    crest_files_out,
                    crest_duration_in,
                    crest_duration_out,
                ),
                nprocs=num_gpus,
                join=True,
            )
    except KeyboardInterrupt:
        logger.warning("Existing-chunk preprocess interrupted; merging partials before exit.")

    new_partials, _ = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=VALUE_COLUMNS,
        bootstrap_audio_paths=audio_paths,
        drop_missing_files=(fuse_audio or single_speaker_only),
        preserve_existing=not overwrite,
    )

    combined = pd.concat(
        [df for df in (leftover_partials, new_partials) if df is not None and not df.empty],
        ignore_index=True,
    ) if (leftover_partials is not None or new_partials is not None) else pd.DataFrame()
    hours = (
        float(combined["total_duration"].fillna(0.0).sum() / 3600.0)
        if "total_duration" in combined.columns
        else 0.0
    )

    record_stage_summary(
        podcasts_path=podcasts_path,
        stage="preprocess",
        files_in=work_plan.total_items,
        files_out=(int(crest_files_out.value) if fuse_audio else max(0, int(processed.value) - int(single_speaker_dropped.value))),
        hours_in=(crest_duration_in.value / 3600.0 if fuse_audio else hours),
        hours_out=hours,
        params={
            "input_mode": "existing_chunks",
            "overwrite": overwrite,
            "audio_paths_source": source,
            "fuse_audio_preprocessing": fuse_audio,
            "single_speaker_only": single_speaker_only,
            "single_speaker_dropped": int(single_speaker_dropped.value),
        },
    )

    if fuse_audio and crest_files_in.value:
        record_stage_summary(
            podcasts_path=podcasts_path,
            stage="crest_factor",
            files_in=crest_files_in.value,
            files_out=crest_files_out.value,
            hours_in=crest_duration_in.value / 3600.0,
            hours_out=crest_duration_out.value / 3600.0,
            params={
                "threshold": config.get(
                    "crest_threshold", config.get("crest_treshold", 10.0)
                ),
                "fused": True,
            },
        )

    write_stage_status(
        stage=1,
        stage_name="preprocess_existing_chunks",
        log_dir=args.log_dir or "./logs",
        processed=processed.value,
        skipped=skipped.value,
        errors=errors.value,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(
        description="Backfill preprocess metadata for already chunked datasets."
    )
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    main(parser.parse_args())
