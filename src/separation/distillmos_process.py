"""DistillMOS scoring of audio chunks with crash-safe CSV state.

Each GPU worker runs DistillMOS over its shard, batching with a sample-length
sorted dataloader for throughput. Results are streamed to a worker-local
``distillmos_part_<rank>.csv`` via :class:`PartialCsvWriter` (``flush()``
after every row), so a forced stop preserves whatever rows were already
produced.

Resume behaviour:

* ``balalaika.csv`` is bootstrapped from the audio tree if it does not yet
  exist (so this stage can be the *first* to write a CSV when running out of
  order).
* Any leftover ``distillmos_part_*.csv`` from a previously interrupted run is
  absorbed into the main CSV before scheduling new work.
* Files already scored (non-null ``DistillMOS`` in ``balalaika.csv``) are
  skipped automatically.
"""
import argparse
import time
from pathlib import Path
from typing import List, Set

import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm

from src.utils.audio_durations import (
    duration_bucket_settings,
    duration_probe_workers,
    ensure_audio_durations,
)
from src.separation.inline_filter import (
    DISTILLMOS,
    INLINE_PARTIAL_FIELDS,
    resolve_inline,
    write_score_row,
)
from src.utils.audit import record_stage_summary
from src.utils.csv_manager import (
    PartialCsvWriter,
    PeriodicCsvMerger,
    absorb_partial_csvs,
    audit_from_filter_partials,
    discover_audio_paths,
    ensure_main_csv,
    load_csv_settings,
    resolve_path,
    unprocessed_paths,
)
from src.utils.datasets.separation import create_distillmos_dataloader
from src.utils.node_profile import resolve_batch_size
from src.utils.gpu import apply_torch_perf_defaults
from src.utils.logging_setup import setup_logging
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config
from src.utils.work_shards import (
    claim_work_shard,
    load_work_shard_size,
    mark_work_shard_done,
    prepare_length_bucketed_work_shards,
    read_annotated_work_shard,
)

apply_torch_perf_defaults(disable_math_sdp=False)


PARTIAL_PREFIX = "distillmos"
PARTIAL_FIELDS = ("filepath", "DistillMOS")
COLUMN = "DistillMOS"
VALUE_COLUMNS = [COLUMN]


def _inline_threshold(config: dict):
    """Inline-delete threshold for stage 5, or ``None`` when score-only."""
    return resolve_inline(
        config.get("distillmos", {}), config.get("distillmos_filter", {})
    )


def _process_files(
    rank: int,
    files: List[str],
    config: dict,
    podcasts_path: Path,
    sqa_model,
    device: torch.device,
    writer: PartialCsvWriter,
    already_done: Set[str],
    processed_counter,
    skipped_counter,
    errors_counter,
    inline_threshold=None,
    audio_lengths=None,
) -> None:
    batch_size = resolve_batch_size(
        "distillmos", config.get("distillmos", {}).get("batch_size"), 16
    )
    num_loader_workers = int(config.get("distillmos", {}).get("num_workers", 2))
    prefetch_factor = int(config.get("distillmos", {}).get("prefetch_factor", 2))

    pending_files = []
    for path in files:
        resolved = resolve_path(path)
        if resolved in already_done:
            skipped_counter.value += 1
            continue
        pending_files.append(path)

    if not pending_files:
        return

    # Shards arrive duration-sorted from prepare_length_bucketed_work_shards;
    # sort_in_loader: true restores the old per-shard re-probe + JSON cache.
    sort_in_loader = bool(config.get("distillmos", {}).get("sort_in_loader", False))
    dataloader = create_distillmos_dataloader(
        pending_files,
        batch_size=batch_size,
        num_workers=num_loader_workers,
        prefetch_factor=prefetch_factor,
        cache_dir=str(podcasts_path),
        assume_sorted=not sort_in_loader,
    )
    prefetch_batches = num_loader_workers * prefetch_factor if num_loader_workers > 0 else 0
    logger.debug(
        f"perf dataloader_config stage=distillmos rank={rank} "
        f"batch_size={batch_size} workers={num_loader_workers} "
        f"prefetch_factor={prefetch_factor} prefetch_batches={prefetch_batches} "
        f"items={len(pending_files)}"
    )

    with torch.inference_mode():
        batch_wait_started_at = time.perf_counter()
        for batch_idx, (paths, batch) in enumerate(tqdm(dataloader, desc=f"DistillMOS-{rank}", position=rank)):
            batch_received_at = time.perf_counter()
            logger.debug(
                f"perf dataloader_wait stage=distillmos rank={rank} "
                f"batch={batch_idx} seconds={batch_received_at - batch_wait_started_at:.6f} "
                f"items={len(paths)}"
            )
            try:
                batch = batch.to(device, non_blocking=True)
                inference_started_at = time.perf_counter()
                mos = sqa_model(batch).detach().flatten().cpu()
                logger.debug(
                    f"perf model=distillmos event=inference rank={rank} "
                    f"batch={batch_idx} seconds={time.perf_counter() - inference_started_at:.6f} "
                    f"items={len(paths)} frames={int(batch.shape[-1])}"
                )
                for path_str, mos_val in zip(paths, mos.tolist()):
                    resolved = resolve_path(path_str)
                    if resolved in already_done:
                        skipped_counter.value += 1
                        continue
                    write_started_at = time.perf_counter()
                    write_score_row(
                        writer,
                        stage=DISTILLMOS,
                        resolved_path=resolved,
                        audio_path=path_str,
                        scores={COLUMN: float(mos_val)},
                        inline_threshold=inline_threshold,
                        audio_lengths=audio_lengths,
                        errors_counter=errors_counter,
                    )
                    logger.debug(
                        f"perf partial_write stage=distillmos rank={rank} "
                        f"seconds={time.perf_counter() - write_started_at:.6f} path={resolved}"
                    )
                    already_done.add(resolved)
                    processed_counter.value += 1
            except torch.cuda.OutOfMemoryError:
                logger.critical(f"CUDA OOM on worker {rank}, stopping")
                errors_counter.value += 1
                raise
            except Exception as exc:
                logger.warning(f"Error processing batch on worker {rank}: {exc}")
                errors_counter.value += 1
                batch_wait_started_at = time.perf_counter()
                continue
            batch_wait_started_at = time.perf_counter()


def run_inference_worker(
    rank: int,
    world_size: int,
    work_dir: str,
    config: dict,
    podcasts_path: Path,
    processed_counter,
    skipped_counter,
    errors_counter,
):
    device = torch.device(f"cuda:{rank}")

    logger.info(f"[cuda:{rank}] Loading DistillMOS model...")
    try:
        import distillmos
        sqa_model = distillmos.ConvTransformerSQAModel()
        sqa_model.to(device)
        sqa_model.eval()
    except Exception as exc:
        logger.error(f"Failed to load distillmos model on worker {rank}: {exc}")
        errors_counter.value += 1
        return

    batch_size = resolve_batch_size(
        "distillmos", config.get("distillmos", {}).get("batch_size"), 16
    )
    num_loader_workers = int(config.get("distillmos", {}).get("num_workers", 2))
    logger.info(
        f"[cuda:{rank}] Claiming DistillMOS shards "
        f"(batch={batch_size}, loader_workers={num_loader_workers})."
    )
    started_at = time.perf_counter()

    inline_threshold = _inline_threshold(config)
    fieldnames = (
        PARTIAL_FIELDS + INLINE_PARTIAL_FIELDS
        if inline_threshold is not None
        else PARTIAL_FIELDS
    )

    claimed = 0
    with PartialCsvWriter(
        podcasts_path, PARTIAL_PREFIX, rank, fieldnames=fieldnames
    ) as writer:
        already_done: Set[str] = writer.already_done()
        if already_done:
            logger.info(
                f"Worker {rank}: {len(already_done)} files already scored in this partial; skipping repeats."
            )

        while True:
            shard_path = claim_work_shard(work_dir, rank)
            if shard_path is None:
                break
            # Annotated shards carry the precomputed duration (empty note when
            # not inline); kept inline rows record hours without a probe.
            items = read_annotated_work_shard(shard_path)
            shard_files = [p for p, _ in items]
            audio_lengths = None
            if inline_threshold is not None and items and all(n for _, n in items):
                try:
                    audio_lengths = {p: float(n) for p, n in items}
                except ValueError:
                    audio_lengths = None
            claimed += 1
            logger.info(f"[cuda:{rank}] Processing {len(shard_files)} files from {shard_path.name}.")
            _process_files(
                rank,
                shard_files,
                config,
                podcasts_path,
                sqa_model,
                device,
                writer,
                already_done,
                processed_counter,
                skipped_counter,
                errors_counter,
                inline_threshold=inline_threshold,
                audio_lengths=audio_lengths,
            )
            mark_work_shard_done(shard_path)

    elapsed = time.perf_counter() - started_at
    logger.success(
        f"[cuda:{rank}] Finished {claimed} shard(s) in {elapsed:.2f}s."
    )

def main():
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N unprocessed files",
    )
    args = parser.parse_args()

    setup_logging("distillmos", log_dir=args.log_dir)

    config = load_config(args.config_path, "separation")
    podcasts_path = Path(config.get("podcasts_path", "."))

    # Score-only unless inline_filter is on; when on, delete low-MOS files in
    # this pass, prune their rows, and emit a filter row (stage 5.5 then no-ops).
    inline_threshold = _inline_threshold(config)
    drop_missing = inline_threshold is not None
    value_columns = (
        VALUE_COLUMNS + ["total_duration"] if drop_missing else VALUE_COLUMNS
    )
    if inline_threshold is not None:
        logger.info(
            f"inline_filter active: deleting files with DistillMOS < {inline_threshold} "
            "during scoring."
        )

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        logger.error("No GPU detected.")
        return

    audio_paths = discover_audio_paths(podcasts_path, config_path=args.config_path)
    if not audio_paths:
        logger.warning("No audio files found.")
        return

    # 1) Make sure balalaika.csv exists; bootstrap from the audio tree if not.
    ensure_main_csv(podcasts_path, audio_paths=audio_paths)

    # 2) Absorb leftover partials from a previous interrupted run.
    leftover_partials, absorbed = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=value_columns,
        drop_missing_files=drop_missing,
        preserve_existing=True,
    )
    if absorbed:
        logger.info(
            f"Absorbed {absorbed} rows from leftover {PARTIAL_PREFIX}_part_*.csv "
            "into balalaika.csv before scheduling new work."
        )

    # 3) Determine work: any file without a DistillMOS score yet.
    unprocessed = unprocessed_paths(podcasts_path, COLUMN, audio_paths)
    if args.limit is not None:
        unprocessed = unprocessed[: args.limit]

    if not unprocessed:
        logger.success("All audio files already have a DistillMOS score. Exiting.")
        return

    shard_size = load_work_shard_size(args.config_path)
    distillmos_cfg = config.get("distillmos", {})
    duration_workers = duration_probe_workers(distillmos_cfg, config)
    durations = ensure_audio_durations(
        podcasts_path,
        unprocessed,
        num_workers=duration_workers,
    )
    bucket_seconds, max_bucket_duration = duration_bucket_settings(
        args.config_path,
        distillmos_cfg,
        config,
    )
    # In inline mode carry each file's (already computed) duration into the
    # shard so kept rows record their hours without a re-probe.
    annotations = (
        {p: str(float(durations.get(p, 0.0) or 0.0)) for p in unprocessed}
        if drop_missing
        else None
    )
    work_plan = prepare_length_bucketed_work_shards(
        podcasts_path,
        PARTIAL_PREFIX,
        unprocessed,
        durations,
        shard_size=shard_size,
        bucket_seconds=bucket_seconds,
        max_duration=max_bucket_duration,
        annotations=annotations,
    )
    del unprocessed
    del durations
    del annotations

    logger.info(
        f"Processing {work_plan.total_items} files on {available_gpus} GPUs "
        f"over {work_plan.shard_count} shard(s)."
    )

    processed = mp.Value('i', 0)
    skipped = mp.Value('i', 0)
    errors = mp.Value('i', 0)

    csv_settings = load_csv_settings(args.config_path)

    try:
        with PeriodicCsvMerger(
            podcasts_path,
            prefix=PARTIAL_PREFIX,
            value_columns=value_columns,
            drop_missing_files=drop_missing,
            preserve_existing=True,
            **csv_settings,
        ):
            mp.spawn(
                run_inference_worker,
                args=(available_gpus, str(work_plan.work_dir), config, podcasts_path, processed, skipped, errors),
                nprocs=available_gpus,
                join=True,
            )
    except KeyboardInterrupt:
        logger.warning("DistillMOS stage interrupted; merging partials before exit.")
    except Exception as exc:
        logger.critical(f"Multiprocessing failed: {exc}")

    # 4) Merge whatever the workers managed to produce (always; even on Ctrl+C).
    new_partials, _ = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=value_columns,
        drop_missing_files=drop_missing,
        preserve_existing=True,
    )

    # Inline mode emits the filter row (stages 5.5 otherwise owns it).
    if inline_threshold is not None:
        partial_frames = [
            df
            for df in (leftover_partials, new_partials)
            if df is not None and not df.empty
        ]
        combined = (
            pd.concat(partial_frames, ignore_index=True)
            if partial_frames
            else pd.DataFrame()
        )
        audit = audit_from_filter_partials(combined)
        record_stage_summary(
            podcasts_path=podcasts_path,
            stage="distillmos_filter",
            files_in=audit["files_in"],
            files_out=audit["files_out"],
            hours_in=audit["hours_in"],
            hours_out=audit["hours_out"],
            params={"threshold": inline_threshold, "deleted": audit["files_deleted"]},
        )

    write_stage_status(
        stage=5,
        stage_name="distillmos_process",
        log_dir=args.log_dir or "./logs",
        processed=processed.value,
        skipped=skipped.value,
        errors=errors.value,
    )


if __name__ == "__main__":
    main()
