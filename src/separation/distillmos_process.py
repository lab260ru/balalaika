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

from src.utils.csv_manager import (
    PartialCsvWriter,
    PeriodicCsvMerger,
    absorb_partial_csvs,
    discover_audio_paths,
    ensure_main_csv,
    load_csv_settings,
    resolve_path,
    unprocessed_paths,
)
from src.utils.datasets.separation import create_distillmos_dataloader
from src.utils.gpu import apply_torch_perf_defaults
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

apply_torch_perf_defaults(disable_math_sdp=False)


PARTIAL_PREFIX = "distillmos"
PARTIAL_FIELDS = ("filepath", "DistillMOS")
COLUMN = "DistillMOS"


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
) -> None:
    batch_size = int(config.get("distillmos", {}).get("batch_size", 16))
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

    dataloader = create_distillmos_dataloader(
        pending_files,
        batch_size=batch_size,
        num_workers=num_loader_workers,
        prefetch_factor=prefetch_factor,
        cache_dir=str(podcasts_path),
    )

    with torch.inference_mode():
        for paths, batch in tqdm(dataloader, desc=f"DistillMOS-{rank}", position=rank):
            try:
                batch = batch.to(device, non_blocking=True)
                mos = sqa_model(batch).detach().flatten().cpu()
                for path_str, mos_val in zip(paths, mos.tolist()):
                    resolved = resolve_path(path_str)
                    if resolved in already_done:
                        skipped_counter.value += 1
                        continue
                    writer.write(
                        {
                            "filepath": resolved,
                            COLUMN: float(mos_val),
                        }
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
                continue


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

    batch_size = int(config.get("distillmos", {}).get("batch_size", 16))
    num_loader_workers = int(config.get("distillmos", {}).get("num_workers", 2))
    logger.info(
        f"[cuda:{rank}] Claiming DistillMOS shards "
        f"(batch={batch_size}, loader_workers={num_loader_workers})."
    )
    started_at = time.perf_counter()

    claimed = 0
    with PartialCsvWriter(
        podcasts_path, PARTIAL_PREFIX, rank, fieldnames=PARTIAL_FIELDS
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
            shard_files = read_work_shard(shard_path)
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
    _, absorbed = absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=[COLUMN],
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
    work_plan = prepare_work_shards(
        podcasts_path,
        PARTIAL_PREFIX,
        unprocessed,
        shard_size=shard_size,
    )
    del unprocessed

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
            value_columns=[COLUMN],
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
    absorb_partial_csvs(
        podcasts_path,
        PARTIAL_PREFIX,
        value_columns=[COLUMN],
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
