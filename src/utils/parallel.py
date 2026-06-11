"""Common multi-GPU parallel-execution patterns for pipeline stages.

Two reusable orchestrators replace the bespoke per-stage scaffolding:

* :func:`run_per_gpu_pool` — round-robin shard ``items`` over GPUs and run
  ``work_fn`` against a :class:`~concurrent.futures.ProcessPoolExecutor` per
  GPU (with ``initializer`` + per-GPU init args). Used by stages that load a
  small model in each worker (punctuation, accents, phonemizer).
* :func:`run_per_gpu_processes` — spawn exactly one process per GPU using
  :class:`multiprocessing.Process` and pass ``(gpu_id, num_gpus, *args)``.
  Used by stages that load one big model per GPU (transcription).

Both helpers handle ``KeyboardInterrupt``: pools shut down cleanly,
processes are terminated, and the wrapper returns to the caller so the
stage can finalize state (merge partials, etc.).
"""
from __future__ import annotations

import multiprocessing as mp
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, List, Optional, Sequence, Tuple

from loguru import logger
from tqdm import tqdm



def shard_round_robin(items: Sequence[Any], num_shards: int) -> List[List[Any]]:
    """Split ``items`` into ``num_shards`` near-equal lists by round-robin."""
    n = max(1, num_shards)
    shards: List[List[Any]] = [[] for _ in range(n)]
    for i, x in enumerate(items):
        shards[i % n].append(x)
    return shards


def run_per_gpu_pool(
    items: Sequence[Any],
    *,
    work_fn: Callable[..., Any],
    initializer: Callable[..., None],
    init_args_factory: Callable[[int], Tuple[Any, ...]],
    num_workers_per_gpu: int = 1,
    gpu_ids: Optional[Sequence[int]] = None,
    desc: str = "Progress",
) -> tuple[int, list[dict]]:
    """Round-robin distribute ``items`` across GPUs and run ``work_fn`` in pools.

    Args:
        items: Work items to submit (each ``items[i]`` is passed positionally
            as the single argument to ``work_fn``).
        work_fn: Callable invoked once per item inside a worker process.
        initializer: Called once at the start of each worker process.
        init_args_factory: ``init_args_factory(gpu_id)`` returns the tuple of
            ``initargs`` for the pool bound to that GPU.
        num_workers_per_gpu: Number of worker processes inside each per-GPU
            pool.
        gpu_ids: Explicit GPU index list; defaults to ``range(torch.cuda.device_count())``.
        desc: Tqdm description.

    Returns:
        A ``(error_count, error_details)`` tuple: the number of items whose
        ``work_fn`` raised, and a list of ``{"item", "reason"}`` dicts (one per
        failure).
    """
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    gpu_ids = list(gpu_ids)
    if not gpu_ids:
        raise RuntimeError("No GPUs available; refusing to run a per-GPU pool.")
    if not items:
        return 0, []

    shards = shard_round_robin(items, len(gpu_ids))
    executors: List[ProcessPoolExecutor] = []
    future_to_item: dict = {}
    error_count = 0
    error_details: list[dict] = []

    try:
        for slot, gpu_id in enumerate(gpu_ids):
            chunk = shards[slot]
            if not chunk:
                continue
            logger.info(
                f"{desc}: launching {num_workers_per_gpu} workers on GPU {gpu_id} "
                f"for {len(chunk)} items."
            )
            ex = ProcessPoolExecutor(
                max_workers=num_workers_per_gpu,
                initializer=initializer,
                initargs=init_args_factory(gpu_id),
            )
            executors.append(ex)
            for item in chunk:
                future_to_item[ex.submit(work_fn, item)] = item

        with tqdm(total=len(future_to_item), desc=desc) as bar:
            for fut in as_completed(future_to_item):
                try:
                    fut.result()
                except Exception as exc:
                    item = future_to_item[fut]
                    logger.error(f"{desc}: task failed for {item}: {exc}")
                    error_count += 1
                    error_details.append({"item": str(item), "reason": str(exc)})
                bar.update(1)
    except KeyboardInterrupt:
        logger.warning(f"{desc}: interrupted by user; shutting down workers...")
    finally:
        for ex in executors:
            ex.shutdown(wait=True, cancel_futures=True)

    return error_count, error_details


def run_per_gpu_processes(
    run_worker: Callable,
    num_gpus: int,
    args: tuple = (),
    join: bool = True,
) -> tuple[int, list[dict]]:
    """Spawn exactly one :class:`multiprocessing.Process` per GPU.

    Each process receives ``(gpu_id, num_gpus, *args)``. With ``num_gpus<=1``
    the function calls ``run_worker`` directly in the parent for simpler
    debugging.

    On ``KeyboardInterrupt`` all live children are terminated and joined.
    Non-zero exit codes are logged.
    """
    if num_gpus <= 1:
        run_worker(0, max(num_gpus, 1), *args)
        return 0, []

    processes: List[mp.Process] = []
    error_count = 0
    error_details: list[dict] = []

    try:
        for gpu_id in range(num_gpus):
            proc = mp.Process(
                target=run_worker,
                args=(gpu_id, num_gpus, *args),
                name=f"{run_worker.__name__}-gpu{gpu_id}",
            )
            proc.start()
            processes.append(proc)
            logger.info(f"Launched {proc.name} with pid={proc.pid}")

        if not join:
            return 0, []

        for proc in processes:
            proc.join()
            if proc.exitcode not in (0, None):
                logger.error(f"{proc.name} exited with code {proc.exitcode}")
                error_count += 1
                error_details.append(
                    {
                        "worker": proc.name,
                        "pid": proc.pid,
                        "exitcode": proc.exitcode,
                    }
                )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; terminating GPU workers...")
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        for proc in processes:
            proc.join()
        raise

    return error_count, error_details

