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
) -> int:
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
        Number of items that completed successfully.
    """
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    gpu_ids = list(gpu_ids)
    if not gpu_ids:
        raise RuntimeError("No GPUs available; refusing to run a per-GPU pool.")
    if not items:
        return 0

    shards = shard_round_robin(items, len(gpu_ids))
    executors: List[ProcessPoolExecutor] = []
    futures: List = []
    completed = 0

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
                futures.append(ex.submit(work_fn, item))

        with tqdm(total=len(futures), desc=desc) as bar:
            for fut in as_completed(futures):
                try:
                    fut.result()
                    completed += 1
                except Exception as exc:
                    logger.error(f"{desc}: task failed: {exc}")
                bar.update(1)
    except KeyboardInterrupt:
        logger.warning(f"{desc}: interrupted by user; shutting down workers...")
    finally:
        for ex in executors:
            ex.shutdown(wait=True, cancel_futures=True)

    return completed


def run_per_gpu_processes(
    work_fn: Callable[..., None],
    *,
    num_gpus: int,
    args: Tuple[Any, ...] = (),
) -> None:
    """Spawn exactly one :class:`multiprocessing.Process` per GPU.

    Each process receives ``(gpu_id, num_gpus, *args)``. With ``num_gpus<=1``
    the function calls ``work_fn`` directly in the parent for simpler
    debugging.

    On ``KeyboardInterrupt`` all live children are terminated and joined.
    Non-zero exit codes are logged.
    """
    if num_gpus <= 1:
        work_fn(0, max(num_gpus, 1), *args)
        return

    procs: List[mp.Process] = []
    try:
        for gid in range(num_gpus):
            p = mp.Process(target=work_fn, args=(gid, num_gpus, *args))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        logger.warning("Interrupted; terminating GPU processes...")
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()

    failed = [p.exitcode for p in procs if p.exitcode]
    if failed:
        logger.error(f"GPU processes failed with exit codes: {failed}")
