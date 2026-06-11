from __future__ import annotations

import multiprocessing as mp
import queue
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from crowdkit.aggregation import ROVER
from loguru import logger
from tqdm import tqdm

from src.utils.csv_manager import discover_audio_paths
from src.utils.sidecars import path_exists, text_sidecar_complete
from src.utils.utils import read_file_content
from src.utils.work_shards import (
    claim_work_shard,
    load_work_shard_size,
    mark_work_shard_done,
    prepare_work_shards,
    read_work_shard,
)


class ROVERWrapper:
    def __init__(
        self,
        podcasts_path: str,
        model_names: List[str],
        config_path: str | None = None,
        *,
        shard_size: Optional[int] = None,
        workers: int = 1,
        retry_empty_outputs: bool = False,
        use_fast_rover: bool = True,
    ):
        self.podcasts_path = Path(podcasts_path)
        self.model_names = model_names
        self.config_path = config_path
        self.shard_size = max(1, int(shard_size)) if shard_size else None
        self.workers = max(1, int(workers or 1))
        self.retry_empty_outputs = retry_empty_outputs
        self.use_fast_rover = use_fast_rover
        self.tokenizer = lambda s: s.lower().split()
        self.detokenizer = lambda tokens: ' '.join(tokens)
        self.excluded_patterns = ('_rover', '_phonemes', '_accent')

    def _make_aggregator(self):
        if self.use_fast_rover:
            try:
                from src.transcription.fast_rover import FastROVER

                return FastROVER(self.tokenizer, self.detokenizer)
            except Exception as exc:
                logger.warning(
                    f"FastROVER unavailable ({exc}); falling back to crowd-kit ROVER"
                )
        return ROVER(self.tokenizer, self.detokenizer)

    def _model_suffix(self, model_name: str) -> str:
        return 'vosk' if 'vosk' in model_name else model_name

    def _rover_output_path(self, audio_path: Path) -> Path:
        return audio_path.with_name(f"{audio_path.stem}_rover.txt")

    def _pending_audio_paths(self, audio_paths: Iterable[str]) -> List[str]:
        pending: List[str] = []
        total = len(audio_paths) if hasattr(audio_paths, "__len__") else None
        for raw_path in tqdm(audio_paths, total=total, desc="find_rover_pending"):
            audio_path = Path(raw_path)
            if any(pattern in audio_path.stem for pattern in self.excluded_patterns):
                continue
            if text_sidecar_complete(
                self._rover_output_path(audio_path),
                retry_empty=self.retry_empty_outputs,
                label="ROVER",
            ):
                continue
            pending.append(str(audio_path))
        return pending

    def _records_for_audio_paths(self, audio_paths: Iterable[str]) -> pd.DataFrame:
        records = []
        total = len(audio_paths) if hasattr(audio_paths, "__len__") else None
        for raw_path in tqdm(audio_paths, total=total, desc="load_rover_transcripts"):
            audio_path = Path(raw_path)
            if any(pattern in audio_path.stem for pattern in self.excluded_patterns):
                continue

            for model_name in self.model_names:
                suffix = self._model_suffix(model_name)
                transcript_path = audio_path.with_name(f"{audio_path.stem}_{suffix}.txt")

                if not path_exists(transcript_path, missing_on_too_long=True, label="Transcript"):
                    continue
                
                try:
                    text = read_file_content(transcript_path)
                    if not text:
                        continue

                    records.append({
                        'task': str(audio_path),
                        'worker': model_name,
                        'text': text.lower()
                    })
                except Exception as e:
                    logger.error(f"Error reading file {transcript_path}: {e}")

        return pd.DataFrame.from_records(records, columns=['task', 'worker', 'text'])

    def _save_results(self, result) -> int:
        saved = 0
        for task_path, agg_text in tqdm(result.items(), desc="save_rover_results"):
            audio_path = Path(task_path)
            output_path = self._rover_output_path(audio_path)
            tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

            try:
                with tmp_path.open("w", encoding="utf-8") as f:
                    f.write("" if agg_text is None else str(agg_text))
                tmp_path.replace(output_path)
                saved += 1
            except IOError as e:
                logger.error(f"Failed to write ROVER result to {output_path}: {e}")

        return saved

    def _aggregate_audio_paths(self, audio_paths: List[str], label: str) -> tuple[int, int, int]:
        if not audio_paths:
            return 0, 0, 0

        df = self._records_for_audio_paths(audio_paths)
        if df.empty:
            logger.warning(f"No transcriptions found in ROVER shard {label}.")
            return len(audio_paths), 0, 0

        task_count = int(df['task'].nunique())
        logger.info(
            f"Running ROVER on shard {label}: "
            f"{task_count} audio file(s), {len(df)} transcript(s)."
        )
        result = self._make_aggregator().fit_predict(df)
        saved = self._save_results(result)
        return len(audio_paths), task_count, saved

    def _aggregate_with_fallback(self, audio_paths: List[str], label: str) -> tuple[int, int, int]:
        try:
            return self._aggregate_audio_paths(audio_paths, label)
        except Exception:
            if len(audio_paths) <= 1:
                raise
            mid = len(audio_paths) // 2
            logger.warning(
                f"ROVER shard {label} failed with {len(audio_paths)} audio file(s); "
                "retrying as two smaller chunks."
            )
            left = self._aggregate_with_fallback(audio_paths[:mid], f"{label}.a")
            right = self._aggregate_with_fallback(audio_paths[mid:], f"{label}.b")
            return (
                left[0] + right[0],
                left[1] + right[1],
                left[2] + right[2],
            )

    def _aggregate_shard(self, shard_path: Path) -> tuple[int, int, int]:
        audio_paths = read_work_shard(shard_path)
        return self._aggregate_with_fallback(audio_paths, shard_path.name)

    def _aggregate_shards_sequential(self, work_dir: Path) -> tuple[int, int, int, int]:
        total_seen = 0
        total_tasks = 0
        total_saved = 0
        failed_shards = 0
        while True:
            shard_path = claim_work_shard(work_dir, 0)
            if shard_path is None:
                break
            try:
                seen, tasks, saved = self._aggregate_shard(shard_path)
                total_seen += seen
                total_tasks += tasks
                total_saved += saved
                mark_work_shard_done(shard_path)
            except Exception as exc:
                failed_shards += 1
                logger.exception(f"ROVER shard failed {shard_path.name}: {exc}")
        return total_seen, total_tasks, total_saved, failed_shards

    def _aggregate_shards_parallel(self, work_dir: Path, worker_count: int) -> tuple[int, int, int, int]:
        stats_queue: mp.Queue = mp.Queue()
        processes: List[mp.Process] = []
        try:
            for worker_id in range(worker_count):
                proc = mp.Process(
                    target=_rover_worker_main,
                    args=(
                        worker_id,
                        str(work_dir),
                        str(self.podcasts_path),
                        list(self.model_names),
                        self.retry_empty_outputs,
                        self.use_fast_rover,
                        stats_queue,
                    ),
                    name=f"rover-worker-{worker_id}",
                )
                proc.start()
                processes.append(proc)
                logger.info(f"Launched {proc.name} with pid={proc.pid}")

            for proc in processes:
                proc.join()
        except KeyboardInterrupt:
            logger.warning("Interrupted by user; terminating ROVER workers...")
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
            for proc in processes:
                proc.join()
            raise

        total_seen = 0
        total_tasks = 0
        total_saved = 0
        failed_shards = 0
        received_stats = 0
        for _ in processes:
            try:
                stats = stats_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            received_stats += 1
            total_seen += int(stats.get("seen", 0))
            total_tasks += int(stats.get("tasks", 0))
            total_saved += int(stats.get("saved", 0))
            failed_shards += int(stats.get("failed_shards", 0))

        for proc in processes:
            if proc.exitcode not in (0, None):
                logger.error(f"{proc.name} exited with code {proc.exitcode}")
                failed_shards += 1

        if received_stats < len(processes):
            missing = len(processes) - received_stats
            logger.warning(f"ROVER did not receive stats from {missing} worker process(es).")

        stats_queue.close()
        stats_queue.join_thread()
        return total_seen, total_tasks, total_saved, failed_shards

    def aggregate_and_save(self):
        logger.info("Starting sharded transcription aggregation based on audio files.")

        all_audio_paths = discover_audio_paths(self.podcasts_path, config_path=self.config_path)
        if not all_audio_paths:
            logger.warning("Audio files not found. Aggregation finished.")
            return

        pending_paths = self._pending_audio_paths(all_audio_paths)
        del all_audio_paths

        if not pending_paths:
            logger.success("No pending ROVER files; aggregation already up to date.")
            return

        shard_size = self.shard_size or load_work_shard_size(self.config_path)
        work_plan = prepare_work_shards(
            self.podcasts_path,
            "transcription_rover",
            pending_paths,
            shard_size=shard_size,
        )
        del pending_paths

        worker_count = min(self.workers, max(1, work_plan.shard_count))
        logger.info(
            f"ROVER aggregation will use {worker_count} worker process(es) "
            f"for {work_plan.shard_count} shard(s)."
        )
        if worker_count <= 1:
            total_seen, total_tasks, total_saved, failed_shards = self._aggregate_shards_sequential(
                work_plan.work_dir
            )
        else:
            total_seen, total_tasks, total_saved, failed_shards = self._aggregate_shards_parallel(
                work_plan.work_dir,
                worker_count,
            )

        logger.info(
            f"ROVER aggregation complete: {total_saved} result(s) saved, "
            f"{total_tasks} task(s) with transcripts, {total_seen} pending audio file(s) seen, "
            f"{failed_shards} failed shard(s)."
        )


def _rover_worker_main(
    worker_id: int,
    work_dir: str,
    podcasts_path: str,
    model_names: List[str],
    retry_empty_outputs: bool,
    use_fast_rover: bool,
    stats_queue,
) -> None:
    wrapper = ROVERWrapper(
        podcasts_path=podcasts_path,
        model_names=model_names,
        retry_empty_outputs=retry_empty_outputs,
        use_fast_rover=use_fast_rover,
    )
    stats = {
        "seen": 0,
        "tasks": 0,
        "saved": 0,
        "failed_shards": 0,
        "claimed_shards": 0,
    }

    try:
        while True:
            shard_path = claim_work_shard(work_dir, worker_id)
            if shard_path is None:
                break
            stats["claimed_shards"] += 1
            try:
                seen, tasks, saved = wrapper._aggregate_shard(shard_path)
                stats["seen"] += seen
                stats["tasks"] += tasks
                stats["saved"] += saved
                mark_work_shard_done(shard_path)
            except Exception as exc:
                stats["failed_shards"] += 1
                logger.exception(f"ROVER worker {worker_id} failed shard {shard_path.name}: {exc}")
    finally:
        stats_queue.put(stats)
        logger.info(
            f"ROVER worker {worker_id} finished: {stats['claimed_shards']} shard(s), "
            f"{stats['saved']} result(s), {stats['failed_shards']} failed shard(s)."
        )
