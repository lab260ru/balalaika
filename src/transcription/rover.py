from __future__ import annotations

import multiprocessing as mp
import queue
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from crowdkit.aggregation import ROVER
from loguru import logger
from tqdm import tqdm

from src.utils.csv_manager import discover_audio_paths
from src.utils.chunk_json import ChunkJsonCache, get_field, update_chunk_json
from src.utils.utils import model_key
from src.utils.work_shards import (
    claim_work_shard,
    load_work_shard_size,
    mark_work_shard_done,
    prepare_work_shards,
    read_work_shard,
)


def _word_tokens(text: object) -> List[str]:
    if text is None:
        return []
    chars = []
    for char in str(text).lower().replace("е", "ё"):
        if unicodedata.category(char).startswith("P"):
            chars.append(" ")
        else:
            chars.append(char)
    return "".join(chars).split()


def _word_edit_distance(left: List[str], right: List[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, token_left in enumerate(left, start=1):
        current = [i]
        for j, token_right in enumerate(right, start=1):
            substitution = previous[j - 1] + (token_left != token_right)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def asr_consistency_from_transcripts(
    transcripts: Iterable[object],
    consensus: object,
) -> Optional[float]:
    consensus_words = _word_tokens(consensus)
    if not consensus_words:
        return None

    model_words = [_word_tokens(text) for text in transcripts]
    if len(model_words) < 2:
        return None

    denom = float(len(consensus_words))
    scores = [
        max(0.0, 1.0 - (_word_edit_distance(words, consensus_words) / denom))
        for words in model_words
    ]
    return (sum(scores) / len(scores)) * 100.0



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
        # Per-instance count of FastROVER -> crowd-kit fallbacks. Surfaced in
        # the worker / orchestrator end-of-run summary so a run that silently
        # mixed fast and stock aggregations is greppable.
        self.fast_path_fallbacks = 0

    def _make_aggregator(self):
        if self.use_fast_rover:
            try:
                from src.transcription.fast_rover import FastROVER

                return FastROVER(self.tokenizer, self.detokenizer)
            except Exception as exc:
                self.fast_path_fallbacks += 1
                logger.warning(
                    f"FastROVER unavailable ({exc}); falling back to crowd-kit ROVER"
                )
        return ROVER(self.tokenizer, self.detokenizer)

    def log_fallback_summary(self, label: str = "") -> None:
        prefix = f"{label} " if label else ""
        logger.info(f"ROVER {prefix}fast-path fallbacks: {self.fast_path_fallbacks}")

    def _model_suffix(self, model_name: str) -> str:
        return model_key(model_name)

    def _pending_audio_paths(self, audio_paths: Iterable[str]) -> List[str]:
        pending: List[str] = []
        total = len(audio_paths) if hasattr(audio_paths, "__len__") else None
        # One scandir per directory + one chunk-JSON read per file replaces the
        # per-audio _rover.txt stat. Built per-process: this runs in the
        # orchestrator before shards are dispatched, never across the spawn
        # boundary.
        cache = ChunkJsonCache()
        for raw_path in tqdm(audio_paths, total=total, desc="find_rover_pending"):
            audio_path = Path(raw_path)
            if any(pattern in audio_path.stem for pattern in self.excluded_patterns):
                continue
            rover_complete = cache.field_complete(
                audio_path, "rover", retry_empty=self.retry_empty_outputs
            )
            consistency_complete = cache.field_complete(
                audio_path, "asr_consistency", retry_empty=False
            )
            if rover_complete and consistency_complete:
                continue
            pending.append(str(audio_path))
        return pending

    def _records_for_audio_paths(
        self,
        audio_paths: Iterable[str],
    ) -> tuple[pd.DataFrame, dict[str, List[str]]]:
        records = []
        transcripts_by_task: dict[str, List[str]] = {}
        total = len(audio_paths) if hasattr(audio_paths, "__len__") else None
        # Per-shard cache: built fresh in whatever (possibly spawned) worker
        # process owns this shard. A shard's files cluster in a handful of
        # episode directories, so one scandir per directory + one JSON read per
        # file amortizes across all model probes for every file in that dir.
        cache = ChunkJsonCache()
        for raw_path in tqdm(audio_paths, total=total, desc="load_rover_transcripts"):
            audio_path = Path(raw_path)
            if any(pattern in audio_path.stem for pattern in self.excluded_patterns):
                continue

            data = cache.get(audio_path)
            task = str(audio_path)
            model_texts: List[str] = []
            for model_name in self.model_names:
                suffix = self._model_suffix(model_name)
                text = get_field(data, f"asr.{suffix}")
                if text is None:
                    continue
                text_str = str(text).lower()
                model_texts.append(text_str)
                if not text_str:
                    continue
                records.append({
                    "task": task,
                    "worker": model_name,
                    "text": text_str,
                })
            if model_texts:
                transcripts_by_task[task] = model_texts

        df = pd.DataFrame.from_records(records, columns=["task", "worker", "text"])
        return df, transcripts_by_task

    def _save_results(
        self,
        result,
        transcripts_by_task: dict[str, List[str]],
    ) -> int:
        saved = 0
        for task_path, agg_text in tqdm(result.items(), desc="save_rover_results"):
            try:
                consensus = "" if agg_text is None else str(agg_text)
                consistency = asr_consistency_from_transcripts(
                    transcripts_by_task.get(str(task_path), []),
                    consensus,
                )
                update_chunk_json(
                    task_path,
                    {
                        "rover": consensus,
                        "asr_consistency": "" if consistency is None else consistency,
                    },
                )
                saved += 1
            except OSError as e:
                logger.error(f"Failed to write ROVER result to {task_path}: {e}")

        return saved

    def _aggregate_audio_paths(self, audio_paths: List[str], label: str) -> tuple[int, int, int]:
        if not audio_paths:
            return 0, 0, 0

        df, transcripts_by_task = self._records_for_audio_paths(audio_paths)
        if df.empty:
            logger.warning(f"No transcriptions found in ROVER shard {label}.")
            return len(audio_paths), 0, 0

        task_count = int(df['task'].nunique())
        logger.info(
            f"Running ROVER on shard {label}: "
            f"{task_count} audio file(s), {len(df)} transcript(s)."
        )
        result = self._make_aggregator().fit_predict(df)
        saved = self._save_results(result, transcripts_by_task)
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
        # Sequential path aggregates on this instance; the parallel path's
        # per-worker counts are logged in each worker's own summary.
        self.log_fallback_summary()


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
        wrapper.log_fallback_summary(label=f"worker {worker_id}")
