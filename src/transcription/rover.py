from __future__ import annotations

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
        retry_empty_outputs: bool = False,
    ):
        self.podcasts_path = Path(podcasts_path)
        self.model_names = model_names
        self.config_path = config_path
        self.shard_size = max(1, int(shard_size)) if shard_size else None
        self.retry_empty_outputs = retry_empty_outputs
        self.tokenizer = lambda s: s.lower().split()
        self.detokenizer = lambda tokens: ' '.join(tokens)
        self.excluded_patterns = ('_rover', '_phonemes', '_accent')

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
        result = ROVER(self.tokenizer, self.detokenizer).fit_predict(df)
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

        total_seen = 0
        total_tasks = 0
        total_saved = 0
        failed_shards = 0
        while True:
            shard_path = claim_work_shard(work_plan.work_dir, 0)
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

        logger.info(
            f"ROVER aggregation complete: {total_saved} result(s) saved, "
            f"{total_tasks} task(s) with transcripts, {total_seen} pending audio file(s) seen, "
            f"{failed_shards} failed shard(s)."
        )
