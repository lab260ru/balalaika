from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.utils.utils import get_audio_paths

from .common import copy_file, eprint, get_audio_duration, sidecar_path
from .models import SampleRecord, TargetSpec
from .targets import min_duration_for_target


def collect_source_samples(
    source_dataset: Path,
    target: TargetSpec,
    config: Dict[str, Any],
    sample_mode: str,
    num_examples: Optional[int],
    seed: int,
) -> List[SampleRecord]:
    all_audio_paths = sorted(Path(path) for path in get_audio_paths(str(source_dataset)))
    minimum_duration = min_duration_for_target(target, config)
    invalid_audio_count = 0
    invalid_audio_examples: List[str] = []

    def build_record(audio_path: Path) -> Optional[SampleRecord]:
        nonlocal invalid_audio_count
        for suffix in target.required_sidecars:
            if not sidecar_path(audio_path, suffix).exists():
                return None
        try:
            if audio_path.stat().st_size == 0:
                raise RuntimeError("empty file")
            duration_sec = get_audio_duration(audio_path)
        except Exception as exc:
            invalid_audio_count += 1
            if len(invalid_audio_examples) < 5:
                invalid_audio_examples.append(f"{audio_path}: {exc}")
            return None
        if minimum_duration is not None and duration_sec <= minimum_duration:
            return None
        try:
            relative_path = str(audio_path.relative_to(source_dataset))
        except ValueError:
            return None
        copied_sidecars = tuple(
            suffix for suffix in target.copied_sidecars if sidecar_path(audio_path, suffix).exists()
        )
        return SampleRecord(
            audio_path=str(audio_path),
            relative_path=relative_path,
            duration_sec=duration_sec,
            copied_sidecars=copied_sidecars,
        )

    if sample_mode == "first":
        records: List[SampleRecord] = []
        for audio_path in all_audio_paths:
            record = build_record(audio_path)
            if not record:
                continue
            records.append(record)
            if num_examples is not None and num_examples > 0 and len(records) >= num_examples:
                break
        if invalid_audio_count:
            eprint(f"Skipped {invalid_audio_count} unreadable audio files during sample selection.")
            for example in invalid_audio_examples:
                eprint(f"  {example}")
        return records

    candidates = [record for record in (build_record(path) for path in all_audio_paths) if record]
    if invalid_audio_count:
        eprint(f"Skipped {invalid_audio_count} unreadable audio files during sample selection.")
        for example in invalid_audio_examples:
            eprint(f"  {example}")
    random.Random(seed).shuffle(candidates)
    if num_examples is not None and num_examples > 0:
        return candidates[:num_examples]
    return candidates


def copy_benchmark_dataset(destination_dataset: Path, sample_records: Sequence[SampleRecord]) -> None:
    if destination_dataset.exists():
        shutil.rmtree(destination_dataset)
    destination_dataset.mkdir(parents=True, exist_ok=True)

    for record in sample_records:
        src_audio = Path(record.audio_path)
        dst_audio = destination_dataset / record.relative_path
        copy_file(src_audio, dst_audio)
        for suffix in record.copied_sidecars:
            src_sidecar = sidecar_path(src_audio, suffix)
            dst_sidecar = sidecar_path(dst_audio, suffix)
            if src_sidecar.exists():
                copy_file(src_sidecar, dst_sidecar)
