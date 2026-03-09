from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .common import REPO_ROOT, eprint, ensure_dict, preserve_repeat_artifacts, summarize_numeric, utc_now, write_yaml
from .models import CommandSpec, SampleRecord, TargetSpec
from .resources import ResourceSampler, build_runtime_library_paths, summarize_resource_samples
from .sampling import copy_benchmark_dataset
from .targets import build_commands


def run_command(
    command: CommandSpec,
    config_path: Path,
    log_path: Path,
    gpu_ids: Optional[List[int]],
    sample_interval_sec: float,
    env: Dict[str, str],
) -> Dict[str, Any]:
    argv = list(command.argv) + [str(config_path)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.monotonic()

    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            argv,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        sampler = ResourceSampler(pid=process.pid, gpu_ids=gpu_ids, interval_sec=sample_interval_sec)
        sampler.start()
        return_code = process.wait()
        sampler.stop()

    wall_time_sec = time.monotonic() - started
    finished_at = utc_now()
    summary = summarize_resource_samples(sampler.samples)
    return {
        "name": command.name,
        "argv": argv,
        "log_path": str(log_path),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "wall_time_sec": wall_time_sec,
        "return_code": return_code,
        "resource_summary": summary,
        "resource_samples": sampler.samples,
        "gpu_query_error": sampler.gpu_query_error,
    }


def aggregate_repeats(repeats: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [repeat for repeat in repeats if repeat.get("success")]
    return {
        "total_repeats": len(repeats),
        "successful_repeats": len(successful),
        "failed_repeats": len(repeats) - len(successful),
        "wall_time_sec": summarize_numeric(repeat.get("wall_time_sec") for repeat in successful),
        "rtf": summarize_numeric(repeat.get("rtf") for repeat in successful),
        "x_realtime": summarize_numeric(repeat.get("x_realtime") for repeat in successful),
        "cpu_util_percent": summarize_numeric(
            repeat.get("resource_summary", {}).get("cpu_util_percent", {}).get("avg")
            for repeat in successful
        ),
        "rss_gb": summarize_numeric(
            repeat.get("resource_summary", {}).get("rss_gb", {}).get("avg") for repeat in successful
        ),
        "gpu_util_percent": summarize_numeric(
            repeat.get("resource_summary", {}).get("gpu_util_percent", {}).get("avg")
            for repeat in successful
        ),
        "gpu_vram_mb": summarize_numeric(
            repeat.get("resource_summary", {}).get("gpu_vram_mb", {}).get("avg")
            for repeat in successful
        ),
    }


def selected_gpu_ids(args: argparse.Namespace, inventory: Dict[str, Any]) -> Optional[List[int]]:
    if args.gpu_ids is not None:
        return args.gpu_ids
    if args.num_gpus is not None:
        return list(range(args.num_gpus))
    gpu_inventory = inventory.get("gpus", [])
    if not gpu_inventory:
        return []
    return [int(gpu["index"]) for gpu in gpu_inventory]


def make_env(args: argparse.Namespace, gpu_ids: Optional[List[int]]) -> Dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    ld_library_parts = build_runtime_library_paths()
    if env.get("LD_LIBRARY_PATH"):
        ld_library_parts.append(env["LD_LIBRARY_PATH"])
    if ld_library_parts:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_library_parts)

    if gpu_ids is not None and gpu_ids != []:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    return env


def repeat_label(prefix: str, index: int) -> str:
    return f"{prefix}_{index:02d}"


def run_single_repeat(
    label: str,
    target: TargetSpec,
    sample_records: Sequence[SampleRecord],
    base_config: Dict[str, Any],
    run_root: Path,
    args: argparse.Namespace,
    gpu_ids: Optional[List[int]],
    env: Dict[str, str],
) -> Dict[str, Any]:
    repeat_root = run_root / label
    repeat_root.mkdir(parents=True, exist_ok=True)

    dataset_root = repeat_root / "dataset"
    copy_benchmark_dataset(destination_dataset=dataset_root, sample_records=sample_records)

    effective_config = copy.deepcopy(base_config)
    if target.mutator is not None:
        target.mutator(effective_config, args, dataset_root)

    config_path = repeat_root / "config.yaml"
    write_yaml(config_path, effective_config)

    commands = build_commands(target.modules)
    command_results: List[Dict[str, Any]] = []
    all_samples: List[Dict[str, Any]] = []
    success = True
    error_message = None

    for index, command in enumerate(commands, start=1):
        log_path = repeat_root / f"command_{index:02d}_{command.name.replace('.', '_')}.log"
        eprint(f"[{label}] running {command.name}")
        result = run_command(
            command=command,
            config_path=config_path,
            log_path=log_path,
            gpu_ids=gpu_ids if target.uses_gpu else [],
            sample_interval_sec=args.sample_interval_sec,
            env=env,
        )
        all_samples.extend(result.pop("resource_samples"))
        command_results.append(result)
        if result["return_code"] != 0:
            success = False
            error_message = f"{command.name} failed with exit code {result['return_code']}"
            break

    samples_path = repeat_root / "resource_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for sample in all_samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    total_audio_sec = sum(record.duration_sec for record in sample_records)
    wall_time_sec = sum(command_result["wall_time_sec"] for command_result in command_results)
    rtf = (wall_time_sec / total_audio_sec) if total_audio_sec > 0 else None
    x_realtime = (total_audio_sec / wall_time_sec) if wall_time_sec > 0 else None

    repeat_result = {
        "label": label,
        "repeat_root": str(repeat_root),
        "dataset_root": str(dataset_root),
        "config_path": str(config_path),
        "samples_path": str(samples_path),
        "success": success,
        "error": error_message,
        "command_results": command_results,
        "wall_time_sec": wall_time_sec,
        "total_audio_sec": total_audio_sec,
        "rtf": rtf,
        "x_realtime": x_realtime,
        "resource_summary": summarize_resource_samples(all_samples),
        "preserved_artifacts": preserve_repeat_artifacts(dataset_root, repeat_root),
    }

    if not args.keep_workdirs:
        shutil.rmtree(dataset_root, ignore_errors=True)

    return repeat_result


def host_metadata() -> Dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpu_count": os.cpu_count(),
        "physical_cpu_count": None,
    }


def resolve_source_dataset(args: argparse.Namespace, config: Dict[str, Any], target: TargetSpec) -> Path:
    if args.dataset:
        return args.dataset.resolve()

    fallback_sections = {
        "collate.stage": "download",
        "pipeline.base": "preprocess",
    }
    preferred_section = fallback_sections.get(target.name)
    if preferred_section:
        preferred = ensure_dict(config, preferred_section).get("podcasts_path")
        if preferred:
            return Path(preferred).resolve()

    for key in ("preprocess", "separation", "transcription", "punctuation", "accent", "phonemizer", "download"):
        value = ensure_dict(config, key).get("podcasts_path")
        if value:
            return Path(value).resolve()

    raise ValueError("Dataset path was not provided and could not be inferred from config")
