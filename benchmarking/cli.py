from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .common import REPO_ROOT, load_full_config, parse_gpu_ids, run_git, utc_now
from .resources import filter_gpu_inventory, query_gpu_inventory
from .runner import (
    aggregate_repeats,
    host_metadata,
    make_env,
    repeat_label,
    resolve_source_dataset,
    run_single_repeat,
    selected_gpu_ids,
)
from .sampling import collect_source_samples
from .targets import TARGETS, list_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark harness for Balalaika stages and models.")
    parser.add_argument("--target", type=str, help="Benchmark target name")
    parser.add_argument("--config-path", type=Path, default=REPO_ROOT / "configs" / "config.yaml")
    parser.add_argument("--dataset", type=Path, help="Source dataset root. Defaults to the dataset from config.")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="How many examples to benchmark. Omit or set <= 0 to use all eligible files.",
    )
    parser.add_argument("--sample-mode", choices=("first", "random"), default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=0)
    parser.add_argument("--gpu-ids", type=parse_gpu_ids)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--cpu-workers-per-gpu", type=int)
    parser.add_argument("--cpu-workers-total", type=int)
    parser.add_argument("--batch-size-override", type=int)
    parser.add_argument("--model-name-override", type=str)
    parser.add_argument("--disable-diarization", action="store_true")
    parser.add_argument("--sample-interval-sec", type=float, default=0.5)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "benchmarking" / "reports")
    parser.add_argument("--keep-workdirs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-targets", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_targets:
        list_targets()
        return 0

    if not args.target:
        raise SystemExit("--target is required unless --list-targets is used")

    if args.target not in TARGETS:
        raise SystemExit(f"Unknown target: {args.target}. Use --list-targets.")

    target = TARGETS[args.target]
    base_config = load_full_config(args.config_path.resolve())
    args.dataset = resolve_source_dataset(args, copy.deepcopy(base_config), target)

    if not args.dataset.exists():
        raise SystemExit(f"Dataset does not exist: {args.dataset}")

    gpu_inventory = query_gpu_inventory()
    gpu_ids = selected_gpu_ids(args, gpu_inventory)
    scoped_gpu_inventory = filter_gpu_inventory(gpu_inventory, gpu_ids)
    env = make_env(args, gpu_ids if target.uses_gpu else [])

    selected_samples = collect_source_samples(
        source_dataset=args.dataset,
        target=target,
        config=base_config,
        sample_mode=args.sample_mode,
        num_examples=args.num_examples,
        seed=args.seed,
    )

    if not selected_samples:
        raise SystemExit(
            "No eligible samples were found. Check dataset contents and target prerequisites."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (args.output_root / f"{timestamp}__{target.name.replace('.', '_')}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    selected_samples_path = run_root / "selected_samples.json"
    with selected_samples_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(sample) for sample in selected_samples], handle, indent=2, ensure_ascii=False)

    summary_stub = {
        "report_version": 1,
        "created_at_utc": utc_now(),
        "repo_root": str(REPO_ROOT),
        "git": {
            "branch": run_git(("git", "branch", "--show-current")),
            "commit": run_git(("git", "rev-parse", "HEAD")),
            "status_short": run_git(("git", "status", "--short")),
        },
        "host": host_metadata(),
        "gpu_inventory": scoped_gpu_inventory,
        "target": {
            "name": target.name,
            "description": target.description,
            "modules": list(target.modules),
            "required_sidecars": list(target.required_sidecars),
            "copied_sidecars": list(target.copied_sidecars),
            "uses_gpu": target.uses_gpu,
        },
        "params": {
            "config_path": str(args.config_path.resolve()),
            "source_dataset": str(args.dataset),
            "num_examples_requested": args.num_examples,
            "num_examples_selected": len(selected_samples),
            "sample_mode": args.sample_mode,
            "seed": args.seed,
            "repeats": args.repeats,
            "warmup_repeats": args.warmup_repeats,
            "gpu_ids": gpu_ids,
            "cpu_workers_per_gpu": args.cpu_workers_per_gpu,
            "cpu_workers_total": args.cpu_workers_total,
            "batch_size_override": args.batch_size_override,
            "model_name_override": args.model_name_override,
            "disable_diarization": args.disable_diarization,
            "sample_interval_sec": args.sample_interval_sec,
            "keep_workdirs": args.keep_workdirs,
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        },
        "selected_samples_path": str(selected_samples_path),
    }

    if args.dry_run:
        print(json.dumps(summary_stub, indent=2, ensure_ascii=False))
        return 0

    warmups: List[Dict[str, Any]] = []
    for warmup_index in range(1, args.warmup_repeats + 1):
        label = repeat_label("warmup", warmup_index)
        warmups.append(
            run_single_repeat(
                label=label,
                target=target,
                sample_records=selected_samples,
                base_config=base_config,
                run_root=run_root,
                args=args,
                gpu_ids=gpu_ids,
                env=env,
            )
        )

    repeats: List[Dict[str, Any]] = []
    for repeat_index in range(1, args.repeats + 1):
        label = repeat_label("repeat", repeat_index)
        repeats.append(
            run_single_repeat(
                label=label,
                target=target,
                sample_records=selected_samples,
                base_config=base_config,
                run_root=run_root,
                args=args,
                gpu_ids=gpu_ids,
                env=env,
            )
        )

    report = dict(summary_stub)
    report["warmups"] = warmups
    report["repeats"] = repeats
    report["aggregate"] = aggregate_repeats(repeats)
    report["successful"] = report["aggregate"]["failed_repeats"] == 0

    report_path = run_root / "report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    final_summary = {
        "report_path": str(report_path),
        "target": target.name,
        "successful": report["successful"],
        "successful_repeats": report["aggregate"]["successful_repeats"],
        "avg_rtf": report["aggregate"]["rtf"]["avg"],
        "avg_gpu_util_percent": report["aggregate"]["gpu_util_percent"]["avg"],
        "avg_gpu_vram_mb": report["aggregate"]["gpu_vram_mb"]["avg"],
        "avg_rss_gb": report["aggregate"]["rss_gb"]["avg"],
        "avg_cpu_util_percent": report["aggregate"]["cpu_util_percent"]["avg"],
    }
    print(json.dumps(final_summary, ensure_ascii=False))
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
