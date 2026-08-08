"""Command-line control plane for a small Balalaika GPU cluster."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .config import ClusterConfig, load_cluster_config, validate_slug
from .db import StateDB
from .scheduler import ClusterScheduler, SchedulerError
from .server import serve_panel
from .transport import executable_prerequisites

DEFAULT_CONFIG = Path("cluster_admin/config.yaml")


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _safe_text(value: object) -> str:
    return "".join(
        character if character.isprintable() else "?" for character in str(value)
    )


def _human_bytes(value: int | float | None) -> str:
    number = float(value or 0)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if abs(number) < 1024 or unit == units[-1]:
            return f"{number:.1f} {unit}"
        number /= 1024
    return f"{number:.1f} TiB"


def _load(path: Path) -> ClusterConfig:
    return load_cluster_config(path)


def _database(config: ClusterConfig) -> StateDB:
    database = StateDB(config.db_path)
    database.initialize()
    return database


def _print_overview(value: dict[str, Any]) -> None:
    run = value.get("active_run") or {}
    if run:
        print(
            f"run={_safe_text(run.get('id'))} state={_safe_text(run.get('state'))} "
            f"progress={float(run.get('progress_percent') or 0):.1f}% "
            f"partitions={run.get('partitions_total', 0)}"
        )
    else:
        print("No runs planned")
    print("\nNODES")
    print(f"{'ID':<18} {'STATE':<11} {'PARTITION':<18} {'GPU':>5} {'DISK FREE':>12}")
    for node in value.get("nodes", []):
        details = node.get("details") or {}
        gpu = details.get("gpu") or {}
        disk = details.get("disk") or {}
        utilization = gpu.get("utilization_percent")
        utilization_text = f"{utilization}%" if utilization is not None else "-"
        print(
            f"{_safe_text(node.get('id', '')):<18.18} "
            f"{_safe_text(node.get('state', '')):<11.11} "
            f"{_safe_text(node.get('current_partition_id') or '-'):<18.18} "
            f"{utilization_text:>5} {_human_bytes(disk.get('free_bytes')):>12}"
        )
    print("\nPARTITIONS")
    print(f"{'ID':<18} {'STATE':<12} {'NODE':<18} {'STAGE':>7} {'PROGRESS':>9}")
    for partition in value.get("partitions", []):
        print(
            f"{_safe_text(partition.get('id', '')):<18.18} "
            f"{_safe_text(partition.get('state', '')):<12.12} "
            f"{_safe_text(partition.get('node_id') or '-'):<18.18} "
            f"{_safe_text(partition.get('current_stage') or '-'):>7.7} "
            f"{float(partition.get('progress_percent') or 0):>8.1f}%"
        )


def command_init(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser()
    if not config_path.exists():
        example = Path(__file__).with_name("config.example.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(example, config_path)
        print(f"Created config template: {config_path}")
        print("Edit node addresses and paths, then run init again.")
        return 0
    config = _load(config_path)
    scheduler = ClusterScheduler(config)
    scheduler.initialize()
    print(f"Initialized controller state: {config.db_path}")
    return 0


def command_nodes(args: argparse.Namespace, config: ClusterConfig) -> int:
    scheduler = ClusterScheduler(config)
    selected = set(args.node_id) if getattr(args, "node_id", None) else None
    if args.nodes_command == "list":
        rows = _database(config).list_nodes()
        if args.json:
            _json(rows)
        else:
            for row in rows:
                print(
                    f"{row['id']:<18} {row['state']:<10} "
                    f"{row['user']}@{row['host']}:{row['port']}"
                )
        return 0
    if args.nodes_command == "bootstrap":
        result = scheduler.bootstrap_nodes(selected)
    elif args.nodes_command == "probe":
        result = scheduler.probe_nodes(selected)
    else:
        raise AssertionError(args.nodes_command)
    _json(result)
    return 0 if all(item.get("ok") for item in result) else 2


def command_run(args: argparse.Namespace, config: ClusterConfig) -> int:
    scheduler = ClusterScheduler(config)
    run_id = validate_slug(args.run_id, "run id")
    if args.run_command == "plan":
        result = scheduler.plan_run(
            run_id,
            partition_count=args.partitions,
            split_state=not args.no_split_state,
        )
        run = result["run"]
        print(
            f"Planned {run['partitions_total']} partitions, "
            f"{_human_bytes(run['input_bytes'])}, "
            f"{float(run['audio_seconds']) / 3600:.1f} audio hours"
        )
        print(f"Manifest: {result['run_manifest_path']}")
        return 0
    if args.run_command == "start":
        overview = scheduler.run_loop(run_id, once=args.once)
        _print_overview(overview)
        state = (overview.get("active_run") or {}).get("state")
        return 1 if state == "FAILED" else 0
    if args.run_command == "retry":
        partition_id = validate_slug(args.partition_id, "partition id")
        scheduler.retry_failed(run_id, partition_id)
        print(f"Queued retry for {run_id}/{partition_id}")
        return 0
    if args.run_command == "cancel":
        partition_id = (
            validate_slug(args.partition_id, "partition id")
            if args.partition_id
            else None
        )
        cancelled = scheduler.cancel(run_id, partition_id)
        target = f"{run_id}/{partition_id}" if partition_id else run_id
        print(f"Cancelled {cancelled} partition(s) for {target}")
        return 0
    raise AssertionError(args.run_command)


def command_status(args: argparse.Namespace, config: ClusterConfig) -> int:
    database = _database(config)
    run_id = validate_slug(args.run_id, "run id") if args.run_id else None
    while True:
        value = database.overview(run_id)
        if args.json:
            if args.watch:
                print(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
            else:
                _json(value)
        else:
            _print_overview(value)
        if not args.watch:
            return 0
        time.sleep(max(2, args.interval))


def command_partitions(args: argparse.Namespace, config: ClusterConfig) -> int:
    run_id = validate_slug(args.run_id, "run id") if args.run_id else None
    rows = _database(config).list_partitions(run_id)
    if args.json:
        _json(rows)
    else:
        for row in rows:
            print(
                f"{row['run_id']}/{row['id']:<18} {row['state']:<12} "
                f"node={row.get('node_id') or '-':<16} "
                f"stage={row.get('current_stage') or '-':<5} "
                f"progress={float(row.get('progress_percent') or 0):.1f}%"
            )
    return 0


def command_doctor(config: ClusterConfig) -> int:
    checks: dict[str, Any] = {
        "executables": executable_prerequisites(),
        "source_root": str(config.source_root),
        "source_root_ok": config.source_root.is_dir(),
        "pipeline_config": str(config.pipeline_config),
        "pipeline_config_ok": config.pipeline_config.is_file(),
        "known_hosts": str(config.known_hosts),
        "known_hosts_ok": config.known_hosts.is_file(),
        "identity": str(config.ssh_identity) if config.ssh_identity else None,
        "identity_ok": not config.ssh_identity or config.ssh_identity.is_file(),
        "nodes": len(config.nodes),
        "gpu_policy": "device=0",
    }
    checks["ok"] = all(
        (
            all(checks["executables"].values()),
            checks["source_root_ok"],
            checks["pipeline_config_ok"],
            checks["known_hosts_ok"],
            checks["identity_ok"],
        )
    )
    _json(checks)
    return 0 if checks["ok"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="cluster YAML config"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="create config template or initialize state")

    nodes = subparsers.add_parser("nodes", help="manage configured worker nodes")
    node_commands = nodes.add_subparsers(dest="nodes_command", required=True)
    node_list = node_commands.add_parser("list")
    node_list.add_argument("--json", action="store_true")
    for name in ("bootstrap", "probe"):
        command = node_commands.add_parser(name)
        command.add_argument("node_id", nargs="*")

    run = subparsers.add_parser("run", help="plan and execute a dataset run")
    run_commands = run.add_subparsers(dest="run_command", required=True)
    plan = run_commands.add_parser("plan")
    plan.add_argument("run_id")
    plan.add_argument("--partitions", type=int)
    plan.add_argument(
        "--no-split-state",
        action="store_true",
        help="do not split an existing balalaika.parquet into partitions",
    )
    start = run_commands.add_parser("start")
    start.add_argument("run_id")
    start.add_argument("--once", action="store_true", help="perform one scheduler tick")
    retry = run_commands.add_parser("retry")
    retry.add_argument("run_id")
    retry.add_argument("partition_id")
    cancel = run_commands.add_parser("cancel")
    cancel.add_argument("run_id")
    cancel.add_argument("partition_id", nargs="?")

    status = subparsers.add_parser("status", help="show controller state")
    status.add_argument("run_id", nargs="?")
    status.add_argument("--json", action="store_true")
    status.add_argument("--watch", action="store_true")
    status.add_argument("--interval", type=int, default=5)

    partitions = subparsers.add_parser("partitions", help="list partitions")
    partitions.add_argument("run_id", nargs="?")
    partitions.add_argument("--json", action="store_true")

    serve = subparsers.add_parser("serve", help="serve the read-only dashboard")
    serve.add_argument("--bind", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    subparsers.add_parser("doctor", help="check local prerequisites")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            return command_init(args)
        config = _load(args.config)
        if args.command == "nodes":
            return command_nodes(args, config)
        if args.command == "run":
            return command_run(args, config)
        if args.command == "status":
            return command_status(args, config)
        if args.command == "partitions":
            return command_partitions(args, config)
        if args.command == "serve":
            serve_panel(config, args.bind, args.port)
            return 0
        if args.command == "doctor":
            return command_doctor(config)
    except (OSError, ValueError, SchedulerError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {_safe_text(exc)}", file=sys.stderr)
        return 2
    parser.error("unknown command")
    return 2
