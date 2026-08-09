from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import cluster_admin.node_runner as node_runner
from cluster_admin.config import ClusterConfig, NodeConfig
from cluster_admin.scheduler import ClusterScheduler


def _ready_attempt(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    work_root = tmp_path / "worker"
    work_root.mkdir()
    manifest = {"files": [], "files_total": 0, "input_bytes": 0}
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    config_bytes = b"preprocess: {}\n"
    request = {
        "work_root": str(work_root),
        "run_id": "direct-safety",
        "partition_id": "part-0000",
        "global_rank": 0,
        "attempt_id": "a" * 32,
        "attempt_ordinal": 1,
        "fencing_token": "b" * 32,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    prepared = node_runner.operation_prepare(
        request | {"operation": "prepare"}
    )
    control = Path(prepared["control_root"])
    control.joinpath("manifest.json").write_bytes(manifest_bytes)
    control.joinpath("config.yaml").write_bytes(config_bytes)
    ready = node_runner.operation_validate_input(
        request | {"operation": "validate_input"}
    )
    assert ready["state"] == "READY"
    return request, prepared


def _direct_runtime_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    pipeline_root = tmp_path / "pipeline"
    pipeline_root.mkdir()
    base = pipeline_root / "base.sh"
    base.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="ascii")
    base.chmod(0o755)

    renderer = pipeline_root / "docker" / "prepare_config.py"
    renderer.parent.mkdir()
    renderer.write_text(
        """import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
shutil.copyfile(args.input, args.output)
""",
        encoding="ascii",
    )

    venv_path = tmp_path / "venv"
    bin_path = venv_path / "bin"
    bin_path.mkdir(parents=True)
    bin_path.joinpath("activate").write_text("# test venv\n", encoding="ascii")
    shutil.copy2(sys.executable, bin_path / "python")

    models_root = tmp_path / "models"
    models_root.mkdir()
    models_root.joinpath("model.bin").write_bytes(b"model")
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    return pipeline_root, venv_path, models_root, cache_root


def _direct_start_request(
    request: dict[str, Any],
    pipeline_root: Path,
    venv_path: Path,
    models_root: Path,
    cache_root: Path,
) -> dict[str, Any]:
    return request | {
        "operation": "start",
        "runtime": "direct",
        "gpu_devices": [0],
        "gpu_uuids": ["GPU-direct00"],
        "image": "unused-in-direct:latest",
        "pipeline_root": str(pipeline_root),
        "venv_path": str(venv_path),
        "models_root": str(models_root),
        "cache_root": str(cache_root),
        "env_file": None,
        "stage_start": "1",
        "stage_stop": "2",
        "shm_size": "8g",
    }


def test_direct_cancel_does_not_signal_reused_pipeline_process_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    request, prepared = _ready_attempt(tmp_path)
    control = Path(prepared["control_root"])
    job_path = control / "job.json"
    process_path = control / "direct-process.json"
    metadata = json.loads(job_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "state": "RUNNING",
            "runtime": "direct",
            "spec_sha256": "c" * 64,
            "supervisor_pid": 41001,
            "supervisor_start_ticks": 501,
            "pipeline_pid": 41002,
            "pipeline_pgid": 41002,
            "pipeline_start_ticks": 502,
            "boot_id": "test-boot-id",
        }
    )
    job_path.write_text(json.dumps(metadata), encoding="utf-8")
    process_path.write_text(json.dumps(metadata), encoding="utf-8")

    group_alive = {"value": True}
    signals: list[tuple[int, int]] = []

    monkeypatch.setattr(node_runner, "_boot_id", lambda: "test-boot-id")
    monkeypatch.setattr(
        node_runner, "_process_start_ticks", lambda _pid: 999999
    )
    monkeypatch.setattr(
        node_runner,
        "_process_group_alive",
        lambda _pgid: group_alive["value"],
    )

    def record_signal(pgid: int, signum: int) -> None:
        signals.append((pgid, signum))
        group_alive["value"] = False

    monkeypatch.setattr(node_runner.os, "killpg", record_signal)

    with pytest.raises(node_runner.RunnerError, match="stale PID identity"):
        node_runner.operation_cancel(request | {"operation": "cancel"})
    assert signals == []


def test_direct_start_without_handshake_returns_to_ready_and_can_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    request, prepared = _ready_attempt(tmp_path)
    pipeline_root, venv_path, models_root, cache_root = _direct_runtime_paths(
        tmp_path
    )
    start_request = _direct_start_request(
        request, pipeline_root, venv_path, models_root, cache_root
    )
    monkeypatch.setattr(
        node_runner,
        "_direct_gpu_probe",
        lambda _venv, _devices: (
            [
                {
                    "index": 0,
                    "uuid": "GPU-direct00",
                    "memory_used_mib": 0,
                    "utilization_percent": 0,
                }
            ],
            None,
        ),
    )

    real_popen = subprocess.Popen

    class MissingHandshakeSupervisor:
        pid = 42001

        @staticmethod
        def poll() -> int:
            return 1

    def fail_supervisor_only(*args: Any, **kwargs: Any):
        argv = args[0]
        if isinstance(argv, list) and "direct-worker" in argv:
            return MissingHandshakeSupervisor()
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(node_runner.subprocess, "Popen", fail_supervisor_only)

    with pytest.raises(node_runner.RunnerError, match="startup handshake"):
        node_runner.operation_start(start_request)

    control = Path(prepared["control_root"])
    recovered = json.loads(
        control.joinpath("job.json").read_text(encoding="utf-8")
    )
    assert recovered["state"] == "READY"
    assert not control.joinpath("direct-process.json").exists()
    assert not control.joinpath("direct-exit.json").exists()
    assert not control.joinpath("direct-spec.json").exists()

    monkeypatch.setattr(node_runner.subprocess, "Popen", real_popen)
    retried = node_runner.operation_start(start_request)
    assert retried["state"] in {"RUNNING", "COMPLETED"}

    deadline = time.monotonic() + 5
    while retried["state"] == "RUNNING" and time.monotonic() < deadline:
        time.sleep(0.02)
        retried = node_runner.operation_status(
            request | {"operation": "status"}
        )
    assert retried["state"] == "COMPLETED"
    assert retried["exit_code"] == 0


def test_direct_environment_drops_process_injection_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in ("LD_PRELOAD", "PYTHONPATH", "BASH_ENV"):
        monkeypatch.setenv(name, f"/ambient/{name.lower()}")
    monkeypatch.setenv("BALALAIKA_SAFE_AMBIENT", "ambient-value")

    env_file = tmp_path / "worker.env"
    env_file.write_text(
        "BALALAIKA_SAFE_FILE=file-value\n",
        encoding="ascii",
    )
    paths = {
        "runtime_config": tmp_path / "runtime.yaml",
        "data": tmp_path / "data",
        "logs": tmp_path / "logs",
        "output": tmp_path / "output",
    }
    for name in ("data", "logs", "output"):
        paths[name].mkdir()
    venv_path = tmp_path / "venv"
    models_root = tmp_path / "models"
    cache_root = tmp_path / "cache"
    venv_path.mkdir()
    models_root.mkdir()
    cache_root.mkdir()

    environment = node_runner._direct_environment(
        paths,
        venv_path,
        models_root,
        cache_root,
        ["GPU-direct00"],
        env_file,
        "part-0000",
        0,
    )

    assert not {"LD_PRELOAD", "PYTHONPATH", "BASH_ENV"} & environment.keys()
    assert environment["BALALAIKA_SAFE_FILE"] == "file-value"


@pytest.mark.parametrize("name", ("LD_PRELOAD", "PYTHONPATH", "BASH_ENV"))
def test_direct_environment_rejects_process_injection_from_env_file(
    name: str, tmp_path: Path
) -> None:
    env_file = tmp_path / "worker.env"
    env_file.write_text(f"{name}=/untrusted/value\n", encoding="ascii")
    paths = {
        "runtime_config": tmp_path / "runtime.yaml",
        "data": tmp_path / "data",
        "logs": tmp_path / "logs",
        "output": tmp_path / "output",
    }
    for directory in paths.values():
        directory.parent.mkdir(parents=True, exist_ok=True)
    venv_path = tmp_path / "venv"
    models_root = tmp_path / "models"
    cache_root = tmp_path / "cache"
    venv_path.mkdir()
    models_root.mkdir()
    cache_root.mkdir()

    with pytest.raises(
        node_runner.RunnerError, match=f"forbidden variable {name}"
    ):
        node_runner._direct_environment(
            paths,
            venv_path,
            models_root,
            cache_root,
            ["GPU-direct00"],
            env_file,
            "part-0000",
            0,
        )


class _BusyGpuTransport:
    def __init__(self, memory_used_mib: int, utilization_percent: int):
        self.memory_used_mib = memory_used_mib
        self.utilization_percent = utilization_percent

    def probe(self, node: NodeConfig) -> dict[str, Any]:
        return {
            "ok": True,
            "state": "ONLINE",
            "runtime": node.runtime,
            "docker_ok": True,
            "gpu_ok": True,
            "gpu_devices": list(node.gpu_devices),
            "gpu_uuids": ["GPU-busy0000"],
            "gpus": [
                {
                    "index": 0,
                    "uuid": "GPU-busy0000",
                    "memory_used_mib": self.memory_used_mib,
                    "utilization_percent": self.utilization_percent,
                }
            ],
            "image_id": "sha256:" + "d" * 64,
            "models_ok": True,
            "runner_schema": node_runner.SCHEMA_VERSION,
            "disk": {"free_bytes": 10**12},
        }


def _scheduler_config(tmp_path: Path, node: NodeConfig) -> ClusterConfig:
    source_root = tmp_path / "dataset"
    item = source_root / "item-a"
    item.mkdir(parents=True)
    item.joinpath("audio.flac").write_bytes(b"audio")
    item.joinpath("metadata.json").write_text(
        '{"id":"item-a"}\n', encoding="ascii"
    )
    pipeline_config = tmp_path / "pipeline.yaml"
    pipeline_config.write_text("preprocess: {}\n", encoding="ascii")
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("test key\n", encoding="ascii")
    return ClusterConfig(
        path=tmp_path / "cluster.yaml",
        state_dir=tmp_path / "state",
        source_root=source_root,
        pipeline_config=pipeline_config,
        image="balalaika@sha256:" + "1" * 64,
        ssh_identity=None,
        known_hosts=known_hosts,
        connect_timeout=2,
        poll_seconds=2,
        max_attempts=2,
        partitions_per_node=1,
        group_depth=1,
        audio_extensions=(".flac",),
        stage_start="1",
        stage_stop="15",
        shm_size="8g",
        nodes=(node,),
        disk_headroom_bytes=0,
    )


@pytest.mark.parametrize(
    ("memory_used_mib", "utilization_percent", "error_fragment"),
    ((513, 0, "513 MiB"), (0, 11, "11% utilization")),
)
def test_busy_gpu_threshold_rejects_node_before_partition_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    memory_used_mib: int,
    utilization_percent: int,
    error_fragment: str,
) -> None:
    node = NodeConfig(
        id="node-a",
        host="node-a.example.test",
        user="balalaika",
        port=22,
        work_root="/var/lib/balalaika",
        models_root="/var/lib/balalaika/models",
        cache_root="/var/lib/balalaika/cache",
        max_gpu_memory_used_mib=512,
        max_gpu_utilization_percent=10,
    )
    transport = _BusyGpuTransport(memory_used_mib, utilization_percent)
    scheduler = ClusterScheduler(
        _scheduler_config(tmp_path, node),
        transport=transport,  # type: ignore[arg-type]
    )
    scheduler.plan_run("busy-gpu", partition_count=1, split_state=False)

    def fail_if_claimed(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("busy node reached partition claim")

    monkeypatch.setattr(scheduler.db, "claim_next", fail_if_claimed)
    overview = scheduler.tick("busy-gpu")

    assert overview["partitions"][0]["state"] == "QUEUED"
    persisted = scheduler.db.get_node("node-a")
    assert persisted is not None
    assert persisted["state"] == "DEGRADED"
    assert error_fragment in persisted["error"]
