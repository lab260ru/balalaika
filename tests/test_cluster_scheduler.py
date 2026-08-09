from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

import cluster_admin.node_runner as node_runner
from cluster_admin.config import ClusterConfig, NodeConfig, load_cluster_config
from cluster_admin.node_runner import SCHEMA_VERSION
from cluster_admin.scheduler import ClusterScheduler, SchedulerError
from cluster_admin.transport import ClusterTransport, TransportError


class FakeTransport:
    """In-process worker protocol used without SSH, rsync, or Docker."""

    def __init__(self, root: Path):
        self.root = root
        self.calls: list[tuple[str, str, str | None]] = []
        self.jobs: dict[str, dict[str, Any]] = {}
        self.pull_count = 0
        self.prepare_requests: list[dict[str, Any]] = []
        self.rpc_requests: list[dict[str, Any]] = []
        self.partition_sources: list[Path] = []
        self.pulled_remote_roots: list[str] = []
        self.probe_overrides: dict[str, Any] = {}
        self.lost_cancel_responses: set[str] = set()
        self.stale_result_tokens: dict[str, str] = {}

    def bootstrap(self, node: NodeConfig) -> dict[str, Any]:
        self.calls.append(("bootstrap", node.id, None))
        return {"ok": True, "runner_path": node.runner_path}

    def probe(self, node: NodeConfig) -> dict[str, Any]:
        self.calls.append(("probe", node.id, None))
        return {
            "ok": True,
            "state": "ONLINE",
            "runtime": "docker",
            "docker_ok": True,
            "gpu_ok": True,
            "gpu_devices": [0],
            "gpu_uuids": ["GPU-fake0000"],
            "image_id": "sha256:" + "a" * 64,
            "models_ok": True,
            "runner_schema": SCHEMA_VERSION,
            "gpu": {
                "index": 0,
                "memory_used_mib": 0,
                "utilization_percent": 0,
            },
            "gpus": [
                {
                    "index": 0,
                    "memory_used_mib": 0,
                    "utilization_percent": 0,
                }
            ],
            "disk": {"free_bytes": 10**12},
            **self.probe_overrides,
        }

    def rpc(
        self,
        node: NodeConfig,
        request: dict[str, Any],
        *,
        timeout: int = 60,
    ) -> dict[str, Any]:
        del timeout
        operation = request["operation"]
        attempt_id = request.get("attempt_id")
        self.calls.append((operation, node.id, attempt_id))
        self.rpc_requests.append(dict(request))
        if operation == "prepare":
            return self._prepare(node, request)

        job = self.jobs[attempt_id]
        assert request["fencing_token"] == job["fencing_token"]
        if operation == "validate_input":
            return self._validate_input(job)
        if operation == "start":
            job.update(request)
            job["state"] = "RUNNING"
            return {
                "ok": True,
                "state": "RUNNING",
                "container_name": f"fake-{attempt_id[:12]}",
            }
        if operation == "status":
            return self._status(job)
        if operation == "cancel":
            job["state"] = "CANCELLED"
            if attempt_id in self.lost_cancel_responses:
                raise TransportError("cancel response was lost")
            return {
                "ok": True,
                "state": "CANCELLED",
                "attempt_id": attempt_id,
            }
        raise AssertionError(f"Unexpected RPC operation: {operation}")

    def _prepare(self, node: NodeConfig, request: dict[str, Any]) -> dict[str, Any]:
        self.prepare_requests.append(dict(request))
        attempt_id = request["attempt_id"]
        attempt_root = (
            self.root
            / node.id
            / request["run_id"]
            / request["partition_id"]
            / attempt_id
        )
        control_root = attempt_root / "control"
        data_partial = attempt_root / "data.partial"
        control_root.mkdir(parents=True, exist_ok=True)
        data_partial.mkdir(parents=True, exist_ok=True)
        self.jobs[attempt_id] = {
            **request,
            "node_id": node.id,
            "state": "STAGING",
            "control_root": control_root,
            "data_partial": data_partial,
        }
        return {
            "ok": True,
            "state": "STAGING",
            "control_root": str(control_root),
            "data_partial": str(data_partial),
        }

    @staticmethod
    def _validate_input(job: dict[str, Any]) -> dict[str, Any]:
        control_root = job["control_root"]
        manifest_path = control_root / "manifest.json"
        config_path = control_root / "config.yaml"
        assert _sha256(manifest_path) == job["manifest_sha256"]
        assert _sha256(config_path) == job["config_sha256"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["files"]:
            transferred = job["data_partial"].joinpath(*item["path"].split("/"))
            assert transferred.is_file()
            assert transferred.stat().st_size == item["size"]
        job["state"] = "READY"
        return {
            "ok": True,
            "state": "READY",
            "files_total": manifest["files_total"],
            "input_bytes": manifest["input_bytes"],
        }

    @staticmethod
    def _status(job: dict[str, Any]) -> dict[str, Any]:
        response = {
            "ok": True,
            "state": job["state"],
            "attempt_id": job["attempt_id"],
            "fencing_token": job["fencing_token"],
            "manifest_sha256": job["manifest_sha256"],
            "config_sha256": job["config_sha256"],
            "container_name": f"fake-{job['attempt_id'][:12]}",
        }
        if job["state"] == "RUNNING":
            response.update(
                current_stage="4",
                progress_percent=25.0,
                files_processed=0,
            )
        return response

    def push_file(self, node: NodeConfig, source: Path, remote_path: str) -> None:
        self.calls.append(("push_file", node.id, source.name))
        target = Path(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    def push_partition_files(
        self,
        node: NodeConfig,
        source_root: Path,
        files_list_path: Path,
        remote_data_partial: str,
    ) -> None:
        self.calls.append(("push_partition", node.id, files_list_path.name))
        self.partition_sources.append(source_root)
        encoded_paths = files_list_path.read_bytes().split(b"\0")
        for encoded in encoded_paths:
            if not encoded:
                continue
            relative = PurePosixPath(encoded.decode("utf-8"))
            source = source_root.joinpath(*relative.parts)
            target = Path(remote_data_partial).joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    def pull_attempt(
        self,
        node: NodeConfig,
        remote_attempt_root: str,
        local_partial: Path,
    ) -> None:
        self.calls.append(("pull_attempt", node.id, remote_attempt_root))
        self.pull_count += 1
        self.pulled_remote_roots.append(remote_attempt_root)
        attempt_id = next(
            candidate
            for candidate in self.jobs
            if candidate[:12] in PurePosixPath(remote_attempt_root).name
        )
        job = self.jobs[attempt_id]
        token = self.stale_result_tokens.get(attempt_id, job["fencing_token"])
        local_partial.mkdir(parents=True, exist_ok=True)
        result = {
            "attempt_id": attempt_id,
            "fencing_token": token,
            "manifest_sha256": job["manifest_sha256"],
            "config_sha256": job["config_sha256"],
            "exit_code": 0,
        }
        (local_partial / "result.json").write_text(json.dumps(result), encoding="utf-8")
        (local_partial / "_SUCCESS").write_text(token + "\n", encoding="ascii")

    def complete(self, attempt_id: str) -> None:
        self.jobs[attempt_id]["state"] = "COMPLETED"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _node(node_id: str = "node-a") -> NodeConfig:
    return NodeConfig(
        id=node_id,
        host=f"{node_id}.example.test",
        user="balalaika",
        port=22,
        work_root="/var/lib/balalaika",
        models_root="/var/lib/balalaika/models",
        cache_root="/var/lib/balalaika/cache",
    )


def _config(tmp_path: Path, *, nodes: tuple[NodeConfig, ...] | None = None):
    source_root = tmp_path / "dataset"
    for name, size in (("video-a", 31), ("video-b", 17)):
        audio = source_root / name / "audio.flac"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(name.encode("ascii").ljust(size, b"."))
        (audio.parent / "metadata.json").write_text(
            json.dumps({"id": name}), encoding="utf-8"
        )
    pipeline_config = tmp_path / "pipeline.yaml"
    pipeline_config.write_text("preprocess: {}\n", encoding="ascii")
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("fake host key\n", encoding="ascii")
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
        partitions_per_node=2,
        group_depth=1,
        audio_extensions=(".flac",),
        stage_start="1",
        stage_stop="15",
        shm_size="8g",
        nodes=nodes or (_node(),),
    )


def _loaded_config(tmp_path: Path, node_values: dict[str, Any]) -> ClusterConfig:
    pipeline_config = tmp_path / "loaded-pipeline.yaml"
    pipeline_config.write_text("preprocess: {}\n", encoding="ascii")
    known_hosts = tmp_path / "loaded-known-hosts"
    known_hosts.write_text("fake host key\n", encoding="ascii")
    raw = {
        "controller": {
            "source_root": str(tmp_path / "loaded-dataset"),
            "state_dir": str(tmp_path / "loaded-state"),
        },
        "pipeline": {
            "config": str(pipeline_config),
            "image": "balalaika@sha256:" + "1" * 64,
        },
        "ssh": {"known_hosts": str(known_hosts)},
        "nodes": [
            {
                "id": "node-a",
                "host": "node-a.example.test",
                "user": "balalaika",
                "work_root": "/var/lib/balalaika",
                "models_root": "/var/lib/balalaika/models",
                "cache_root": "/var/lib/balalaika/cache",
                **node_values,
            }
        ],
    }
    path = tmp_path / "loaded-cluster.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_cluster_config(path)


def _current_attempt(scheduler: ClusterScheduler, run_id: str) -> dict[str, Any]:
    attempts = scheduler.db.list_active_attempts(run_id)
    assert len(attempts) == 1
    return attempts[0]


def _ready_runner_attempt(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    work_root = tmp_path / "runner-work"
    work_root.mkdir()
    manifest = {"files": [], "files_total": 0, "input_bytes": 0}
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    config_bytes = b"preprocess: {}\n"
    request = {
        "work_root": str(work_root),
        "run_id": "runtime-run",
        "partition_id": "part-0000",
        "global_rank": 0,
        "attempt_id": "a" * 32,
        "attempt_ordinal": 1,
        "fencing_token": "b" * 32,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    prepared = node_runner.operation_prepare(request | {"operation": "prepare"})
    control = Path(prepared["control_root"])
    control.joinpath("manifest.json").write_bytes(manifest_bytes)
    control.joinpath("config.yaml").write_bytes(config_bytes)
    ready = node_runner.operation_validate_input(
        request | {"operation": "validate_input"}
    )
    assert ready["state"] == "READY"
    return request, prepared


def _direct_runtime_paths(
    tmp_path: Path, script: str
) -> tuple[Path, Path, Path, Path]:
    pipeline_root = tmp_path / "direct-pipeline"
    pipeline_root.mkdir()
    base = pipeline_root / "base.sh"
    base.write_text(script, encoding="utf-8")
    base.chmod(0o755)

    venv_path = tmp_path / "direct-venv"
    bin_dir = venv_path / "bin"
    bin_dir.mkdir(parents=True)
    bin_dir.joinpath("activate").write_text("# test venv\n", encoding="ascii")
    shutil.copy2(sys.executable, bin_dir / "python")
    shutil.copy2(sys.executable, bin_dir / "python3")

    docker_dir = pipeline_root / "docker"
    docker_dir.mkdir()
    docker_dir.joinpath("prepare_config.py").write_text(
        """import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
shutil.copyfile(args.input, args.output)
""",
        encoding="utf-8",
    )

    models_root = tmp_path / "direct-models"
    models_root.mkdir()
    models_root.joinpath("model.bin").write_bytes(b"model")
    cache_root = tmp_path / "direct-cache"
    cache_root.mkdir()
    return pipeline_root, venv_path, models_root, cache_root


def _deny_docker_calls(monkeypatch) -> None:
    original_run_command = node_runner.run_command

    def guarded_run_command(argv, **kwargs):
        if Path(argv[0]).name == "docker":
            raise AssertionError("direct runtime called Docker CLI")
        return original_run_command(argv, **kwargs)

    def guarded_docker_inspect(*args, **kwargs):
        raise AssertionError("direct runtime inspected a Docker container")

    monkeypatch.setattr(node_runner, "run_command", guarded_run_command)
    monkeypatch.setattr(node_runner, "docker_inspect", guarded_docker_inspect)


def test_scheduler_end_to_end_queues_one_partition_per_node(tmp_path):
    config = _config(tmp_path)
    transport = FakeTransport(tmp_path / "fake-workers")
    scheduler = ClusterScheduler(config, transport=transport)

    plan = scheduler.plan_run("run-e2e", partition_count=2, split_state=False)
    assert len(plan["partitions"]) == 2

    scheduler.tick("run-e2e")
    first = _current_attempt(scheduler, "run-e2e")
    assert first["state"] == "RUNNING"
    assert [item["state"] for item in scheduler.db.list_partitions("run-e2e")] == [
        "RUNNING",
        "QUEUED",
    ]
    assert [call[0] for call in transport.calls].count("start") == 1

    # A reconciliation pass must not claim the queued partition while GPU 0 is busy.
    scheduler.tick("run-e2e")
    assert _current_attempt(scheduler, "run-e2e")["id"] == first["id"]
    assert [call[0] for call in transport.calls].count("start") == 1
    assert (
        scheduler.db.get_partition("run-e2e", "part-0000")["progress_percent"] == 25.0
    )

    transport.complete(first["id"])
    scheduler.tick("run-e2e")
    second = _current_attempt(scheduler, "run-e2e")
    assert second["id"] != first["id"]
    assert transport.pull_count == 1
    assert [call[0] for call in transport.calls].count("start") == 2
    start_requests = [
        request for request in transport.rpc_requests if request["operation"] == "start"
    ]
    assert [
        (request["partition_id"], request["global_rank"])
        for request in start_requests
    ] == [("part-0000", 0), ("part-0001", 1)]
    assert sorted(
        item["state"] for item in scheduler.db.list_partitions("run-e2e")
    ) == ["RUNNING", "SUCCEEDED"]

    transport.complete(second["id"])
    overview = scheduler.tick("run-e2e")
    assert overview["active_run"]["state"] == "SUCCEEDED"
    assert overview["active_run"]["progress_percent"] == 100.0
    assert scheduler.db.list_active_attempts("run-e2e") == []
    assert transport.pull_count == 2
    assert len(list(config.results_dir.rglob("_SUCCESS"))) == 2


def test_drained_node_finishes_current_attempt_but_gets_no_new_partition(tmp_path):
    config = _config(tmp_path)
    transport = FakeTransport(tmp_path / "fake-workers")
    scheduler = ClusterScheduler(config, transport=transport)
    scheduler.plan_run("run-drain", partition_count=2, split_state=False)
    scheduler.tick("run-drain")
    first = _current_attempt(scheduler, "run-drain")

    node = scheduler.db.set_node_drained("node-a", True)
    assert node["drained"] is True
    assert node["enabled"] is True
    probe = scheduler.probe_nodes({"node-a"})
    assert probe[0]["drained"] is True

    starts_before = [call[0] for call in transport.calls].count("start")
    transport.complete(first["id"])
    overview = scheduler.tick("run-drain")

    assert scheduler.db.list_active_attempts("run-drain") == []
    assert sorted(
        item["state"] for item in scheduler.db.list_partitions("run-drain")
    ) == ["QUEUED", "SUCCEEDED"]
    assert [call[0] for call in transport.calls].count("start") == starts_before
    assert overview["nodes"][0]["drained"] is True

    scheduler.db.set_node_drained("node-a", False)
    scheduler.tick("run-drain")
    assert _current_attempt(scheduler, "run-drain")["state"] == "RUNNING"
    assert [call[0] for call in transport.calls].count("start") == starts_before + 1


def test_scheduler_claim_is_exclusive_across_runs(tmp_path):
    config = _config(tmp_path)
    transport = FakeTransport(tmp_path / "fake-workers")
    scheduler = ClusterScheduler(config, transport=transport)
    scheduler.plan_run("run-a", partition_count=1, split_state=False)
    scheduler.plan_run("run-b", partition_count=1, split_state=False)

    scheduler.tick("run-a")
    active = _current_attempt(scheduler, "run-a")
    scheduler.tick("run-b")

    assert scheduler.db.active_attempt_for_node("node-a")["id"] == active["id"]
    assert scheduler.db.list_partitions("run-b")[0]["state"] == "QUEUED"
    assert [call[0] for call in transport.calls].count("start") == 1


def test_scheduler_rejects_stale_collected_fencing_token(tmp_path):
    config = _config(tmp_path)
    transport = FakeTransport(tmp_path / "fake-workers")
    scheduler = ClusterScheduler(config, transport=transport)
    scheduler.plan_run("run-fence", partition_count=1, split_state=False)
    scheduler.tick("run-fence")
    attempt = _current_attempt(scheduler, "run-fence")

    transport.complete(attempt["id"])
    transport.stale_result_tokens[attempt["id"]] = "f" * 32
    scheduler.tick("run-fence")

    rejected = scheduler.db.get_attempt(attempt["id"])
    assert rejected["state"] == "UNKNOWN"
    assert "invalid fencing_token" in rejected["error"]
    assert not (config.results_dir / "run-fence" / "partitions" / "part-0000").exists()

    del transport.stale_result_tokens[attempt["id"]]
    overview = scheduler.tick("run-fence")
    assert overview["active_run"]["state"] == "SUCCEEDED"
    result = json.loads(
        (
            config.results_dir
            / "run-fence"
            / "partitions"
            / "part-0000"
            / "result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["fencing_token"] == attempt["fencing_token"]


def test_run_cancel_stops_active_and_prevents_new_claims(tmp_path):
    config = _config(tmp_path)
    transport = FakeTransport(tmp_path / "fake-workers")
    scheduler = ClusterScheduler(config, transport=transport)
    scheduler.plan_run("run-cancel", partition_count=2, split_state=False)
    scheduler.tick("run-cancel")

    assert scheduler.cancel("run-cancel") == 2
    assert scheduler.db.get_run("run-cancel")["desired_state"] == "CANCELLED"
    assert scheduler.db.get_run("run-cancel")["state"] == "CANCELLED"
    assert scheduler.db.list_active_attempts("run-cancel") == []
    assert {item["state"] for item in scheduler.db.list_partitions("run-cancel")} == {
        "CANCELLED"
    }

    scheduler.tick("run-cancel")
    assert [call[0] for call in transport.calls].count("start") == 1


def test_resume_uses_persisted_source_and_attempt_remote_root(tmp_path):
    original = _config(tmp_path)
    transport = FakeTransport(tmp_path / "fake-workers")
    scheduler = ClusterScheduler(original, transport=transport)
    scheduler.plan_run("run-resume", partition_count=1, split_state=False)
    claimed = scheduler.db.claim_next(
        "run-resume",
        original.nodes[0].id,
        original.nodes[0].work_root,
        node_host=original.nodes[0].host,
        node_user=original.nodes[0].user,
        node_port=original.nodes[0].port,
        gpu_uuids=["GPU-fake0000"],
    )
    assert claimed is not None

    # Make the persisted paths observably different from the current YAML. This
    # models state restored after a controller/storage migration: resume must read
    # immutable run/attempt fields instead of recomputing paths from config.
    persisted_source = tmp_path / "persisted-dataset-root"
    shutil.copytree(original.source_root, persisted_source, copy_function=shutil.copy2)
    persisted_remote = "/srv/persisted-worker-root"
    with scheduler.db.connect() as connection:
        connection.execute(
            "UPDATE runs SET source_root=? WHERE id='run-resume'",
            (str(persisted_source),),
        )
        connection.execute(
            "UPDATE attempts SET remote_root=? WHERE id=?",
            (persisted_remote, claimed["attempt_id"]),
        )
    resumed = ClusterScheduler(original, transport=transport)

    resumed.tick("run-resume")
    attempt = _current_attempt(resumed, "run-resume")
    assert attempt["state"] == "RUNNING"
    assert transport.partition_sources == [persisted_source]
    assert transport.prepare_requests[-1]["work_root"] == persisted_remote

    transport.complete(attempt["id"])
    resumed.tick("run-resume")
    assert transport.pulled_remote_roots[-1].startswith(persisted_remote + "/runs/")


def test_resume_rejects_incompatible_execution_config_before_claim(tmp_path):
    original = _config(tmp_path)
    transport = FakeTransport(tmp_path / "fake-workers")
    scheduler = ClusterScheduler(original, transport=transport)
    scheduler.plan_run("run-config-fence", partition_count=1, split_state=False)
    persisted = scheduler.db.get_run("run-config-fence")
    assert len(persisted["execution_config_sha256"]) == 64

    # shm_size is part of docker execution but is not represented by the input
    # manifest. A restart must fail closed instead of silently changing it.
    changed = replace(original, shm_size="16g")
    resumed = ClusterScheduler(changed, transport=transport)

    with pytest.raises(SchedulerError, match="(?i)execution config|shm_size"):
        resumed.run_loop("run-config-fence", once=True)
    assert resumed.db.list_active_attempts("run-config-fence") == []
    assert not any(
        call[0] in {"prepare", "push_partition", "start"} for call in transport.calls
    )


def test_start_reconciles_completed_attempt_from_another_run(tmp_path):
    config = _config(tmp_path)
    transport = FakeTransport(tmp_path / "fake-workers")
    scheduler = ClusterScheduler(config, transport=transport)
    scheduler.plan_run("run-owner", partition_count=1, split_state=False)
    scheduler.plan_run("run-blocked", partition_count=1, split_state=False)
    scheduler.tick("run-owner")
    owner_attempt = _current_attempt(scheduler, "run-owner")
    transport.complete(owner_attempt["id"])

    # CLI `run start` targets run-blocked, but it must first globally reconcile
    # run-owner. Its completed attempt then releases the only node in this tick.
    overview = scheduler.run_loop("run-blocked", once=True)

    assert scheduler.db.get_run("run-owner")["state"] == "SUCCEEDED"
    assert scheduler.db.list_partitions("run-blocked")[0]["state"] == "RUNNING"
    assert overview["active_run"]["id"] == "run-blocked"
    assert [call[0] for call in transport.calls].count("start") == 2
    assert transport.pull_count == 1


def test_partition_cancel_survives_lost_rpc_response_and_restart(tmp_path):
    config = _config(tmp_path)
    transport = FakeTransport(tmp_path / "fake-workers")
    scheduler = ClusterScheduler(config, transport=transport)
    scheduler.plan_run("run-cancel-fence", partition_count=1, split_state=False)
    scheduler.tick("run-cancel-fence")
    attempt = _current_attempt(scheduler, "run-cancel-fence")
    transport.lost_cancel_responses.add(attempt["id"])

    with pytest.raises(SchedulerError, match="cancel response was lost"):
        scheduler.cancel("run-cancel-fence", attempt["partition_id"])
    assert transport.jobs[attempt["id"]]["state"] == "CANCELLED"

    # The cancel intent must already be durable. A fresh scheduler reconciles the
    # remote terminal state and must never turn this partition back into QUEUED.
    restarted = ClusterScheduler(config, transport=transport)
    overview = restarted.run_loop("run-cancel-fence", once=True)
    partition = restarted.db.get_partition("run-cancel-fence", attempt["partition_id"])
    assert partition["state"] == "CANCELLED"
    assert overview["active_run"]["state"] == "CANCELLED"
    assert restarted.db.list_active_attempts("run-cancel-fence") == []
    assert [call[0] for call in transport.calls].count("start") == 1


def test_cancel_uses_saved_endpoint_after_non_endpoint_config_change(tmp_path):
    config = _config(tmp_path)
    transport = FakeTransport(tmp_path / "fake-workers")
    scheduler = ClusterScheduler(config, transport=transport)
    scheduler.plan_run("run-cancel-config", partition_count=1, split_state=False)
    scheduler.tick("run-cancel-config")

    changed = replace(config, shm_size="16g")
    restarted = ClusterScheduler(changed, transport=transport)

    assert restarted.cancel("run-cancel-config") == 1
    assert restarted.db.get_run("run-cancel-config")["state"] == "CANCELLED"
    assert transport.jobs[next(iter(transport.jobs))]["state"] == "CANCELLED"


def test_cancel_intent_survives_endpoint_mismatch_until_config_is_restored(tmp_path):
    config = _config(tmp_path)
    transport = FakeTransport(tmp_path / "fake-workers")
    scheduler = ClusterScheduler(config, transport=transport)
    scheduler.plan_run("run-cancel-endpoint", partition_count=1, split_state=False)
    scheduler.tick("run-cancel-endpoint")
    attempt = _current_attempt(scheduler, "run-cancel-endpoint")

    changed_node = replace(config.nodes[0], host="replacement.example.test")
    changed = replace(config, nodes=(changed_node,))
    restarted = ClusterScheduler(changed, transport=transport)

    with pytest.raises(SchedulerError, match="saved SSH endpoint"):
        restarted.cancel("run-cancel-endpoint", attempt["partition_id"])
    partition = restarted.db.get_partition(
        "run-cancel-endpoint", attempt["partition_id"]
    )
    assert partition["desired_state"] == "CANCELLED"
    assert transport.jobs[attempt["id"]]["state"] == "RUNNING"

    restored = ClusterScheduler(config, transport=transport)
    overview = restored.run_loop("run-cancel-endpoint", once=True)
    assert overview["active_run"]["state"] == "CANCELLED"
    assert transport.jobs[attempt["id"]]["state"] == "CANCELLED"


@pytest.mark.parametrize(
    "probe_override",
    [
        {"ok": False, "docker_ok": False},
        {"ok": False, "gpu_ok": False},
        {"runner_schema": SCHEMA_VERSION + 1},
        {"gpu_devices": [1], "gpu_uuids": ["GPU-fake0001"]},
        {"gpu_uuids": []},
    ],
    ids=(
        "docker-unavailable",
        "gpu-unavailable",
        "runner-schema-mismatch",
        "gpu-index-mismatch",
        "gpu-uuid-missing",
    ),
)
def test_unusable_probe_never_claims_or_stages_partition(tmp_path, probe_override):
    config = _config(tmp_path)
    transport = FakeTransport(tmp_path / "fake-workers")
    transport.probe_overrides = probe_override
    scheduler = ClusterScheduler(config, transport=transport)
    scheduler.plan_run("run-bad-probe", partition_count=1, split_state=False)

    scheduler.tick("run-bad-probe")

    assert scheduler.db.list_partitions("run-bad-probe")[0]["state"] == "QUEUED"
    assert scheduler.db.list_active_attempts("run-bad-probe") == []
    assert scheduler.db.list_nodes()[0]["state"] == "DEGRADED"
    assert not any(
        call[0] in {"prepare", "push_partition", "start"} for call in transport.calls
    )


def test_runtime_config_defaults_keep_legacy_docker_gpu0_behavior(tmp_path):
    config = _loaded_config(tmp_path, {})
    node = config.nodes[0]

    assert node.runtime == "docker"
    assert node.gpu_devices == (0,)
    assert node.pipeline_root == "/opt/balalaika/app"
    assert node.venv_path == "/opt/balalaika/.venv"


def test_runtime_config_loads_direct_mode_and_ordered_gpu_namespace(tmp_path):
    config = _loaded_config(
        tmp_path,
        {
            "runtime": "direct",
            "gpu_devices": [2, 0],
            "pipeline_root": "/workspace/balalaika",
            "venv_path": "/workspace/venv",
            "models_root": "/workspace/models",
        },
    )
    node = config.nodes[0]

    assert node.runtime == "direct"
    assert node.gpu_devices == (2, 0)
    assert node.pipeline_root == "/workspace/balalaika"
    assert node.venv_path == "/workspace/venv"
    assert node.models_root == "/workspace/models"


@pytest.mark.parametrize(
    "gpu_devices",
    ([], [0, 0], [-1], [True], ["0"], [0.5], 0),
    ids=("empty", "duplicate", "negative", "bool", "string", "float", "not-list"),
)
def test_runtime_config_rejects_invalid_gpu_devices(tmp_path, gpu_devices):
    with pytest.raises(ValueError, match="gpu_devices"):
        _loaded_config(tmp_path, {"gpu_devices": gpu_devices})


def test_runtime_config_rejects_unknown_mode_and_unsafe_paths(tmp_path):
    with pytest.raises(ValueError, match="runtime"):
        _loaded_config(tmp_path, {"runtime": "nested-docker"})
    with pytest.raises(ValueError, match="pipeline_root"):
        _loaded_config(tmp_path, {"pipeline_root": "/app;touch"})
    with pytest.raises(ValueError, match="venv_path"):
        _loaded_config(tmp_path, {"venv_path": "/venv$(id)"})


def test_transport_probe_forwards_runtime_gpu_and_runtime_paths(monkeypatch, tmp_path):
    node = replace(
        _node(),
        runtime="direct",
        gpu_devices=(2, 0),
        pipeline_root="/workspace/balalaika",
        venv_path="/workspace/venv",
        models_root="/workspace/models",
    )
    transport = ClusterTransport(_config(tmp_path, nodes=(node,)))
    captured: dict[str, Any] = {}

    def fake_rpc(actual_node, request, **kwargs):
        captured.update(request)
        return {"ok": True}

    monkeypatch.setattr(transport, "rpc", fake_rpc)
    transport.probe(node)

    assert captured == {
        "operation": "probe",
        "work_root": node.work_root,
        "runtime": "direct",
        "gpu_devices": [2, 0],
        "image": node.image or transport.config.image,
        "pipeline_root": "/workspace/balalaika",
        "venv_path": "/workspace/venv",
        "models_root": "/workspace/models",
        "cache_root": node.cache_root,
    }


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("runtime", "direct"),
        ("gpu_devices", (0, 1)),
        ("pipeline_root", "/srv/other-pipeline"),
        ("venv_path", "/srv/other-venv"),
        ("models_root", "/srv/other-models"),
    ],
)
def test_runtime_and_paths_are_part_of_execution_snapshot(
    tmp_path, field, changed_value
):
    config = _config(tmp_path)
    scheduler = ClusterScheduler(config, transport=FakeTransport(tmp_path / "remote"))
    scheduler.plan_run("run-runtime-snapshot", partition_count=1, split_state=False)
    run = scheduler.db.get_run("run-runtime-snapshot")
    changed_node = replace(config.nodes[0], **{field: changed_value})
    changed = replace(config, nodes=(changed_node,))

    with pytest.raises(SchedulerError, match="execution config changed"):
        ClusterScheduler(changed)._assert_execution_config(run)


def test_execution_snapshot_v2_preserves_legacy_docker_resume(tmp_path):
    config = _config(tmp_path)
    scheduler = ClusterScheduler(config, transport=FakeTransport(tmp_path / "remote"))
    scheduler.plan_run("run-snapshot-v2", partition_count=1, split_state=False)
    run = scheduler.db.get_run("run-snapshot-v2")

    assert run["execution_config_version"] == 2
    snapshot = json.loads(run["execution_config_json"])
    assert snapshot["schema_version"] == 2
    assert snapshot["nodes"][0]["runtime"] == "docker"
    assert snapshot["nodes"][0]["gpu_devices"] == [0]
    scheduler._assert_execution_config(run)

    legacy = dict(run)
    legacy["execution_config_version"] = None
    legacy["execution_config_sha256"] = scheduler._hash_execution_payload(
        scheduler._legacy_execution_config_payload()
    )
    scheduler._assert_execution_config(legacy)

    direct_node = replace(config.nodes[0], runtime="direct")
    changed = replace(config, nodes=(direct_node,))
    with pytest.raises(SchedulerError, match="legacy Docker/GPU-0"):
        ClusterScheduler(changed)._assert_execution_config(legacy)


def test_direct_scheduler_uses_all_gpus_and_forwards_snapshot_paths(tmp_path):
    node = replace(
        _node(),
        runtime="direct",
        gpu_devices=(2, 0),
        pipeline_root="/workspace/balalaika",
        venv_path="/workspace/venv",
        models_root="/workspace/models",
    )
    config = _config(tmp_path, nodes=(node,))
    transport = FakeTransport(tmp_path / "fake-workers")
    transport.probe_overrides = {
        "ok": True,
        "runtime": "direct",
        "docker_ok": False,
        "image_id": None,
        "direct_ok": True,
        "gpu_ok": True,
        "gpu_devices": [2, 0],
        "gpu_uuids": ["GPU-direct02", "GPU-direct00"],
        "gpus": [
            {"index": 2, "memory_used_mib": 0, "utilization_percent": 0},
            {"index": 0, "memory_used_mib": 0, "utilization_percent": 0},
        ],
        "pipeline_ok": True,
        "renderer_ok": True,
        "venv_ok": True,
        "models_ok": True,
    }
    scheduler = ClusterScheduler(config, transport=transport)
    scheduler.plan_run("run-direct", partition_count=2, split_state=False)

    scheduler.tick("run-direct")

    start = next(
        request
        for request in transport.rpc_requests
        if request["operation"] == "start"
    )
    assert start["runtime"] == "direct"
    assert start["gpu_devices"] == [2, 0]
    assert start["gpu_uuids"] == ["GPU-direct02", "GPU-direct00"]
    assert start["pipeline_root"] == "/workspace/balalaika"
    assert start["venv_path"] == "/workspace/venv"
    assert start["models_root"] == "/workspace/models"
    assert [call[0] for call in transport.calls].count("start") == 1
    assert sorted(
        partition["state"]
        for partition in scheduler.db.list_partitions("run-direct")
    ) == ["QUEUED", "RUNNING"]


def test_direct_probe_does_not_require_docker_and_checks_configured_gpus(
    monkeypatch, tmp_path
):
    pipeline_root, venv_path, models_root, cache_root = _direct_runtime_paths(
        tmp_path,
        "#!/usr/bin/env bash\nexit 0\n",
    )
    work_root = tmp_path / "probe-work"
    work_root.mkdir()
    seen: list[list[str]] = []
    original = node_runner.run_command

    def fake_run_command(argv, **kwargs):
        seen.append(argv)
        if Path(argv[0]).name == "docker":
            raise AssertionError("direct probe called Docker CLI")
        if Path(argv[0]).name == "nvidia-smi":
            index = int(next(value for value in argv if value.startswith("--id="))[5:])
            name = "GPU two" if index == 2 else "GPU zero"
            uuid = "GPU-direct02" if index == 2 else "GPU-direct00"
            used = 10 if index == 2 else 20
            output = f"{index}, {name}, {uuid}, 24000, {used}, 0\n"
            return subprocess.CompletedProcess(argv, 0, stdout=output, stderr="")
        if Path(argv[0]) == venv_path / "bin" / "python":
            output = json.dumps({"torch": "2.8.0", "cuda": "12.8"})
            return subprocess.CompletedProcess(argv, 0, stdout=output, stderr="")
        return original(argv, **kwargs)

    monkeypatch.setattr(node_runner, "run_command", fake_run_command)
    response = node_runner.operation_probe(
        {
            "operation": "probe",
            "runtime": "direct",
            "gpu_devices": [2, 0],
            "work_root": str(work_root),
            "image": "ignored-in-direct:latest",
            "pipeline_root": str(pipeline_root),
            "venv_path": str(venv_path),
            "models_root": str(models_root),
            "cache_root": str(cache_root),
        }
    )

    assert response["ok"] is True
    assert response["gpu_ok"] is True
    assert response["gpu_devices"] == [2, 0]
    assert response["gpu_uuids"] == ["GPU-direct02", "GPU-direct00"]
    assert [gpu["index"] for gpu in response["gpus"]] == [2, 0]
    python_probe = next(argv for argv in seen if Path(argv[0]) == venv_path / "bin" / "python")
    assert "get_device_properties" not in python_probe[2]
    assert "mem_get_info" not in python_probe[2]
    assert not any(Path(argv[0]).name == "docker" for argv in seen)


def test_docker_start_uses_fenced_gpu_uuids_and_child_logical_ids(
    monkeypatch, tmp_path
):
    request, prepared = _ready_runner_attempt(tmp_path)
    models_root = tmp_path / "docker-models"
    models_root.mkdir()
    cache_root = tmp_path / "docker-cache"
    cache_root.mkdir()
    commands: list[list[str]] = []

    monkeypatch.setattr(node_runner, "docker_inspect", lambda name: None)
    monkeypatch.setattr(
        node_runner,
        "_docker_gpu_probe",
        lambda devices: (
            [
                {
                    "index": 2,
                    "uuid": "GPU-device02",
                    "memory_used_mib": 0,
                    "utilization_percent": 0,
                },
                {
                    "index": 0,
                    "uuid": "GPU-device00",
                    "memory_used_mib": 0,
                    "utilization_percent": 0,
                },
            ],
            None,
        ),
    )

    def fake_run_command(argv, **kwargs):
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="container-id\n", stderr="")

    monkeypatch.setattr(node_runner, "run_command", fake_run_command)
    response = node_runner.operation_start(
        request
        | {
            "operation": "start",
            "runtime": "docker",
            "gpu_devices": [2, 0],
            "gpu_uuids": ["GPU-device02", "GPU-device00"],
            "image": "balalaika:test",
            "pipeline_root": "/opt/balalaika/app",
            "venv_path": "/opt/balalaika/.venv",
            "models_root": str(models_root),
            "cache_root": str(cache_root),
            "env_file": None,
            "stage_start": "1",
            "stage_stop": "2",
            "shm_size": "8g",
        }
    )

    assert response["state"] == "RUNNING"
    docker_argv = next(
        argv for argv in commands if argv[:2] == ["/usr/bin/docker", "run"]
    )
    assert docker_argv[docker_argv.index("--gpus") + 1] == (
        '\"device=GPU-device02,GPU-device00\"'
    )
    assert "CUDA_VISIBLE_DEVICES=0,1" in docker_argv
    assert "BALALAIKA_PARTITION_ID=part-0000" in docker_argv
    assert "BALALAIKA_GLOBAL_RANK=0" in docker_argv
    metadata = json.loads(
        Path(prepared["control_root"], "job.json").read_text(encoding="utf-8")
    )
    assert metadata["runtime"] == "docker"
    assert metadata["gpu_devices"] == [2, 0]
    assert metadata["pipeline_root"] == "/opt/balalaika/app"
    assert metadata["venv_path"] == "/opt/balalaika/.venv"
    assert metadata["models_root"] == str(models_root)


def test_start_rejects_gpu_uuid_remap_after_probe(monkeypatch, tmp_path):
    request, _ = _ready_runner_attempt(tmp_path)
    models_root = tmp_path / "remap-models"
    models_root.mkdir()
    cache_root = tmp_path / "remap-cache"
    cache_root.mkdir()
    monkeypatch.setattr(
        node_runner,
        "_docker_gpu_probe",
        lambda devices: ([{"index": 2, "uuid": "GPU-replaced2"}], None),
    )

    with pytest.raises(node_runner.RunnerError, match="GPU mapping changed"):
        node_runner.operation_start(
            request
            | {
                "operation": "start",
                "runtime": "docker",
                "gpu_devices": [2],
                "gpu_uuids": ["GPU-original2"],
                "image": "balalaika:test",
                "pipeline_root": "/opt/balalaika/app",
                "venv_path": "/opt/balalaika/.venv",
                "models_root": str(models_root),
                "cache_root": str(cache_root),
                "env_file": None,
                "stage_start": "1",
                "stage_stop": "2",
                "shm_size": "8g",
            }
        )


def test_direct_fast_completion_keeps_startup_handshake_and_never_calls_docker(
    monkeypatch, tmp_path
):
    request, prepared = _ready_runner_attempt(tmp_path)
    script = """#!/usr/bin/env bash
set -euo pipefail
{
    printf '%s\\n' "$CUDA_VISIBLE_DEVICES"
    printf '%s\\n' "$VIRTUAL_ENV"
    printf '%s\\n' "$BALALAIKA_MODELS_ROOT"
    printf '%s\\n' "$BALALAIKA_DATA_ROOT"
    printf '%s\\n' "$BALALAIKA_PARTITION_ID"
    printf '%s\\n' "$BALALAIKA_GLOBAL_RANK"
} > "$BALALAIKA_OUTPUT_ROOT/direct-env.txt"
"""
    pipeline_root, venv_path, models_root, cache_root = _direct_runtime_paths(
        tmp_path, script
    )
    _deny_docker_calls(monkeypatch)
    monkeypatch.setattr(
        node_runner,
        "_direct_gpu_probe",
        lambda venv, devices: (
            [
                {
                    "index": 2,
                    "uuid": "GPU-direct02",
                    "memory_used_mib": 0,
                    "utilization_percent": 0,
                },
                {
                    "index": 0,
                    "uuid": "GPU-direct00",
                    "memory_used_mib": 0,
                    "utilization_percent": 0,
                },
            ],
            None,
        ),
    )
    start_request = request | {
        "operation": "start",
        "runtime": "direct",
        "gpu_devices": [2, 0],
        "gpu_uuids": ["GPU-direct02", "GPU-direct00"],
        "image": "ignored-in-direct:latest",
        "pipeline_root": str(pipeline_root),
        "venv_path": str(venv_path),
        "models_root": str(models_root),
        "cache_root": str(cache_root),
        "env_file": None,
        "stage_start": "1",
        "stage_stop": "2",
        "shm_size": "8g",
    }

    started = node_runner.operation_start(start_request)
    assert started["state"] in {"RUNNING", "COMPLETED"}
    process_record = json.loads(
        Path(prepared["control_root"], "direct-process.json").read_text(
            encoding="utf-8"
        )
    )
    assert process_record["fencing_token"] == request["fencing_token"]
    assert len(process_record["spec_sha256"]) == 64
    assert isinstance(process_record["supervisor_pid"], int)
    assert isinstance(process_record["pipeline_pid"], int)
    assert process_record["pipeline_pgid"] == process_record["pipeline_pid"]
    status_request = request | {"operation": "status"}
    deadline = time.monotonic() + 5
    while True:
        status = node_runner.operation_status(status_request)
        if status["state"] != "RUNNING" or time.monotonic() >= deadline:
            break
        time.sleep(0.05)

    assert status["state"] == "COMPLETED"
    assert status["exit_code"] == 0
    attempt_root = Path(prepared["attempt_root"])
    values = attempt_root.joinpath("output", "direct-env.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    assert values == [
        "GPU-direct02,GPU-direct00",
        str(venv_path),
        str(models_root),
        str(attempt_root / "data"),
        "part-0000",
        "0",
    ]
    assert attempt_root.joinpath("_SUCCESS").is_file()
    metadata = json.loads(
        Path(prepared["control_root"], "job.json").read_text(encoding="utf-8")
    )
    assert metadata["runtime"] == "direct"
    assert metadata["gpu_devices"] == [2, 0]
    assert metadata["pipeline_root"] == str(pipeline_root)
    assert metadata["venv_path"] == str(venv_path)
    assert metadata["models_root"] == str(models_root)


def test_direct_cancel_stops_entire_process_group_and_is_durable(
    monkeypatch, tmp_path
):
    request, prepared = _ready_runner_attempt(tmp_path)
    script = """#!/usr/bin/env bash
set -u
trap 'exit 130' INT
trap 'exit 143' TERM

(
    trap 'exit 130' INT
    trap 'exit 143' TERM
    sleep 300 &
    leaf=$!
    printf '%s\\n' "$BASHPID" "$leaf" > "$BALALAIKA_OUTPUT_ROOT/branch-pids.txt"
    wait "$leaf"
) &
branch=$!

while [[ ! -s "$BALALAIKA_OUTPUT_ROOT/branch-pids.txt" ]]; do
    sleep 0.01
done
{
    printf '%s\\n' "$$" "$branch"
    cat "$BALALAIKA_OUTPUT_ROOT/branch-pids.txt"
} > "$BALALAIKA_OUTPUT_ROOT/process-pids.txt"
wait "$branch"
"""
    pipeline_root, venv_path, models_root, cache_root = _direct_runtime_paths(
        tmp_path, script
    )
    _deny_docker_calls(monkeypatch)
    monkeypatch.setattr(
        node_runner,
        "_direct_gpu_probe",
        lambda venv, devices: (
            [
                {
                    "index": 0,
                    "uuid": "GPU-cancel00",
                    "memory_used_mib": 0,
                    "utilization_percent": 0,
                }
            ],
            None,
        ),
    )
    start_request = request | {
        "operation": "start",
        "runtime": "direct",
        "gpu_devices": [0],
        "gpu_uuids": ["GPU-cancel00"],
        "image": "ignored-in-direct:latest",
        "pipeline_root": str(pipeline_root),
        "venv_path": str(venv_path),
        "models_root": str(models_root),
        "cache_root": str(cache_root),
        "env_file": None,
        "stage_start": "1",
        "stage_stop": "2",
        "shm_size": "8g",
    }
    started = node_runner.operation_start(start_request)
    assert started["state"] == "RUNNING"

    attempt_root = Path(prepared["attempt_root"])
    pid_path = attempt_root / "output" / "process-pids.txt"
    deadline = time.monotonic() + 5
    while not pid_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert pid_path.is_file()
    pids = [int(value) for value in pid_path.read_text("ascii").splitlines()]
    assert len(pids) == 4
    metadata = json.loads(
        Path(prepared["control_root"], "job.json").read_text(encoding="utf-8")
    )
    pgid = metadata["pipeline_pgid"]
    assert pids[0] == metadata["pipeline_pid"] == pgid

    def group_exists() -> bool:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        return True

    try:
        cancelled = node_runner.operation_cancel(request | {"operation": "cancel"})
        assert cancelled["state"] == "CANCELLED"

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not group_exists() and all(
                node_runner._process_start_ticks(pid) is None for pid in pids
            ):
                break
            time.sleep(0.05)
        group_still_exists = group_exists()
        live_pids = [
            pid
            for pid in pids
            if node_runner._process_start_ticks(pid) is not None
        ]

        first = node_runner.operation_status(request | {"operation": "status"})
        time.sleep(0.05)
        second = node_runner.operation_status(request | {"operation": "status"})
        assert first["state"] == second["state"] == "CANCELLED"
        persisted = json.loads(
            Path(prepared["control_root"], "job.json").read_text(encoding="utf-8")
        )
        assert persisted["state"] == "CANCELLED"
        assert not group_still_exists, (
            f"direct process group {pgid} remains after cancellation; "
            f"live recorded PIDs: {live_pids}"
        )
        assert not live_pids
    finally:
        if group_exists():
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
