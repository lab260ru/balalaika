from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from cluster_admin.config import ClusterConfig, NodeConfig
from cluster_admin.node_runner import SCHEMA_VERSION
from cluster_admin.scheduler import ClusterScheduler, SchedulerError
from cluster_admin.transport import TransportError


class FakeTransport:
    """In-process worker protocol used without SSH, rsync, or Docker."""

    def __init__(self, root: Path):
        self.root = root
        self.calls: list[tuple[str, str, str | None]] = []
        self.jobs: dict[str, dict[str, Any]] = {}
        self.pull_count = 0
        self.prepare_requests: list[dict[str, Any]] = []
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
            "docker_ok": True,
            "gpu_ok": True,
            "image_id": "sha256:" + "a" * 64,
            "models_ok": True,
            "runner_schema": SCHEMA_VERSION,
            "gpu": {"index": 0},
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


def _current_attempt(scheduler: ClusterScheduler, run_id: str) -> dict[str, Any]:
    attempts = scheduler.db.list_active_attempts(run_id)
    assert len(attempts) == 1
    return attempts[0]


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
        "run-resume", original.nodes[0].id, original.nodes[0].work_root
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
    ],
    ids=("docker-unavailable", "gpu0-unavailable", "runner-schema-mismatch"),
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
