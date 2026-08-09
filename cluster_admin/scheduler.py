"""Small-cluster scheduler for SSH-accessible Balalaika Docker nodes."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from .config import ClusterConfig, NodeConfig
from .db import StateDB
from .node_runner import SCHEMA_VERSION as RUNNER_SCHEMA_VERSION
from .partitioner import build_partition_plan, validate_source_manifest
from .transport import ClusterTransport, TransportError


class SchedulerError(RuntimeError):
    """The controller cannot safely advance a run."""


@contextmanager
def controller_lock(state_dir: Path, *, blocking: bool = False) -> Iterator[None]:
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "controller.lock"
    with lock_path.open("a+b") as handle:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError as exc:
            raise SchedulerError(
                f"Another scheduler owns {lock_path}; use the existing controller"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n".encode("ascii"))
        handle.flush()
        yield


class ClusterScheduler:
    def __init__(
        self,
        config: ClusterConfig,
        *,
        database: StateDB | None = None,
        transport: ClusterTransport | None = None,
    ):
        self.config = config
        self.db = database or StateDB(config.db_path)
        self.transport = transport or ClusterTransport(config)

    def initialize(self) -> None:
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.config.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        self.db.initialize()
        self.db.sync_nodes(self.config.nodes)

    def _execution_config_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "source_root": str(self.config.source_root),
            "image": self.config.image,
            "stage_start": self.config.stage_start,
            "stage_stop": self.config.stage_stop,
            "shm_size": self.config.shm_size,
            "nodes": [
                {
                    "id": node.id,
                    "host": node.host,
                    "user": node.user,
                    "port": node.port,
                    "work_root": node.work_root,
                    "models_root": node.models_root,
                    "cache_root": node.cache_root,
                    "image": node.image,
                    "env_file": node.env_file,
                    "enabled": node.enabled,
                    "runtime": node.runtime,
                    "gpu_devices": list(node.gpu_devices),
                    "pipeline_root": node.pipeline_root,
                    "venv_path": node.venv_path,
                    "max_gpu_memory_used_mib": node.max_gpu_memory_used_mib,
                    "max_gpu_utilization_percent": (
                        node.max_gpu_utilization_percent
                    ),
                }
                for node in sorted(self.config.nodes, key=lambda item: item.id)
            ],
        }

    def _legacy_execution_config_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source_root": str(self.config.source_root),
            "image": self.config.image,
            "stage_start": self.config.stage_start,
            "stage_stop": self.config.stage_stop,
            "shm_size": self.config.shm_size,
            "nodes": [
                {
                    "id": node.id,
                    "host": node.host,
                    "user": node.user,
                    "port": node.port,
                    "work_root": node.work_root,
                    "models_root": node.models_root,
                    "cache_root": node.cache_root,
                    "image": node.image,
                    "env_file": node.env_file,
                    "enabled": node.enabled,
                }
                for node in sorted(self.config.nodes, key=lambda item: item.id)
            ],
        }

    @staticmethod
    def _hash_execution_payload(payload: dict[str, Any]) -> str:
        encoded = json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def _execution_config_sha256(self) -> str:
        return self._hash_execution_payload(self._execution_config_payload())

    def _assert_execution_config(self, run: dict[str, Any]) -> None:
        expected = run.get("execution_config_sha256")
        if not expected:
            raise SchedulerError(
                f"Run {run['id']} predates the execution-config snapshot; "
                "create a new run plan"
            )
        version = run.get("execution_config_version")
        if version is None:
            legacy_compatible = all(
                node.runtime == "docker"
                and node.gpu_devices == (0,)
                and node.pipeline_root == "/opt/balalaika/app"
                and node.venv_path == "/opt/balalaika/.venv"
                for node in self.config.nodes
            )
            if not legacy_compatible:
                raise SchedulerError(
                    f"Run {run['id']} uses the legacy Docker/GPU-0 execution snapshot; "
                    "restore legacy node runtime settings or create a new run"
                )
            actual = self._hash_execution_payload(
                self._legacy_execution_config_payload()
            )
        elif version == 2:
            actual = self._execution_config_sha256()
        else:
            raise SchedulerError(
                f"Run {run['id']} has unsupported execution-config version {version!r}"
            )
        if expected != actual:
            raise SchedulerError(
                f"Cluster execution config changed after run {run['id']} was planned; "
                "restore the original config or create a new run"
            )

    def plan_run(
        self,
        run_id: str,
        *,
        partition_count: int | None = None,
        split_state: bool = True,
    ) -> dict[str, Any]:
        self.initialize()
        if self.db.get_run(run_id):
            raise SchedulerError(f"Run already exists: {run_id}")
        enabled = [node for node in self.config.nodes if node.enabled]
        if not enabled:
            raise SchedulerError("No enabled nodes are configured")
        count = (
            partition_count
            if partition_count is not None
            else len(enabled) * self.config.partitions_per_node
        )
        plan = build_partition_plan(
            self.config,
            run_id,
            partition_count=count,
            split_state=split_state,
        )
        execution_payload = self._execution_config_payload()
        plan.run["execution_config_version"] = 2
        plan.run["execution_config_json"] = json.dumps(
            execution_payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        plan.run["execution_config_sha256"] = self._hash_execution_payload(
            execution_payload
        )
        self.db.create_run(plan.run, plan.partitions)
        return {
            "run": plan.run,
            "partitions": plan.partitions,
            "run_manifest_path": str(plan.run_manifest_path),
        }

    def bootstrap_nodes(self, node_ids: set[str] | None = None) -> list[dict[str, Any]]:
        self.initialize()
        results = []
        for node in self.config.nodes:
            if node_ids and node.id not in node_ids:
                continue
            try:
                result = self.transport.bootstrap(node)
                results.append({"node_id": node.id, **result})
            except TransportError as exc:
                self.db.update_node(node.id, "OFFLINE", error=str(exc))
                results.append({"node_id": node.id, "ok": False, "error": str(exc)})
        return results

    def probe_nodes(self, node_ids: set[str] | None = None) -> list[dict[str, Any]]:
        self.initialize()
        results = []
        for node in self.config.nodes:
            if node_ids and node.id not in node_ids:
                continue
            if not node.enabled:
                continue
            persisted = self.db.get_node(node.id)
            drained = bool(persisted and persisted.get("drained"))
            try:
                result = self.transport.probe(node)
                error = self._probe_error(node, result)
                state = "ONLINE" if error is None else "DEGRADED"
                self.db.update_node(node.id, state, result, error)
                results.append(
                    {
                        **result,
                        "node_id": node.id,
                        "ok": state == "ONLINE",
                        "error": error,
                        "drained": drained,
                    }
                )
            except TransportError as exc:
                self.db.update_node(node.id, "OFFLINE", error=str(exc))
                results.append(
                    {
                        "node_id": node.id,
                        "ok": False,
                        "error": str(exc),
                        "drained": drained,
                    }
                )
        return results

    def _probe_error(self, node: NodeConfig, result: dict[str, Any]) -> str | None:
        if result.get("runner_schema") != RUNNER_SCHEMA_VERSION:
            return (
                "Node runner schema mismatch: expected "
                f"{RUNNER_SCHEMA_VERSION}, got {result.get('runner_schema')!r}; "
                "run nodes bootstrap"
            )
        if result.get("runtime") != node.runtime:
            return (
                f"Worker runtime mismatch: expected {node.runtime!r}, "
                f"got {result.get('runtime')!r}"
            )
        if tuple(result.get("gpu_devices") or ()) != node.gpu_devices:
            return (
                f"Worker GPU mapping mismatch: expected {list(node.gpu_devices)}, "
                f"got {result.get('gpu_devices')!r}"
            )
        gpu_uuids = result.get("gpu_uuids")
        if (
            not isinstance(gpu_uuids, list)
            or len(gpu_uuids) != len(node.gpu_devices)
            or any(not isinstance(value, str) or not value for value in gpu_uuids)
        ):
            return "Worker did not resolve every configured GPU to a stable UUID"
        if result.get("gpu_ok") is not True:
            return (
                f"Configured GPUs {list(node.gpu_devices)} are unavailable: "
                f"{result.get('gpu_error') or 'unknown error'}"
            )
        gpu_details = result.get("gpus")
        if not isinstance(gpu_details, list) or len(gpu_details) != len(
            node.gpu_devices
        ):
            return "Worker did not return occupancy for every configured GPU"
        for detail in gpu_details:
            if not isinstance(detail, dict):
                return "Worker returned invalid GPU occupancy details"
            used = detail.get("memory_used_mib")
            utilization = detail.get("utilization_percent")
            if not isinstance(used, int) or not isinstance(utilization, int):
                return "Worker returned incomplete GPU occupancy metrics"
            if used > node.max_gpu_memory_used_mib:
                return (
                    f"GPU {detail.get('index')} is busy: {used} MiB used exceeds "
                    f"limit {node.max_gpu_memory_used_mib} MiB"
                )
            if utilization > node.max_gpu_utilization_percent:
                return (
                    f"GPU {detail.get('index')} is busy: {utilization}% utilization "
                    f"exceeds limit {node.max_gpu_utilization_percent}%"
                )
        if result.get("ok") is not True:
            return "Node probe did not report a healthy worker"
        if result.get("models_ok") is not True:
            return f"Models directory is empty or missing: {node.models_root}"
        if node.runtime == "direct":
            if result.get("pipeline_ok") is not True:
                return f"Pipeline is missing or invalid: {node.pipeline_root}/base.sh"
            if result.get("renderer_ok") is not True:
                return (
                    "Direct config renderer is missing on the node; "
                    "run nodes bootstrap"
                )
            if result.get("venv_ok") is not True:
                return f"Virtual environment is missing or invalid: {node.venv_path}"
        else:
            if result.get("docker_ok") is not True:
                return (
                    "Docker is unavailable: "
                    f"{result.get('docker_error') or 'unknown error'}"
                )
            if not result.get("image_id"):
                return f"Docker image is not present: {node.image or self.config.image}"
        return None

    @staticmethod
    def _attempt_request(attempt: dict[str, Any]) -> dict[str, Any]:
        return {
            "work_root": attempt["remote_root"],
            "run_id": attempt["run_id"],
            "partition_id": attempt["partition_id"],
            "attempt_id": attempt["id"],
            "attempt_ordinal": int(attempt["ordinal"]),
            "fencing_token": attempt["fencing_token"],
        }

    @staticmethod
    def _attempt_root(attempt: dict[str, Any]) -> str:
        name = f"attempt-{int(attempt['ordinal']):03d}-{attempt['id'][:12]}"
        return str(
            PurePosixPath(attempt["remote_root"])
            / "runs"
            / attempt["run_id"]
            / "partitions"
            / attempt["partition_id"]
            / name
        )

    def _cancel_node_for_attempt(self, attempt: dict[str, Any]) -> NodeConfig:
        try:
            node = self.config.node(attempt["node_id"])
        except KeyError as exc:
            raise SchedulerError(
                f"Cannot cancel attempt {attempt['id']}: node {attempt['node_id']} "
                "is absent from the current config; restore its original endpoint"
            ) from exc
        expected = (
            attempt.get("node_host"),
            attempt.get("node_user"),
            attempt.get("node_port"),
            attempt.get("remote_root"),
        )
        actual = (node.host, node.user, node.port, node.work_root)
        if None in expected or expected != actual:
            raise SchedulerError(
                f"Cannot safely cancel attempt {attempt['id']}: its saved SSH endpoint "
                "or work_root differs from the current config; restore the original "
                "node endpoint and retry cancel"
            )
        return node

    def _run_and_partition(
        self, attempt: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        run = self.db.get_run(attempt["run_id"])
        partition = self.db.get_partition(attempt["run_id"], attempt["partition_id"])
        if run is None or partition is None:
            raise SchedulerError(f"Attempt {attempt['id']} references missing state")
        if partition.get("current_attempt_id") != attempt["id"]:
            raise SchedulerError(f"Attempt {attempt['id']} is no longer current")
        return run, partition

    def _stage_and_start(self, node: NodeConfig, attempt: dict[str, Any]) -> None:
        run, partition = self._run_and_partition(attempt)
        manifest_path = Path(partition["manifest_path"])
        files_list_path = Path(partition["files_list_path"])
        config_path = Path(run["config_path"])
        source_root = Path(run["source_root"])
        validate_source_manifest(source_root, manifest_path)
        common = self._attempt_request(attempt)

        self.db.update_attempt(attempt["id"], "STAGING")
        prepared = self.transport.rpc(
            node,
            {
                **common,
                "operation": "prepare",
                "manifest_sha256": partition["manifest_sha256"],
                "config_sha256": run["config_sha256"],
            },
        )
        if prepared.get("state") in {"READY", "STARTING", "RUNNING", "COMPLETED"}:
            self.db.update_attempt(attempt["id"], "READY")
            self._start(node, attempt, run)
            return
        self.transport.push_file(
            node, manifest_path, f"{prepared['control_root']}/manifest.json"
        )
        self.transport.push_file(
            node, config_path, f"{prepared['control_root']}/config.yaml"
        )
        self.transport.push_partition_files(
            node,
            source_root,
            files_list_path,
            prepared["data_partial"],
        )
        validate_source_manifest(source_root, manifest_path)
        state_fragment = partition.get("state_fragment_path")
        if state_fragment:
            self.transport.push_file(
                node,
                Path(state_fragment),
                f"{prepared['data_partial']}/balalaika.parquet",
            )
        ready = self.transport.rpc(
            node, {**common, "operation": "validate_input"}, timeout=300
        )
        if ready.get("state") not in {"READY", "STARTING", "RUNNING", "COMPLETED"}:
            raise SchedulerError(
                f"Node {node.id} did not commit input: {ready.get('state')}"
            )
        self.db.update_attempt(attempt["id"], "READY")
        self._start(node, attempt, run)

    def _start(
        self,
        node: NodeConfig,
        attempt: dict[str, Any],
        run: dict[str, Any] | None = None,
    ) -> None:
        saved_run, partition = self._run_and_partition(attempt)
        if run is None:
            run = saved_run
        try:
            gpu_uuids = json.loads(attempt.get("gpu_uuids_json") or "null")
        except json.JSONDecodeError as exc:
            raise SchedulerError("Attempt contains invalid GPU UUID state") from exc
        if (
            not isinstance(gpu_uuids, list)
            or len(gpu_uuids) != len(node.gpu_devices)
            or any(not isinstance(value, str) or not value for value in gpu_uuids)
        ):
            raise SchedulerError("Attempt has no complete GPU UUID snapshot")
        self.db.update_attempt(attempt["id"], "STARTING")
        response = self.transport.rpc(
            node,
            {
                **self._attempt_request(attempt),
                "operation": "start",
                "image": node.image or run["image"],
                "runtime": node.runtime,
                "gpu_devices": list(node.gpu_devices),
                "gpu_uuids": gpu_uuids,
                "pipeline_root": node.pipeline_root,
                "venv_path": node.venv_path,
                "models_root": node.models_root,
                "cache_root": node.cache_root,
                "env_file": node.env_file,
                "global_rank": int(partition["ordinal"]),
                "max_gpu_memory_used_mib": node.max_gpu_memory_used_mib,
                "max_gpu_utilization_percent": node.max_gpu_utilization_percent,
                "stage_start": run["stage_start"],
                "stage_stop": run["stage_stop"],
                "shm_size": self.config.shm_size,
            },
            timeout=180,
        )
        state = response.get("state")
        if state == "COMPLETED":
            self._collect(node, attempt, response)
            return
        if state != "RUNNING":
            raise SchedulerError(f"Node {node.id} returned start state {state!r}")
        self.db.update_attempt(
            attempt["id"], "RUNNING", container_name=response.get("container_name")
        )

    def _collect(
        self, node: NodeConfig, attempt: dict[str, Any], remote_status: dict[str, Any]
    ) -> None:
        run, partition = self._run_and_partition(attempt)
        if remote_status.get("fencing_token") != attempt["fencing_token"]:
            raise SchedulerError("Remote result has a stale fencing token")
        if remote_status.get("manifest_sha256") != partition["manifest_sha256"]:
            raise SchedulerError("Remote result manifest digest does not match")
        if remote_status.get("config_sha256") != run["config_sha256"]:
            raise SchedulerError("Remote result config digest does not match")
        self.db.update_attempt(attempt["id"], "COLLECTING")
        final_root = (
            self.config.results_dir
            / attempt["run_id"]
            / "partitions"
            / attempt["partition_id"]
        )
        partial_root = final_root.with_name(
            f".{final_root.name}.{attempt['id']}.partial"
        )
        final_root.parent.mkdir(parents=True, exist_ok=True)
        if final_root.exists():
            self._verify_collected_result(final_root, attempt, run, partition)
        else:
            self.transport.pull_attempt(node, self._attempt_root(attempt), partial_root)
            self.db.update_attempt(attempt["id"], "VERIFYING")
            self._verify_collected_result(partial_root, attempt, run, partition)
            os.rename(partial_root, final_root)
        self.db.update_attempt(
            attempt["id"],
            "COMPLETED",
            exit_code=0,
            current_stage=run["stage_stop"],
            progress_percent=100.0,
            files_processed=partition["files_total"],
        )

    @staticmethod
    def _verify_collected_result(
        root: Path,
        attempt: dict[str, Any],
        run: dict[str, Any],
        partition: dict[str, Any],
    ) -> None:
        result_path = root / "result.json"
        success_path = root / "_SUCCESS"
        if not result_path.is_file() or not success_path.is_file():
            raise SchedulerError("Collected result is missing result.json or _SUCCESS")
        result = json.loads(result_path.read_text("utf-8"))
        marker = success_path.read_text("ascii").strip()
        checks = {
            "attempt_id": attempt["id"],
            "fencing_token": attempt["fencing_token"],
            "manifest_sha256": partition["manifest_sha256"],
            "config_sha256": run["config_sha256"],
        }
        for key, expected in checks.items():
            if result.get(key) != expected:
                raise SchedulerError(f"Collected result has invalid {key}")
        if marker != attempt["fencing_token"] or result.get("exit_code") != 0:
            raise SchedulerError("Collected success marker is invalid")

    def _reconcile_attempt(self, attempt: dict[str, Any]) -> None:
        run, partition = self._run_and_partition(attempt)
        if (
            run.get("desired_state") == "CANCELLED"
            or partition.get("desired_state") == "CANCELLED"
        ):
            node = self._cancel_node_for_attempt(attempt)
            self._cancel_attempt(node, attempt)
            return
        node = self.config.node(attempt["node_id"])
        state = attempt["state"]
        if state in {"ASSIGNED", "STAGING"}:
            self._stage_and_start(node, attempt)
            return
        if state in {"READY", "STARTING"}:
            self._start(node, attempt)
            return
        response = self.transport.rpc(
            node, {**self._attempt_request(attempt), "operation": "status"}
        )
        remote_state = response.get("state")
        self.db.update_node(node.id, "ONLINE", response)
        if remote_state == "RUNNING":
            self.db.update_attempt(
                attempt["id"],
                "RUNNING",
                container_name=response.get("container_name"),
                current_stage=response.get("current_stage"),
                progress_percent=response.get("progress_percent"),
                files_processed=response.get("files_processed"),
            )
        elif remote_state == "COMPLETED":
            self._collect(node, attempt, response)
        elif remote_state == "CANCELLED":
            if (
                run.get("desired_state") == "CANCELLED"
                or partition.get("desired_state") == "CANCELLED"
            ):
                self.db.update_attempt(attempt["id"], "CANCELLED")
            else:
                self.db.requeue_attempt(
                    attempt["id"],
                    "Remote attempt was cancelled without a controller request",
                    max_attempts=self.config.max_attempts,
                )
        elif remote_state == "FAILED":
            message = response.get("error") or f"Remote state is {remote_state}"
            self.db.requeue_attempt(
                attempt["id"], str(message), max_attempts=self.config.max_attempts
            )
        elif remote_state == "READY":
            self._start(node, attempt)
        elif remote_state == "STAGING":
            self._stage_and_start(node, attempt)
        elif remote_state == "MISSING":
            self._stage_and_start(node, attempt)
        else:
            self.db.update_attempt(
                attempt["id"],
                "UNKNOWN",
                error=response.get("error") or f"Remote state is {remote_state}",
            )

    def _record_attempt_error(self, attempt: dict[str, Any], error: Exception) -> None:
        changed = self.db.update_attempt(attempt["id"], "UNKNOWN", error=str(error))
        if not changed:
            return
        node_state = "OFFLINE" if isinstance(error, TransportError) else "DEGRADED"
        self.db.update_node(attempt["node_id"], node_state, error=str(error))

    def tick(self, run_id: str) -> dict[str, Any]:
        run = self.db.get_run(run_id)
        if run is None:
            raise SchedulerError(f"Unknown run: {run_id}")

        for attempt in self.db.list_active_attempts():
            attempt_run, partition = self._run_and_partition(attempt)
            if (
                attempt_run.get("desired_state") != "CANCELLED"
                and partition.get("desired_state") != "CANCELLED"
            ):
                continue
            try:
                node = self._cancel_node_for_attempt(attempt)
                self._cancel_attempt(node, attempt)
            except (
                TransportError,
                SchedulerError,
                RuntimeError,
                OSError,
                ValueError,
            ) as exc:
                self._record_attempt_error(attempt, exc)

        run = self.db.get_run(run_id)
        if run is None:
            raise SchedulerError(f"Run disappeared during reconciliation: {run_id}")
        if run.get("desired_state") == "CANCELLED":
            self.db.cancel_queued(run_id)
            return self.db.overview(run_id)
        if run.get("state") in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return self.db.overview(run_id)
        self._assert_execution_config(run)

        active_attempts = self.db.list_active_attempts()
        checked_runs = {run_id}
        for attempt in active_attempts:
            attempt_run_id = attempt["run_id"]
            attempt_run, partition = self._run_and_partition(attempt)
            if (
                attempt_run.get("desired_state") == "CANCELLED"
                or partition.get("desired_state") == "CANCELLED"
            ):
                continue
            if attempt_run_id not in checked_runs:
                self._assert_execution_config(attempt_run)
                checked_runs.add(attempt_run_id)
            try:
                self._reconcile_attempt(attempt)
            except (
                TransportError,
                SchedulerError,
                RuntimeError,
                OSError,
                ValueError,
            ) as exc:
                self._record_attempt_error(attempt, exc)

        active_by_node = {item["node_id"] for item in self.db.list_active_attempts()}
        persisted_nodes = {item["id"]: item for item in self.db.list_nodes()}
        for node in self.config.nodes:
            persisted = persisted_nodes.get(node.id)
            if (
                not node.enabled
                or persisted is None
                or not persisted["enabled"]
                or persisted["drained"]
                or node.id in active_by_node
            ):
                continue
            queued = self.db.next_queued_partition(run_id)
            if queued is None:
                break
            try:
                probe = self.transport.probe(node)
                probe_error = self._probe_error(node, probe)
                if probe_error is not None:
                    self.db.update_node(
                        node.id,
                        "DEGRADED",
                        probe,
                        probe_error,
                    )
                    continue
                free_bytes = (probe.get("disk") or {}).get("free_bytes")
                required_bytes = int(
                    int(queued["input_bytes"]) * self.config.workspace_multiplier
                    + self.config.disk_headroom_bytes
                )
                if isinstance(free_bytes, int) and free_bytes < required_bytes:
                    self.db.update_node(
                        node.id,
                        "DEGRADED",
                        probe,
                        f"Insufficient disk: need {required_bytes} bytes, have {free_bytes}",
                    )
                    continue
                self.db.update_node(node.id, "ONLINE", probe)
            except TransportError as exc:
                self.db.update_node(node.id, "OFFLINE", error=str(exc))
                continue
            claimed = self.db.claim_next(
                run_id,
                node.id,
                node.work_root,
                node_host=node.host,
                node_user=node.user,
                node_port=node.port,
                gpu_uuids=probe["gpu_uuids"],
            )
            if not claimed:
                continue
            attempt = self.db.get_attempt(claimed["attempt_id"])
            if attempt is None:
                raise SchedulerError("Claimed attempt disappeared from the database")
            try:
                # Transfers are intentionally serialized here to protect one source HDD.
                self._stage_and_start(node, attempt)
            except (
                TransportError,
                SchedulerError,
                RuntimeError,
                OSError,
                ValueError,
            ) as exc:
                self._record_attempt_error(attempt, exc)

        return self.db.overview(run_id)

    def run_loop(self, run_id: str, *, once: bool = False) -> dict[str, Any]:
        self.initialize()
        with controller_lock(self.config.state_dir):
            while True:
                overview = self.tick(run_id)
                run = overview.get("active_run") or {}
                if once or run.get("state") in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                    return overview
                time.sleep(self.config.poll_seconds)

    def cancel(self, run_id: str, partition_id: str | None = None) -> int:
        run = self.db.get_run(run_id)
        if run is None:
            raise SchedulerError(f"Unknown run: {run_id}")
        if partition_id and self.db.get_partition(run_id, partition_id) is None:
            raise SchedulerError(f"Unknown partition: {run_id}/{partition_id}")
        requested = 0
        if partition_id is None:
            self.db.request_cancel_run(run_id)
        else:
            requested = self.db.request_cancel_partition(run_id, partition_id)
        cancelled, errors = self._cancel_active_attempts(run_id, partition_id)
        cancelled += self.db.cancel_queued(run_id, partition_id)
        if errors:
            raise SchedulerError("; ".join(errors))
        return cancelled + requested

    def _cancel_attempt(self, node: NodeConfig, attempt: dict[str, Any]) -> bool:
        common = self._attempt_request(attempt)
        response = self.transport.rpc(
            node,
            {**common, "operation": "status"},
            timeout=60,
        )
        if response.get("state") == "COMPLETED":
            self._collect(node, attempt, response)
            return False
        if response.get("state") in {"CANCELLED", "MISSING"}:
            self.db.update_attempt(attempt["id"], "CANCELLED")
            return True
        response = self.transport.rpc(
            node,
            {**common, "operation": "cancel"},
            timeout=60,
        )
        if response.get("state") == "COMPLETED":
            self._collect(node, attempt, response)
            return False
        if response.get("state") != "CANCELLED":
            raise SchedulerError(
                f"Node {node.id} returned cancel state {response.get('state')!r}"
            )
        self.db.update_attempt(attempt["id"], "CANCELLED")
        return True

    def _cancel_active_attempts(
        self, run_id: str, partition_id: str | None = None
    ) -> tuple[int, list[str]]:
        cancelled = 0
        errors: list[str] = []
        for attempt in self.db.list_active_attempts(run_id):
            if partition_id and attempt["partition_id"] != partition_id:
                continue
            try:
                node = self._cancel_node_for_attempt(attempt)
                cancelled += int(self._cancel_attempt(node, attempt))
            except (TransportError, SchedulerError, RuntimeError) as exc:
                self.db.update_attempt(attempt["id"], "UNKNOWN", error=str(exc))
                errors.append(str(exc))
        return cancelled, errors

    def retry_failed(self, run_id: str, partition_id: str) -> None:
        partition = self.db.get_partition(run_id, partition_id)
        if not partition:
            raise SchedulerError(f"Unknown partition: {run_id}/{partition_id}")
        if partition["state"] != "FAILED" or partition.get("current_attempt_id"):
            raise SchedulerError("Only a terminal FAILED partition can be retried")
        if not self.db.retry_failed_partition(run_id, partition_id):
            raise SchedulerError("Failed partition changed before it could be retried")
