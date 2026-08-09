from __future__ import annotations

import hashlib
import json
import shlex
import sqlite3
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from cluster_admin.config import (
    ClusterConfig,
    NodeConfig,
    load_cluster_config,
    validate_slug,
)
from cluster_admin.cli import build_parser, command_nodes
from cluster_admin.db import StateDB
from cluster_admin.node_runner import (
    RunnerError,
    operation_cancel,
    operation_prepare,
    operation_status,
    operation_validate_input,
)
from cluster_admin.transport import ClusterTransport, TransportError


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


def _config(tmp_path: Path, *nodes: NodeConfig) -> ClusterConfig:
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("node key\n", encoding="ascii")
    pipeline_config = tmp_path / "pipeline.yaml"
    pipeline_config.write_text("preprocess: {}\n", encoding="ascii")
    return ClusterConfig(
        path=tmp_path / "cluster.yaml",
        state_dir=tmp_path / "state",
        source_root=tmp_path / "dataset",
        pipeline_config=pipeline_config,
        image="balalaika@sha256:" + "1" * 64,
        ssh_identity=None,
        known_hosts=known_hosts,
        connect_timeout=7,
        poll_seconds=2,
        max_attempts=2,
        partitions_per_node=2,
        group_depth=2,
        audio_extensions=(".flac", ".wav"),
        stage_start="1",
        stage_stop="15",
        shm_size="8g",
        nodes=tuple(nodes) or (_node(),),
    )


def _raw_config(tmp_path: Path) -> dict:
    return {
        "controller": {
            "source_root": str(tmp_path / "dataset"),
            "state_dir": str(tmp_path / "state"),
        },
        "pipeline": {
            "config": str(tmp_path / "pipeline.yaml"),
            "image": "balalaika@sha256:" + "1" * 64,
        },
        "ssh": {"known_hosts": str(tmp_path / "known_hosts")},
        "nodes": [
            {
                "id": "node-a",
                "host": "node-a.example.test",
                "user": "balalaika",
                "work_root": "/var/lib/balalaika",
                "models_root": "/var/lib/balalaika/models",
                "cache_root": "/var/lib/balalaika/cache",
            }
        ],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "node;touch_tmp"),
        ("host", "node.example;touch"),
        ("user", "root$(id)"),
        ("work_root", "/tmp/work;touch"),
        ("models_root", "/tmp/models$(id)"),
        ("cache_root", "/tmp/cache name"),
        ("env_file", "/tmp/env`id`"),
    ],
)
def test_config_rejects_ssh_and_remote_path_injection(tmp_path, field, value):
    raw = _raw_config(tmp_path)
    raw["nodes"][0][field] = value
    path = tmp_path / "cluster.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError):
        load_cluster_config(path)


def test_config_rejects_malformed_image_and_slug(tmp_path):
    raw = _raw_config(tmp_path)
    raw["pipeline"]["image"] = "balalaika:latest;touch_/tmp/owned"
    path = tmp_path / "cluster.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="image"):
        load_cluster_config(path)
    with pytest.raises(ValueError, match="identifier"):
        validate_slug("run-a; rm -rf")


def test_transport_rpc_uses_argv_and_json_stdin(monkeypatch, tmp_path):
    node = _node()
    transport = ClusterTransport(_config(tmp_path, node))
    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=b'{"ok":true,"state":"READY"}\n',
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    request = {
        "operation": "status",
        "run_id": "value; touch /tmp/owned",
        "nested": {"value": "$(id) `uname`"},
    }

    response = transport.rpc(node, request)

    assert response == {"ok": True, "state": "READY"}
    argv = captured["argv"]
    kwargs = captured["kwargs"]
    assert isinstance(argv, list)
    assert kwargs["shell"] is False
    assert kwargs["check"] is False
    assert json.loads(kwargs["input"]) == request
    assert request["run_id"] not in " ".join(argv)
    assert request["nested"]["value"] not in " ".join(argv)
    assert argv[-1] == shlex.join(["python3", node.runner_path, "rpc"])
    assert "StrictHostKeyChecking=yes" in argv
    assert "BatchMode=yes" in argv


def test_transport_rpc_rejects_invalid_or_failed_response(monkeypatch, tmp_path):
    node = _node()
    transport = ClusterTransport(_config(tmp_path, node))
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, stdout=b"not-json", stderr=b"bad"),
            subprocess.CompletedProcess(
                [],
                2,
                stdout=b'{"ok":false,"error":"invalid request"}',
                stderr=b"",
            ),
        ]
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: next(responses))

    with pytest.raises(TransportError, match="invalid RPC JSON"):
        transport.rpc(node, {"operation": "status"})
    with pytest.raises(TransportError, match="invalid request"):
        transport.rpc(node, {"operation": "status"})


def _create_db(tmp_path: Path) -> tuple[StateDB, tuple[NodeConfig, NodeConfig]]:
    nodes = (_node("node-a"), _node("node-b"))
    db = StateDB(tmp_path / "controller.sqlite3")
    db.initialize()
    db.sync_nodes(nodes)
    db.create_run(
        {
            "id": "run-a",
            "source_root": str(tmp_path / "dataset"),
            "image": "balalaika@sha256:" + "1" * 64,
            "config_path": str(tmp_path / "pipeline.yaml"),
            "config_sha256": "2" * 64,
            "stage_start": "1",
            "stage_stop": "15",
            "input_bytes": 100,
            "audio_seconds": 100.0,
        },
        [
            {
                "id": "part-heavy",
                "ordinal": 0,
                "manifest_path": str(tmp_path / "heavy.json"),
                "manifest_sha256": "3" * 64,
                "files_list_path": str(tmp_path / "heavy.files0"),
                "files_total": 9,
                "input_bytes": 90,
                "audio_seconds": 90.0,
                "weight": 9.0,
            },
            {
                "id": "part-light",
                "ordinal": 1,
                "manifest_path": str(tmp_path / "light.json"),
                "manifest_sha256": "4" * 64,
                "files_list_path": str(tmp_path / "light.files0"),
                "files_total": 1,
                "input_bytes": 10,
                "audio_seconds": 10.0,
                "weight": 1.0,
            },
        ],
    )
    return db, nodes


def test_node_drain_migrates_and_survives_config_sync(tmp_path):
    path = tmp_path / "controller.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE nodes (
                 id TEXT PRIMARY KEY, host TEXT NOT NULL, user TEXT NOT NULL,
                 port INTEGER NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
                 state TEXT NOT NULL DEFAULT 'UNKNOWN', last_seen_at TEXT,
                 current_partition_id TEXT, details_json TEXT NOT NULL DEFAULT '{}',
                 error TEXT
               )""")
        connection.execute("""INSERT INTO nodes(id, host, user, port, enabled)
               VALUES ('node-a', 'old.example.test', 'old-user', 23, 1)""")

    db = StateDB(path)
    db.initialize()
    assert db.get_node("node-a")["drained"] is False

    drained = db.set_node_drained("node-a", True)
    assert drained["drained"] is True
    assert drained["enabled"] is True

    disabled_config = replace(_node("node-a"), enabled=False)
    db.sync_nodes((disabled_config,))
    persisted = db.get_node("node-a")
    assert persisted["drained"] is True
    assert persisted["enabled"] is False

    resumed = db.set_node_drained("node-a", False)
    assert resumed["drained"] is False
    assert resumed["enabled"] is False


def test_drained_or_disabled_node_cannot_claim(tmp_path):
    db, nodes = _create_db(tmp_path)
    node = nodes[0]

    db.set_node_drained(node.id, True)
    assert db.claim_next("run-a", node.id, node.work_root) is None
    db.sync_nodes((node, replace(nodes[1], enabled=False)))
    assert db.claim_next("run-a", nodes[1].id, nodes[1].work_root) is None

    db.set_node_drained(node.id, False)
    claimed = db.claim_next("run-a", node.id, node.work_root)
    assert claimed is not None


def test_cli_nodes_drain_and_resume(tmp_path, capsys):
    config = _config(tmp_path, _node("node-a"))
    parser = build_parser()

    drain_args = parser.parse_args(["nodes", "drain", "node-a"])
    assert command_nodes(drain_args, config) == 0
    drained_payload = json.loads(capsys.readouterr().out)
    assert drained_payload == {
        "drained": True,
        "enabled": True,
        "node_id": "node-a",
        "ok": True,
    }

    resume_args = parser.parse_args(["nodes", "resume", "node-a"])
    assert command_nodes(resume_args, config) == 0
    resumed_payload = json.loads(capsys.readouterr().out)
    assert resumed_payload["drained"] is False
    assert StateDB(config.db_path).get_node("node-a")["drained"] is False


def test_db_claim_is_weighted_exclusive_and_requeue_increments_attempt(tmp_path):
    db, nodes = _create_db(tmp_path)

    first = db.claim_next("run-a", nodes[0].id, nodes[0].work_root)
    assert first is not None
    assert first["id"] == "part-heavy"
    assert first["attempt_ordinal"] == 1
    assert db.claim_next("run-a", nodes[0].id, nodes[0].work_root) is None

    assert (
        db.requeue_attempt(first["attempt_id"], "transient", max_attempts=2) == "QUEUED"
    )
    retry = db.claim_next("run-a", nodes[0].id, nodes[0].work_root)
    assert retry is not None
    assert retry["id"] == "part-heavy"
    assert retry["attempt_ordinal"] == 2
    assert retry["attempt_id"] != first["attempt_id"]
    assert retry["fencing_token"] != first["fencing_token"]

    assert (
        db.requeue_attempt(retry["attempt_id"], "permanent", max_attempts=2) == "FAILED"
    )
    partition = db.get_partition("run-a", "part-heavy")
    assert partition["state"] == "FAILED"
    # A failed partition must not stop another partition that is still queued.
    assert db.get_run("run-a")["state"] == "PLANNED"
    remaining = db.claim_next("run-a", nodes[1].id, nodes[1].work_root)
    assert remaining is not None
    db.requeue_attempt(remaining["attempt_id"], "permanent", max_attempts=1)
    assert db.get_run("run-a")["state"] == "FAILED"


def test_db_persists_stage_progress_and_aggregates_it_by_weight(tmp_path):
    db, nodes = _create_db(tmp_path)
    heavy = db.claim_next("run-a", nodes[0].id, nodes[0].work_root)
    light = db.claim_next("run-a", nodes[1].id, nodes[1].work_root)
    assert heavy is not None and light is not None

    for state in ("STAGING", "READY", "STARTING"):
        db.update_attempt(heavy["attempt_id"], state)
        db.update_attempt(light["attempt_id"], state)
    db.update_attempt(
        heavy["attempt_id"],
        "RUNNING",
        current_stage="6.5",
        progress_percent=50.0,
        files_processed=4,
    )

    persisted = db.get_partition("run-a", "part-heavy")
    assert persisted["state"] == "RUNNING"
    assert persisted["current_stage"] == "6.5"
    assert persisted["progress_percent"] == 50.0
    assert persisted["files_processed"] == 4
    assert db.overview("run-a")["active_run"]["progress_percent"] == 45.0

    db.update_attempt(heavy["attempt_id"], "COLLECTING")
    db.update_attempt(heavy["attempt_id"], "VERIFYING")
    db.update_attempt(heavy["attempt_id"], "COMPLETED", progress_percent=100.0)
    assert db.overview("run-a")["active_run"]["progress_percent"] == 90.0
    db.update_attempt(light["attempt_id"], "RUNNING")
    db.update_attempt(light["attempt_id"], "COLLECTING")
    db.update_attempt(light["attempt_id"], "VERIFYING")
    db.update_attempt(light["attempt_id"], "COMPLETED", progress_percent=100.0)
    assert db.get_run("run-a")["state"] == "SUCCEEDED"
    assert db.overview("run-a")["active_run"]["progress_percent"] == 100.0


def _attempt_request(tmp_path: Path, manifest: dict, config_bytes: bytes = b"x: 1\n"):
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    request = {
        "operation": "prepare",
        "work_root": str(tmp_path / "worker"),
        "run_id": "run-a",
        "partition_id": "part-0001",
        "attempt_id": "a" * 32,
        "attempt_ordinal": 1,
        "fencing_token": "b" * 32,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    prepared = operation_prepare(request)
    control = Path(prepared["control_root"])
    control.joinpath("manifest.json").write_bytes(manifest_bytes)
    control.joinpath("config.yaml").write_bytes(config_bytes)
    return request, prepared


def test_node_runner_prepare_and_validate_manifest_without_docker(tmp_path):
    content = b"audio bytes"
    manifest = {
        "files": [
            {
                "path": "2026-08/video-a/audio.flac",
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        ],
        "files_total": 1,
        "input_bytes": len(content),
    }
    request, prepared = _attempt_request(tmp_path, manifest)
    source = Path(prepared["data_partial"]) / manifest["files"][0]["path"]
    source.parent.mkdir(parents=True)
    source.write_bytes(content)

    ready = operation_validate_input(request | {"operation": "validate_input"})

    assert ready["ok"] is True
    assert ready["state"] == "READY"
    assert ready["files_total"] == 1
    assert ready["input_bytes"] == len(content)
    assert (
        Path(ready["data_root"], manifest["files"][0]["path"]).read_bytes() == content
    )
    assert not Path(prepared["data_partial"]).exists()

    repeated = operation_validate_input(request | {"operation": "validate_input"})
    assert repeated["state"] == "READY"
    assert repeated["data_root"] == ready["data_root"]


def test_node_runner_cancelled_attempt_cannot_return_to_ready(tmp_path):
    manifest = {"files": [], "files_total": 0, "input_bytes": 0}
    request, _ = _attempt_request(tmp_path, manifest)

    cancelled = operation_cancel(request | {"operation": "cancel"})
    assert cancelled["state"] == "CANCELLED"
    assert operation_status(request | {"operation": "status"})["state"] == "CANCELLED"
    with pytest.raises(RunnerError, match="already CANCELLED"):
        operation_validate_input(request | {"operation": "validate_input"})


@pytest.mark.parametrize(
    ("paths", "match"),
    [
        (["same/audio.flac", "same/audio.flac"], "Duplicate manifest path"),
        (["../outside.flac"], "Unsafe manifest path"),
        (["/etc/passwd"], "Unsafe manifest path"),
        (["./not-canonical.flac"], "Non-canonical manifest path"),
    ],
)
def test_node_runner_rejects_duplicate_and_unsafe_manifest_paths(
    tmp_path, paths, match
):
    content = b"x"
    manifest = {
        "files": [{"path": path, "size": 1} for path in paths],
        "files_total": len(paths),
        "input_bytes": len(paths),
    }
    request, prepared = _attempt_request(tmp_path, manifest)
    safe = Path(prepared["data_partial"]) / "same/audio.flac"
    safe.parent.mkdir(parents=True)
    safe.write_bytes(content)

    with pytest.raises(RunnerError, match=match):
        operation_validate_input(request | {"operation": "validate_input"})


def test_node_runner_rejects_bad_and_conflicting_fencing_tokens(tmp_path):
    manifest = {"files": [], "files_total": 0, "input_bytes": 0}
    request, _ = _attempt_request(tmp_path, manifest)

    with pytest.raises(RunnerError, match="Fencing token"):
        operation_validate_input(
            request
            | {
                "operation": "validate_input",
                "fencing_token": "c" * 32,
            }
        )
    with pytest.raises(RunnerError, match="Invalid fencing_token"):
        operation_validate_input(
            request
            | {
                "operation": "validate_input",
                "fencing_token": "bad;token",
            }
        )
    with pytest.raises(RunnerError, match="different job"):
        operation_prepare(request | {"fencing_token": "c" * 32})


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("work_root", "/tmp/worker;touch", "Invalid work_root"),
        ("run_id", "../run", "Invalid run_id"),
        ("partition_id", "part/../../x", "Invalid partition_id"),
        ("attempt_id", "not-a-token", "Invalid attempt_id"),
    ],
)
def test_node_runner_rejects_injected_attempt_identity(tmp_path, field, value, match):
    manifest = {"files": [], "files_total": 0, "input_bytes": 0}
    request, _ = _attempt_request(tmp_path, manifest)
    request[field] = value

    with pytest.raises(RunnerError, match=match):
        operation_prepare(request)
