from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from cluster_admin.config import ClusterConfig, NodeConfig
from cluster_admin.db import StateDB
from cluster_admin.server import build_server


def _config(tmp_path: Path) -> ClusterConfig:
    source = tmp_path / "dataset"
    source.mkdir()
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text("preprocess: {}\n", encoding="ascii")
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("host key\n", encoding="ascii")
    return ClusterConfig(
        path=tmp_path / "cluster.yaml",
        state_dir=tmp_path / "state",
        source_root=source,
        pipeline_config=pipeline,
        image="balalaika:cuda12.8",
        ssh_identity=None,
        known_hosts=known_hosts,
        connect_timeout=2,
        poll_seconds=2,
        max_attempts=2,
        partitions_per_node=2,
        group_depth=1,
        audio_extensions=(".flac",),
        stage_start="1",
        stage_stop="2",
        shm_size="8g",
        nodes=(
            NodeConfig(
                id="node-a",
                host="node-a.example.test",
                user="balalaika",
                port=22,
                work_root="/var/lib/balalaika",
                models_root="/var/lib/balalaika/models",
                cache_root="/var/lib/balalaika/cache",
            ),
        ),
    )


class RecordingCancelScheduler:
    def __init__(self, database: StateDB):
        self.database = database
        self.calls: list[tuple[str, str | None]] = []

    def cancel(self, run_id: str, partition_id: str | None = None) -> int:
        self.calls.append((run_id, partition_id))
        raise AssertionError("dashboard must not perform remote scheduler work")


def _seed_active_partition(database: StateDB, config: ClusterConfig) -> None:
    database.create_run(
        {
            "id": "run-active",
            "source_root": str(config.source_root),
            "image": config.image,
            "config_path": str(config.pipeline_config),
            "config_sha256": "a" * 64,
            "execution_config_sha256": "b" * 64,
            "stage_start": config.stage_start,
            "stage_stop": config.stage_stop,
            "input_bytes": 100,
            "audio_seconds": 10.0,
        },
        [
            {
                "id": "part-0000",
                "ordinal": 0,
                "manifest_path": "/tmp/part-0000.json",
                "manifest_sha256": "c" * 64,
                "files_list_path": "/tmp/part-0000.files",
                "files_total": 1,
                "input_bytes": 100,
                "audio_seconds": 10.0,
                "weight": 10.0,
            }
        ],
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE partitions SET state='RUNNING', node_id='node-a' "
            "WHERE run_id='run-active' AND id='part-0000'"
        )
        connection.execute(
            "UPDATE nodes SET state='BUSY', current_partition_id='part-0000' "
            "WHERE id='node-a'"
        )


@contextmanager
def _running_server(
    config: ClusterConfig, scheduler: RecordingCancelScheduler | None = None
) -> Iterator[tuple[object, str, urllib.request.OpenerDirector]]:
    server = build_server(config, "127.0.0.1", 0, scheduler=scheduler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        yield server, base, opener
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _csrf_token(opener: urllib.request.OpenerDirector, base: str) -> str:
    with opener.open(f"{base}/api/v1/overview", timeout=2) as response:
        return json.load(response)["controller"]["csrf_token"]


def _cancel_request(
    base: str,
    token: str,
    *,
    node_id: str = "node-a",
    origin: str | None = None,
) -> urllib.request.Request:
    return urllib.request.Request(
        f"{base}/api/v1/runs/run-active/partitions/part-0000/cancel",
        data=json.dumps({"node_id": node_id}).encode("ascii"),
        headers={
            "Content-Type": "application/json",
            "Origin": origin or base,
            "X-CSRF-Token": token,
        },
        method="POST",
    )


def _node_state_request(
    base: str,
    token: str,
    operation: str,
    *,
    node_id: str = "node-a",
) -> urllib.request.Request:
    return urllib.request.Request(
        f"{base}/api/v1/nodes/{node_id}/{operation}",
        data=b"{}",
        headers={
            "Content-Type": "application/json",
            "Origin": base,
            "X-CSRF-Token": token,
        },
        method="POST",
    )


def test_dashboard_serves_api_and_static_assets(tmp_path):
    config = _config(tmp_path)
    database = StateDB(config.db_path)
    database.initialize()
    database.sync_nodes(config.nodes)
    with _running_server(config) as (_, base, opener):
        with opener.open(f"{base}/api/v1/overview", timeout=2) as response:
            overview = json.load(response)
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["X-Frame-Options"] == "DENY"
        assert overview["controller"]["status"] == "online"
        assert overview["controller"]["csrf_token"]
        assert overview["nodes"][0]["id"] == "node-a"

        for path, expected in (
            ("/", b"Balalaika Cluster Admin"),
            ("/static/app.js", b"X-CSRF-Token"),
            ("/static/styles.css", b".button--danger"),
        ):
            with opener.open(f"{base}{path}", timeout=2) as response:
                assert expected in response.read()

        request = urllib.request.Request(f"{base}/", method="HEAD")
        with opener.open(request, timeout=2) as response:
            assert response.status == 200
            assert response.read() == b""

        try:
            opener.open(f"{base}/static/../config.py", timeout=2)
        except urllib.error.HTTPError as error:
            assert error.code == 404
        else:  # pragma: no cover - makes a path traversal regression explicit
            raise AssertionError("path traversal unexpectedly succeeded")


def test_cancel_api_requires_same_origin_and_csrf_token(tmp_path):
    config = _config(tmp_path)
    database = StateDB(config.db_path)
    database.initialize()
    database.sync_nodes(config.nodes)
    _seed_active_partition(database, config)
    scheduler = RecordingCancelScheduler(database)

    with _running_server(config, scheduler) as (_, base, opener):
        token = _csrf_token(opener, base)
        for request in (
            _cancel_request(base, "invalid-token"),
            _cancel_request(base, token, origin="http://attacker.example"),
        ):
            try:
                opener.open(request, timeout=2)
            except urllib.error.HTTPError as error:
                assert error.code == 403
            else:  # pragma: no cover - makes a CSRF regression explicit
                raise AssertionError("cross-origin cancellation unexpectedly accepted")

    assert scheduler.calls == []
    partition = database.get_partition("run-active", "part-0000")
    assert partition["desired_state"] == "RUNNING"
    assert database.get_node("node-a")["drained"] is False


def test_cancel_api_targets_the_visible_partition_on_the_node(tmp_path):
    config = _config(tmp_path)
    database = StateDB(config.db_path)
    database.initialize()
    database.sync_nodes(config.nodes)
    _seed_active_partition(database, config)
    scheduler = RecordingCancelScheduler(database)

    with _running_server(config, scheduler) as (_, base, opener):
        request = _cancel_request(base, _csrf_token(opener, base))
        with opener.open(request, timeout=2) as response:
            payload = json.load(response)
            assert response.status == 202

    assert payload["cancellation_requested"] is True
    assert payload["desired_state"] == "CANCELLED"
    assert payload["node_drained"] is True
    assert payload["drained_node_id"] == "node-a"
    assert scheduler.calls == []
    partition = database.get_partition("run-active", "part-0000")
    assert partition["desired_state"] == "CANCELLED"
    assert database.get_node("node-a")["drained"] is True


def test_cancel_api_only_persists_intent_for_the_scheduler_loop(tmp_path):
    config = _config(tmp_path)
    database = StateDB(config.db_path)
    database.initialize()
    database.sync_nodes(config.nodes)
    _seed_active_partition(database, config)
    scheduler = RecordingCancelScheduler(database)

    with _running_server(config, scheduler) as (_, base, opener):
        request = _cancel_request(base, _csrf_token(opener, base))
        with opener.open(request, timeout=2) as response:
            payload = json.load(response)
            assert response.status == 202

    assert payload["cancellation_requested"] is True
    assert payload["desired_state"] == "CANCELLED"
    assert payload["node_drained"] is True
    assert scheduler.calls == []
    partition = database.get_partition("run-active", "part-0000")
    assert partition["desired_state"] == "CANCELLED"
    assert database.get_node("node-a")["drained"] is True


def test_cancel_api_rejects_a_stale_node_card(tmp_path):
    config = _config(tmp_path)
    database = StateDB(config.db_path)
    database.initialize()
    database.sync_nodes(config.nodes)
    _seed_active_partition(database, config)
    scheduler = RecordingCancelScheduler(database)
    with database.connect() as connection:
        connection.execute(
            "UPDATE nodes SET current_partition_id=NULL WHERE id='node-a'"
        )

    with _running_server(config, scheduler) as (_, base, opener):
        request = _cancel_request(base, _csrf_token(opener, base))
        try:
            opener.open(request, timeout=2)
        except urllib.error.HTTPError as error:
            assert error.code == 409
        else:  # pragma: no cover - makes stale UI fencing explicit
            raise AssertionError("stale node cancellation unexpectedly accepted")

    assert scheduler.calls == []
    assert database.get_node("node-a")["drained"] is False


def test_dashboard_rejects_non_loopback_bind(tmp_path):
    config = _config(tmp_path)
    for host in ("0.0.0.0", "192.0.2.10", "public.example.test"):
        try:
            build_server(config, host, 0)
        except ValueError as exc:
            assert "SSH tunnel" in str(exc)
        else:  # pragma: no cover - external dashboard exposure is a regression
            raise AssertionError(f"non-loopback dashboard bind accepted: {host}")


def test_node_drain_and_resume_api_are_explicit_and_preserve_enabled(tmp_path):
    config = _config(tmp_path)
    database = StateDB(config.db_path)
    database.initialize()
    database.sync_nodes(config.nodes)

    with _running_server(config) as (_, base, opener):
        token = _csrf_token(opener, base)
        with opener.open(
            _node_state_request(base, token, "drain"), timeout=2
        ) as response:
            drained = json.load(response)
            assert response.status == 200
        assert drained == {
            "ok": True,
            "node_id": "node-a",
            "drained": True,
            "enabled": True,
        }

        with opener.open(f"{base}/api/v1/overview", timeout=2) as response:
            overview = json.load(response)
        assert overview["nodes"][0]["drained"] is True

        with opener.open(
            _node_state_request(base, token, "resume"), timeout=2
        ) as response:
            resumed = json.load(response)
        assert resumed["drained"] is False
        assert resumed["enabled"] is True

    assert database.get_node("node-a")["drained"] is False
