from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

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


def test_dashboard_serves_read_only_api_and_static_assets(tmp_path):
    config = _config(tmp_path)
    database = StateDB(config.db_path)
    database.initialize()
    database.sync_nodes(config.nodes)
    server = build_server(config, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{base}/api/v1/overview", timeout=2) as response:
            overview = json.load(response)
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["X-Frame-Options"] == "DENY"
        assert overview["controller"]["status"] == "online"
        assert overview["nodes"][0]["id"] == "node-a"

        for path, expected in (
            ("/", b"Balalaika Cluster Admin"),
            ("/static/app.js", b"/api/v1/overview"),
            ("/static/styles.css", b"--accent"),
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
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
