from __future__ import annotations

import hashlib
import io
import json
import shlex
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest
import yaml

import cluster_admin.node_runner as node_runner
from cluster_admin.config import ClusterConfig, NodeConfig, load_cluster_config
from cluster_admin.transport import (
    STREAM_PROTOCOL_VERSION,
    ClusterTransport,
    TransportError,
)


def _node(tmp_path: Path, *, transfer: str = "stream") -> NodeConfig:
    work_root = tmp_path / "worker"
    return NodeConfig(
        id="node-a",
        host="node-a.example.test",
        user="balalaika",
        port=22,
        work_root=str(work_root),
        models_root=str(work_root / "models"),
        cache_root=str(work_root / "cache"),
        transfer=transfer,
    )


def _config(tmp_path: Path, node: NodeConfig) -> ClusterConfig:
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
        audio_extensions=(".flac",),
        stage_start="1",
        stage_stop="15",
        shm_size="8g",
        nodes=(node,),
    )


class _LocalRunnerTransport(ClusterTransport):
    """Replace SSH with a subprocess while preserving the fixed command contract."""

    def _ssh_command(self, node: NodeConfig, remote_command: str) -> list[str]:
        runner = Path(node_runner.__file__).resolve()
        for command in ("stream-push", "stream-pull"):
            expected = shlex.join(["python3", node.runner_path, command])
            if remote_command == expected:
                return [sys.executable, str(runner), command]
        raise AssertionError(
            f"Unexpected or non-fixed remote command: {remote_command}"
        )


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def _prepare(
    tmp_path: Path, manifest: dict[str, Any], config: bytes = b"preprocess: {}\n"
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    manifest_bytes = _canonical_json(manifest)
    request = {
        "work_root": str(tmp_path / "worker"),
        "run_id": "run-a",
        "partition_id": "part-0000",
        "attempt_id": "a" * 32,
        "attempt_ordinal": 1,
        "fencing_token": "b" * 32,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "config_sha256": hashlib.sha256(config).hexdigest(),
    }
    prepared = node_runner.operation_prepare(request)
    return request, prepared, manifest_bytes


def _stream_frame(
    attempt_root: str,
    role: str,
    members: list[tuple[tarfile.TarInfo, bytes | None]],
) -> io.BytesIO:
    output = io.BytesIO()
    node_runner._stream_write_header(
        output,
        {
            "protocol": STREAM_PROTOCOL_VERSION,
            "attempt_root": attempt_root,
            "role": role,
        },
    )
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for member, content in members:
            archive.addfile(
                member, io.BytesIO(content) if content is not None else None
            )
    output.seek(0)
    return output


def _regular_member(name: str, content: bytes) -> tuple[tarfile.TarInfo, bytes]:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    return member, content


def test_stream_transport_stages_and_pulls_attempt_without_rsync(tmp_path):
    audio = b"audio payload"
    state = b"parquet fragment"
    config_bytes = b"preprocess: {}\n"
    manifest = {
        "files": [
            {
                "path": "video-a/audio.flac",
                "size": len(audio),
                "sha256": hashlib.sha256(audio).hexdigest(),
            }
        ],
        "files_total": 1,
        "input_bytes": len(audio),
        "state_fragment": "balalaika.parquet",
        "state_fragment_metadata": {
            "path": "balalaika.parquet",
            "bytes": len(state),
            "sha256": hashlib.sha256(state).hexdigest(),
        },
    }
    request, prepared, manifest_bytes = _prepare(tmp_path, manifest, config_bytes)
    source_root = tmp_path / "dataset"
    source = source_root / "video-a" / "audio.flac"
    source.parent.mkdir(parents=True)
    source.write_bytes(audio)
    files_list = tmp_path / "part.files0"
    files_list.write_bytes(b"video-a/audio.flac\0")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    config_path = tmp_path / "config.yaml"
    config_path.write_bytes(config_bytes)
    state_path = tmp_path / "state.parquet"
    state_path.write_bytes(state)

    node = _node(tmp_path, transfer="auto")
    transport = _LocalRunnerTransport(_config(tmp_path, node))
    transport.rsync_binary = tmp_path / "missing-rsync"
    transport.push_file(
        node, manifest_path, f"{prepared['control_root']}/manifest.json"
    )
    transport.push_file(node, config_path, f"{prepared['control_root']}/config.yaml")
    transport.push_partition_files(
        node, source_root, files_list, prepared["data_partial"]
    )
    transport.push_file(
        node, state_path, f"{prepared['data_partial']}/balalaika.parquet"
    )

    ready = node_runner.operation_validate_input(request)
    assert ready["state"] == "READY"
    attempt_root = Path(prepared["attempt_root"])
    metadata = node_runner.read_json(attempt_root / "control" / "job.json")
    metadata["state"] = "COMPLETED"
    node_runner.atomic_json(attempt_root / "control" / "job.json", metadata)
    result = {
        "attempt_id": request["attempt_id"],
        "fencing_token": request["fencing_token"],
        "manifest_sha256": request["manifest_sha256"],
        "config_sha256": request["config_sha256"],
        "exit_code": 0,
    }
    node_runner.atomic_json(attempt_root / "result.json", result)
    (attempt_root / "_SUCCESS").write_text(
        request["fencing_token"] + "\n", encoding="ascii"
    )

    local_result = tmp_path / "result.partial"
    transport.pull_attempt(node, str(attempt_root), local_result)

    assert (local_result / "data" / "video-a" / "audio.flac").read_bytes() == audio
    assert (local_result / "data" / "balalaika.parquet").read_bytes() == state
    assert json.loads((local_result / "result.json").read_text("utf-8")) == result
    assert (local_result / "_SUCCESS").read_text("ascii").strip() == "b" * 32


@pytest.mark.parametrize(
    ("member_factory", "match"),
    [
        (lambda: _regular_member("../outside", b"x"), "Unsafe manifest path"),
        (lambda: _regular_member("extra.flac", b"x"), "Unexpected stream member"),
        (
            lambda: (
                _link_member("safe/audio.flac", tarfile.SYMTYPE, "elsewhere"),
                None,
            ),
            "not a regular file",
        ),
        (
            lambda: (
                _link_member("safe/audio.flac", tarfile.LNKTYPE, "elsewhere"),
                None,
            ),
            "not a regular file",
        ),
        (
            lambda: (_special_member("safe/audio.flac"), None),
            "not a regular file",
        ),
    ],
)
def test_stream_push_rejects_traversal_links_special_and_extra_paths(
    tmp_path, member_factory, match
):
    content = b"x"
    manifest = {
        "files": [
            {
                "path": "safe/audio.flac",
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        ],
        "files_total": 1,
        "input_bytes": len(content),
    }
    _, prepared, manifest_bytes = _prepare(tmp_path, manifest)
    control = Path(prepared["control_root"])
    control.joinpath("manifest.json").write_bytes(manifest_bytes)
    control.joinpath("config.yaml").write_bytes(b"preprocess: {}\n")

    with pytest.raises(node_runner.RunnerError, match=match):
        node_runner.stream_push(
            _stream_frame(prepared["attempt_root"], "data", [member_factory()])
        )

    assert not (tmp_path / "outside").exists()


def _link_member(name: str, kind: bytes, target: str) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.type = kind
    member.linkname = target
    return member


def _special_member(name: str) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.type = tarfile.CHRTYPE
    member.devmajor = 1
    member.devminor = 3
    return member


def test_stream_push_rejects_manifest_that_differs_from_prepared_digest(tmp_path):
    manifest = {"files": [], "files_total": 0, "input_bytes": 0}
    _, prepared, _ = _prepare(tmp_path, manifest)
    changed = _canonical_json(manifest | {"input_bytes": 1})

    with pytest.raises(node_runner.RunnerError, match="SHA256"):
        node_runner.stream_push(
            _stream_frame(
                prepared["attempt_root"],
                "manifest",
                [_regular_member("manifest.json", changed)],
            )
        )

    assert not Path(prepared["control_root"], "manifest.json").exists()


def test_stream_pull_rejects_symlink_in_remote_attempt(tmp_path):
    manifest = {"files": [], "files_total": 0, "input_bytes": 0}
    request, prepared, manifest_bytes = _prepare(tmp_path, manifest)
    root = Path(prepared["attempt_root"])
    control = root / "control"
    control.joinpath("manifest.json").write_bytes(manifest_bytes)
    control.joinpath("config.yaml").write_bytes(b"preprocess: {}\n")
    metadata = node_runner.read_json(control / "job.json")
    metadata["state"] = "COMPLETED"
    node_runner.atomic_json(control / "job.json", metadata)
    node_runner.atomic_json(root / "result.json", {"attempt_id": request["attempt_id"]})
    (root / "_SUCCESS").write_text(request["fencing_token"] + "\n", encoding="ascii")
    (root / "unsafe-link").symlink_to("result.json")
    request_stream = io.BytesIO()
    node_runner._stream_write_header(
        request_stream,
        {"protocol": STREAM_PROTOCOL_VERSION, "attempt_root": str(root)},
    )
    request_stream.seek(0)

    with pytest.raises(node_runner.RunnerError, match="unsafe filesystem entry"):
        node_runner.stream_pull(request_stream, io.BytesIO())


def test_auto_transfer_defaults_to_stream_and_uses_probe_capabilities(
    monkeypatch, tmp_path
):
    node = _node(tmp_path, transfer="auto")
    transport = ClusterTransport(_config(tmp_path, node))
    assert transport._transfer_mode(node) == "stream"

    monkeypatch.setattr(
        transport,
        "rpc",
        lambda *_args, **_kwargs: {
            "ok": True,
            "rsync_ok": False,
            "stream_protocol": STREAM_PROTOCOL_VERSION,
        },
    )
    assert transport.probe(node)["selected_transfer"] == "stream"

    monkeypatch.setattr(
        transport,
        "rpc",
        lambda *_args, **_kwargs: {
            "ok": True,
            "rsync_ok": True,
            "stream_protocol": STREAM_PROTOCOL_VERSION,
        },
    )
    transport.rsync_binary = Path("/usr/bin/rsync")
    assert transport.probe(node)["selected_transfer"] == "rsync"


def test_cluster_config_transfer_defaults_and_validation(tmp_path):
    raw = {
        "controller": {"source_root": str(tmp_path / "dataset")},
        "pipeline": {"config": str(tmp_path / "pipeline.yaml")},
        "ssh": {"known_hosts": str(tmp_path / "known_hosts")},
        "nodes": [
            {
                "id": "node-a",
                "host": "node-a.example.test",
                "work_root": "/var/lib/balalaika",
                "models_root": "/var/lib/balalaika/models",
                "cache_root": "/var/lib/balalaika/cache",
            }
        ],
    }
    config_path = tmp_path / "cluster.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_cluster_config(config_path).nodes[0].transfer == "rsync"

    raw["nodes"][0]["transfer"] = "stream"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_cluster_config(config_path).nodes[0].transfer == "stream"

    raw["nodes"][0]["transfer"] = "scp;touch"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="transfer"):
        load_cluster_config(config_path)


def test_stream_command_contains_no_transfer_paths(tmp_path):
    node = _node(tmp_path)
    transport = ClusterTransport(_config(tmp_path, node))
    argv = transport._stream_command(node, "stream-push")

    assert argv[-1] == shlex.join(["python3", node.runner_path, "stream-push"])
    assert "dataset" not in argv[-1]
    with pytest.raises(TransportError, match="Invalid runner stream command"):
        transport._stream_command(node, "stream-push; touch /tmp/owned")


def test_explicit_rsync_mode_keeps_existing_transfer_path(monkeypatch, tmp_path):
    node = _node(tmp_path, transfer="rsync")
    transport = ClusterTransport(_config(tmp_path, node))
    source = tmp_path / "manifest.json"
    source.write_text("{}", encoding="ascii")
    captured: list[tuple[NodeConfig, list[str]]] = []

    def fake_rsync(
        selected_node: NodeConfig, arguments: list[str], **_kwargs
    ) -> None:
        captured.append((selected_node, arguments))

    monkeypatch.setattr(transport, "_run_rsync", fake_rsync)
    transport.push_file(
        node,
        source,
        "/var/lib/balalaika/runs/run-a/partitions/part-a/"
        "attempt-001-aaaaaaaaaaaa/control/manifest.json",
    )

    assert captured[0][0] == node
    assert captured[0][1][0:2] == ["--times", "--"]
