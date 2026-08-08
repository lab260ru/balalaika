from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,252}$")
USER_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")
REMOTE_PATH_RE = re.compile(r"^/[A-Za-z0-9_./-]+$")
IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,255}$")
STAGE_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]{1,2})?$")
SHM_RE = re.compile(r"^[1-9][0-9]*[kKmMgG]?[bB]?$")
ALLOWED_STAGES = (
    "0",
    "1",
    "2",
    "3",
    "3.5",
    "4",
    "4.5",
    "5",
    "5.5",
    "6",
    "6.5",
    "7",
    "7.5",
    "8",
    "9",
    "10",
    "11",
    "12",
    "12.5",
    "13",
    "14",
    "15",
)


def validate_slug(value: str, label: str = "identifier") -> str:
    if not SLUG_RE.fullmatch(value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def _local_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _remote_path(value: str, label: str) -> str:
    if not REMOTE_PATH_RE.fullmatch(value):
        raise ValueError(
            f"{label} must be an absolute POSIX path containing only "
            f"letters, digits, '_', '.', '-' and '/': {value!r}"
        )
    path = PurePosixPath(value)
    if ".." in path.parts:
        raise ValueError(f"{label} cannot contain '..': {value!r}")
    return str(path)


@dataclass(frozen=True)
class NodeConfig:
    id: str
    host: str
    user: str
    port: int
    work_root: str
    models_root: str
    cache_root: str
    image: str | None = None
    env_file: str | None = None
    enabled: bool = True

    @property
    def destination(self) -> str:
        return f"{self.user}@{self.host}"

    @property
    def runner_path(self) -> str:
        return f"{self.work_root}/bin/balalaika-node-runner.py"


@dataclass(frozen=True)
class ClusterConfig:
    path: Path
    state_dir: Path
    source_root: Path
    pipeline_config: Path
    image: str
    ssh_identity: Path | None
    known_hosts: Path
    connect_timeout: int
    poll_seconds: int
    max_attempts: int
    partitions_per_node: int
    group_depth: int
    audio_extensions: tuple[str, ...]
    stage_start: str
    stage_stop: str
    shm_size: str
    nodes: tuple[NodeConfig, ...]
    exclude_top_level: tuple[str, ...] = (
        ".balalaika_work",
        "balalaika_analysis",
        "filter_report.md",
        "filter_summary.csv",
    )
    workspace_multiplier: float = 2.0
    disk_headroom_bytes: int = 20 * 1024**3

    @property
    def db_path(self) -> Path:
        return self.state_dir / "controller.sqlite3"

    @property
    def manifests_dir(self) -> Path:
        return self.state_dir / "manifests"

    @property
    def results_dir(self) -> Path:
        return self.state_dir / "results"

    def node(self, node_id: str) -> NodeConfig:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(f"Unknown node: {node_id}")


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def load_cluster_config(path: str | Path) -> ClusterConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Cluster config top-level value must be a mapping")

    base = config_path.parent
    controller = _mapping(raw, "controller")
    pipeline = _mapping(raw, "pipeline")
    partitioning = _mapping(raw, "partitioning")
    ssh = _mapping(raw, "ssh")

    source_root = _local_path(
        controller.get("source_root", "/mnt/hdd_6tb_1/youtube_data_incoming"),
        base,
    )
    state_dir = _local_path(controller.get("state_dir", ".balalaika-cluster"), base)
    pipeline_config = _local_path(pipeline.get("config", "configs/config.yaml"), base)
    known_hosts = _local_path(ssh.get("known_hosts", "~/.ssh/known_hosts"), base)
    identity_value = ssh.get("identity_file")
    identity = _local_path(identity_value, base) if identity_value else None

    nodes_raw = raw.get("nodes")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise ValueError("nodes must be a non-empty list")
    nodes: list[NodeConfig] = []
    seen: set[str] = set()
    for item in nodes_raw:
        if not isinstance(item, dict):
            raise ValueError("Every node entry must be a mapping")
        node_id = validate_slug(str(item.get("id", "")), "node id")
        if node_id in seen:
            raise ValueError(f"Duplicate node id: {node_id}")
        seen.add(node_id)
        host = str(item.get("host", ""))
        user = str(item.get("user", "balalaika"))
        if not HOST_RE.fullmatch(host):
            raise ValueError(f"Invalid SSH host for {node_id}: {host!r}")
        if not USER_RE.fullmatch(user):
            raise ValueError(f"Invalid SSH user for {node_id}: {user!r}")
        port = int(item.get("port", 22))
        if not 1 <= port <= 65535:
            raise ValueError(f"Invalid SSH port for {node_id}: {port}")
        env_file = item.get("env_file")
        nodes.append(
            NodeConfig(
                id=node_id,
                host=host,
                user=user,
                port=port,
                work_root=_remote_path(
                    str(item.get("work_root", "/var/lib/balalaika")),
                    f"nodes.{node_id}.work_root",
                ),
                models_root=_remote_path(
                    str(item.get("models_root", "/var/lib/balalaika/models")),
                    f"nodes.{node_id}.models_root",
                ),
                cache_root=_remote_path(
                    str(item.get("cache_root", "/var/lib/balalaika/cache")),
                    f"nodes.{node_id}.cache_root",
                ),
                image=str(item["image"]) if item.get("image") else None,
                env_file=(
                    _remote_path(str(env_file), f"nodes.{node_id}.env_file")
                    if env_file
                    else None
                ),
                enabled=bool(item.get("enabled", True)),
            )
        )

    extensions = partitioning.get(
        "audio_extensions", [".flac", ".wav", ".mp3", ".ogg", ".opus", ".m4a"]
    )
    if not isinstance(extensions, list) or not extensions:
        raise ValueError("partitioning.audio_extensions must be a non-empty list")
    normalized_extensions = tuple(
        sorted(
            {
                str(ext).lower() if str(ext).startswith(".") else f".{ext}"
                for ext in extensions
            }
        )
    )
    excluded = partitioning.get(
        "exclude_top_level",
        [
            ".balalaika_work",
            "balalaika_analysis",
            "filter_report.md",
            "filter_summary.csv",
        ],
    )
    if not isinstance(excluded, list):
        raise ValueError("partitioning.exclude_top_level must be a list")
    normalized_excluded: list[str] = []
    for value in excluded:
        name = str(value)
        if (
            not name
            or name in {".", ".."}
            or "/" in name
            or "\x00" in name
            or any(ord(character) < 32 for character in name)
        ):
            raise ValueError(f"Invalid excluded top-level name: {name!r}")
        normalized_excluded.append(name)

    image = str(pipeline.get("image", "balalaika:cuda12.8"))
    if not IMAGE_RE.fullmatch(image):
        raise ValueError("pipeline.image contains unsupported characters")
    for node in nodes:
        if node.image is not None and not IMAGE_RE.fullmatch(node.image):
            raise ValueError(f"nodes.{node.id}.image contains unsupported characters")

    stage_start = str(pipeline.get("stage_start", "1"))
    stage_stop = str(pipeline.get("stage_stop", "15"))
    if not STAGE_RE.fullmatch(stage_start) or not STAGE_RE.fullmatch(stage_stop):
        raise ValueError("pipeline stage values must be non-negative numbers")
    if stage_start not in ALLOWED_STAGES or stage_stop not in ALLOWED_STAGES:
        raise ValueError("pipeline contains an unsupported stage")
    if ALLOWED_STAGES.index(stage_start) > ALLOWED_STAGES.index(stage_stop):
        raise ValueError("pipeline.stage_start must not be after stage_stop")
    shm_size = str(pipeline.get("shm_size", "8g"))
    if not SHM_RE.fullmatch(shm_size):
        raise ValueError("pipeline.shm_size is invalid")
    workspace_multiplier = float(controller.get("workspace_multiplier", 2.0))
    if not math.isfinite(workspace_multiplier) or workspace_multiplier < 1.0:
        raise ValueError("controller.workspace_multiplier must be at least 1.0")
    disk_headroom_gib = float(controller.get("disk_headroom_gib", 20))
    if not math.isfinite(disk_headroom_gib) or disk_headroom_gib < 0:
        raise ValueError("controller.disk_headroom_gib cannot be negative")

    return ClusterConfig(
        path=config_path,
        state_dir=state_dir,
        source_root=source_root,
        pipeline_config=pipeline_config,
        image=image,
        ssh_identity=identity,
        known_hosts=known_hosts,
        connect_timeout=max(1, int(ssh.get("connect_timeout", 10))),
        poll_seconds=max(2, int(controller.get("poll_seconds", 15))),
        max_attempts=max(1, int(controller.get("max_attempts", 3))),
        partitions_per_node=max(1, int(partitioning.get("partitions_per_node", 4))),
        group_depth=max(1, int(partitioning.get("group_depth", 2))),
        audio_extensions=normalized_extensions,
        stage_start=stage_start,
        stage_stop=stage_stop,
        shm_size=shm_size,
        nodes=tuple(nodes),
        exclude_top_level=tuple(sorted(set(normalized_excluded))),
        workspace_multiplier=workspace_multiplier,
        disk_headroom_bytes=int(disk_headroom_gib * 1024**3),
    )
