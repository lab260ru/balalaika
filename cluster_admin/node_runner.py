#!/usr/bin/env python3
"""Restricted host-side lifecycle runner for one Balalaika GPU node.

The controller invokes only the ``rpc`` command and sends a JSON request on
stdin.  Keeping request values out of the SSH command line avoids treating
cluster metadata as remote shell syntax.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
HEX_RE = re.compile(r"^[a-f0-9]{32,64}$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,255}$")
STAGE_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]{1,2})?$")
SHM_RE = re.compile(r"^[1-9][0-9]*[kKmMgG]?[bB]?$")
REMOTE_PATH_RE = re.compile(r"^/[A-Za-z0-9_./-]+$")
ALLOWED_OPERATIONS = {
    "probe",
    "prepare",
    "validate_input",
    "start",
    "status",
    "cancel",
    "cleanup",
}
STAGES = (
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


class RunnerError(RuntimeError):
    """A request is invalid or conflicts with node state."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any], mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_marker(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="ascii") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"Expected a JSON object in {path}")
    return value


def require_slug(request: dict[str, Any], name: str) -> str:
    value = request.get(name)
    if not isinstance(value, str) or not SLUG_RE.fullmatch(value):
        raise RunnerError(f"Invalid {name}")
    return value


def require_hex(request: dict[str, Any], name: str, *, sha256: bool = False) -> str:
    value = request.get(name)
    expression = SHA256_RE if sha256 else HEX_RE
    if not isinstance(value, str) or not expression.fullmatch(value):
        raise RunnerError(f"Invalid {name}")
    return value


def require_remote_path(request: dict[str, Any], name: str) -> Path:
    value = request.get(name)
    if not isinstance(value, str) or not REMOTE_PATH_RE.fullmatch(value):
        raise RunnerError(f"Invalid {name}")
    pure = PurePosixPath(value)
    if ".." in pure.parts:
        raise RunnerError(f"Invalid {name}")
    return Path(value)


def validate_relative_path(value: Any) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RunnerError("Manifest contains an invalid path")
    if any(ord(character) < 32 for character in value):
        raise RunnerError("Manifest path contains a control character")
    pure = PurePosixPath(value)
    if pure.is_absolute() or pure == PurePosixPath(".") or ".." in pure.parts:
        raise RunnerError(f"Unsafe manifest path: {value!r}")
    normalized = str(pure)
    if normalized != value or value.startswith("-"):
        raise RunnerError(f"Non-canonical manifest path: {value!r}")
    return Path(*pure.parts)


def ensure_root(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise RunnerError(f"Work root cannot be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise RunnerError(f"Work root is not a directory: {path}")
    return resolved


def ensure_directory_chain(root: Path, parts: tuple[str, ...]) -> Path:
    """Create slug-derived directories without following an injected symlink."""
    current = root
    for part in parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            current.mkdir(mode=0o700)
            info = current.lstat()
        if not stat.S_ISDIR(info.st_mode) or current.is_symlink():
            raise RunnerError(f"Unsafe directory in worker state: {current}")
    return current


def attempt_paths(request: dict[str, Any]) -> dict[str, Path]:
    work_root = ensure_root(require_remote_path(request, "work_root"))
    run_id = require_slug(request, "run_id")
    partition_id = require_slug(request, "partition_id")
    attempt_id = require_hex(request, "attempt_id")
    ordinal = request.get("attempt_ordinal")
    if not isinstance(ordinal, int) or not 1 <= ordinal <= 9999:
        raise RunnerError("Invalid attempt_ordinal")
    attempt_name = f"attempt-{ordinal:03d}-{attempt_id[:12]}"
    root = work_root / "runs" / run_id / "partitions" / partition_id / attempt_name
    return {
        "work_root": work_root,
        "root": root,
        "control": root / "control",
        "data_partial": root / "data.partial",
        "data": root / "data",
        "logs": root / "logs",
        "output": root / "output",
        "job": root / "control" / "job.json",
        "manifest": root / "control" / "manifest.json",
        "config": root / "control" / "config.yaml",
        "result": root / "result.json",
        "success": root / "_SUCCESS",
        "lock": work_root / "locks" / "gpu0.lock",
        "owner": work_root / "locks" / "gpu0.owner.json",
        "attempt_lock": work_root
        / "locks"
        / f"attempt-{run_id}-{partition_id}-{attempt_id}.lock",
    }


def run_command(
    argv: list[str], *, timeout: int = 45, check: bool = False
) -> subprocess.CompletedProcess[str]:
    try:
        process = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
            shell=False,
            env={
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            },
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RunnerError(f"Command failed to execute: {argv[0]}: {exc}") from exc
    if check and process.returncode != 0:
        message = process.stderr.strip()[-2000:]
        raise RunnerError(
            f"{Path(argv[0]).name} failed ({process.returncode}): {message}"
        )
    return process


def docker_inspect(container_name: str) -> dict[str, Any] | None:
    process = run_command(
        ["/usr/bin/docker", "container", "inspect", container_name], timeout=20
    )
    if process.returncode != 0:
        diagnostic = process.stderr.lower()
        if "no such object" in diagnostic or "no such container" in diagnostic:
            return None
        raise RunnerError(
            f"docker inspect failed ({process.returncode}): "
            f"{process.stderr.strip()[-1000:]}"
        )
    try:
        values = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise RunnerError("docker inspect returned invalid JSON") from exc
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], dict)
    ):
        raise RunnerError("docker inspect returned an unexpected value")
    return values[0]


def operation_probe(request: dict[str, Any]) -> dict[str, Any]:
    work_root = ensure_root(require_remote_path(request, "work_root"))
    image = request.get("image")
    if image is not None and (
        not isinstance(image, str) or not IMAGE_RE.fullmatch(image)
    ):
        raise RunnerError("Invalid image")
    models_value = request.get("models_root")
    models_root = (
        require_remote_path(request, "models_root")
        if models_value is not None
        else None
    )
    cache_value = request.get("cache_root")
    cache_root = (
        require_remote_path(request, "cache_root") if cache_value is not None else None
    )
    if cache_root is not None:
        ensure_root(cache_root)

    docker = run_command(["/usr/bin/docker", "version", "--format", "{{json .Server}}"])
    gpu = run_command(
        [
            "/usr/bin/nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
            "--id=0",
        ]
    )
    disk = shutil.disk_usage(work_root)
    image_id = None
    if image and docker.returncode == 0:
        inspected = run_command(
            ["/usr/bin/docker", "image", "inspect", image, "--format", "{{.Id}}"],
            timeout=20,
        )
        if inspected.returncode == 0:
            image_id = inspected.stdout.strip()

    gpu_details: dict[str, Any] = {"index": 0}
    if gpu.returncode == 0:
        fields = [field.strip() for field in gpu.stdout.strip().split(",")]
        if len(fields) == 6:
            gpu_details.update(
                {
                    "name": fields[1],
                    "uuid": fields[2],
                    "memory_total_mib": int(fields[3]),
                    "memory_used_mib": int(fields[4]),
                    "utilization_percent": int(fields[5]),
                }
            )
    return {
        "ok": docker.returncode == 0 and gpu.returncode == 0,
        "hostname": socket.gethostname(),
        "docker_ok": docker.returncode == 0,
        "docker_error": docker.stderr.strip()[-500:] if docker.returncode else None,
        "gpu_ok": gpu.returncode == 0,
        "gpu_error": gpu.stderr.strip()[-500:] if gpu.returncode else None,
        "gpu": gpu_details,
        "disk": {"total_bytes": disk.total, "free_bytes": disk.free},
        "image": image,
        "image_id": image_id,
        "models_ok": bool(
            models_root
            and models_root.is_dir()
            and not models_root.is_symlink()
            and next(models_root.iterdir(), None) is not None
        ),
        "models_root": str(models_root) if models_root else None,
        "cache_root": str(cache_root) if cache_root else None,
        "runner_schema": SCHEMA_VERSION,
    }


def operation_prepare(request: dict[str, Any]) -> dict[str, Any]:
    paths = attempt_paths(request)
    token = require_hex(request, "fencing_token")
    manifest_sha256 = require_hex(request, "manifest_sha256", sha256=True)
    config_sha256 = require_hex(request, "config_sha256", sha256=True)
    ensure_directory_chain(
        paths["work_root"],
        (
            "runs",
            request["run_id"],
            "partitions",
            request["partition_id"],
            paths["root"].name,
        ),
    )
    for name in ("control", "logs", "output"):
        ensure_directory_chain(paths["root"], (name,))

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "run_id": request["run_id"],
        "partition_id": request["partition_id"],
        "attempt_id": request["attempt_id"],
        "attempt_ordinal": request["attempt_ordinal"],
        "fencing_token": token,
        "manifest_sha256": manifest_sha256,
        "config_sha256": config_sha256,
        "state": "STAGING",
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    if paths["job"].exists():
        existing = read_json(paths["job"])
        immutable = (
            "run_id",
            "partition_id",
            "attempt_id",
            "attempt_ordinal",
            "fencing_token",
            "manifest_sha256",
            "config_sha256",
        )
        if any(existing.get(key) != metadata[key] for key in immutable):
            raise RunnerError("Attempt directory already contains a different job")
        metadata = existing
    else:
        atomic_json(paths["job"], metadata)
    if metadata.get("state") == "STAGING":
        ensure_directory_chain(paths["root"], ("data.partial",))
    return {
        "ok": True,
        "state": metadata["state"],
        "attempt_root": str(paths["root"]),
        "data_partial": str(paths["data_partial"]),
        "control_root": str(paths["control"]),
    }


def _validate_manifest(
    paths: dict[str, Path], metadata: dict[str, Any]
) -> dict[str, Any]:
    try:
        data_info = paths["data_partial"].lstat()
    except FileNotFoundError as exc:
        raise RunnerError("Partial data directory is missing") from exc
    if not stat.S_ISDIR(data_info.st_mode) or paths["data_partial"].is_symlink():
        raise RunnerError("Partial data directory is unsafe")
    for control_file in (paths["manifest"], paths["config"]):
        try:
            control_info = control_file.lstat()
        except FileNotFoundError as exc:
            raise RunnerError(f"Control file is missing: {control_file.name}") from exc
        if not stat.S_ISREG(control_info.st_mode) or control_file.is_symlink():
            raise RunnerError(f"Control file is unsafe: {control_file.name}")
    if sha256_file(paths["manifest"]) != metadata["manifest_sha256"]:
        raise RunnerError("Manifest SHA256 does not match the planned value")
    if sha256_file(paths["config"]) != metadata["config_sha256"]:
        raise RunnerError("Pipeline config SHA256 does not match the planned value")
    manifest = read_json(paths["manifest"])
    files = manifest.get("files")
    if not isinstance(files, list):
        raise RunnerError("Manifest files must be a list")
    if manifest.get("files_total") != len(files):
        raise RunnerError("Manifest files_total does not match files")

    seen: set[str] = set()
    total_bytes = 0
    for item in files:
        if not isinstance(item, dict):
            raise RunnerError("Manifest file entry must be an object")
        relative = validate_relative_path(item.get("path"))
        relative_text = relative.as_posix()
        if relative_text in seen:
            raise RunnerError(f"Duplicate manifest path: {relative_text}")
        seen.add(relative_text)
        expected_size = item.get("size")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise RunnerError(f"Invalid size for {relative_text}")
        current = paths["data_partial"]
        for part in relative.parts[:-1]:
            current /= part
            try:
                directory_info = current.lstat()
            except FileNotFoundError as exc:
                raise RunnerError(
                    f"Input directory is missing: {relative_text}"
                ) from exc
            if not stat.S_ISDIR(directory_info.st_mode) or current.is_symlink():
                raise RunnerError(f"Input directory is unsafe: {relative_text}")
        path = paths["data_partial"] / relative
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise RunnerError(f"Input file is missing: {relative_text}") from exc
        if not stat.S_ISREG(info.st_mode) or path.is_symlink():
            raise RunnerError(f"Input is not a regular file: {relative_text}")
        if info.st_size != expected_size:
            raise RunnerError(
                f"Input size mismatch for {relative_text}: {info.st_size} != {expected_size}"
            )
        expected_sha = item.get("sha256")
        if expected_sha is not None:
            if not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(
                expected_sha
            ):
                raise RunnerError(f"Invalid SHA256 for {relative_text}")
            if sha256_file(path) != expected_sha:
                raise RunnerError(f"Input SHA256 mismatch for {relative_text}")
        total_bytes += info.st_size
    if manifest.get("input_bytes") != total_bytes:
        raise RunnerError("Manifest input_bytes does not match transferred files")

    state_fragment = manifest.get("state_fragment")
    if state_fragment:
        state_name = validate_relative_path(state_fragment)
        state_path = paths["data_partial"] / state_name
        if not state_path.is_file() or state_path.is_symlink():
            raise RunnerError("State fragment is missing")
        state_metadata = manifest.get("state_fragment_metadata")
        if not isinstance(state_metadata, dict):
            raise RunnerError("State fragment metadata is missing")
        if state_metadata.get("path") != state_fragment:
            raise RunnerError("State fragment path does not match its metadata")
        expected_bytes = state_metadata.get("bytes")
        expected_sha256 = state_metadata.get("sha256")
        if (
            not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or state_path.stat().st_size != expected_bytes
        ):
            raise RunnerError("State fragment size does not match")
        if (
            not isinstance(expected_sha256, str)
            or not SHA256_RE.fullmatch(expected_sha256)
            or sha256_file(state_path) != expected_sha256
        ):
            raise RunnerError("State fragment SHA256 does not match")
    return {"files_total": len(files), "input_bytes": total_bytes}


def operation_validate_input(request: dict[str, Any]) -> dict[str, Any]:
    paths = attempt_paths(request)
    token = require_hex(request, "fencing_token")
    metadata = read_json(paths["job"])
    if metadata.get("fencing_token") != token:
        raise RunnerError("Fencing token does not match this attempt")
    if metadata.get("state") in {"CANCELLED", "FAILED"}:
        raise RunnerError(f"Attempt is already {metadata['state']}")
    if paths["data"].exists():
        if not paths["data"].is_dir() or paths["data"].is_symlink():
            raise RunnerError("Committed data path is unsafe")
        if metadata.get("state") not in {"READY", "STARTING", "RUNNING", "COMPLETED"}:
            raise RunnerError("Committed input conflicts with job state")
        return {
            "ok": True,
            "state": metadata["state"],
            "attempt_root": str(paths["root"]),
            "data_root": str(paths["data"]),
        }
    if metadata.get("state") != "STAGING":
        raise RunnerError(f"Cannot validate input in state {metadata.get('state')}")
    summary = _validate_manifest(paths, metadata)
    if paths["data"].exists():
        raise RunnerError("Final data path already exists")
    os.rename(paths["data_partial"], paths["data"])
    metadata.update(summary)
    metadata.update({"state": "READY", "updated_at": utc_now()})
    atomic_json(paths["job"], metadata)
    return {
        "ok": True,
        "state": "READY",
        "attempt_root": str(paths["root"]),
        "data_root": str(paths["data"]),
        **summary,
    }


def _validate_stage(value: Any, name: str) -> str:
    if not isinstance(value, str) or not STAGE_RE.fullmatch(value):
        raise RunnerError(f"Invalid {name}")
    if value not in STAGES:
        raise RunnerError(f"Unsupported {name}: {value}")
    return value


def _container_name(metadata: dict[str, Any]) -> str:
    raw = (
        f"balalaika-{metadata['run_id']}-{metadata['partition_id']}-"
        f"{metadata['attempt_id'][:12]}"
    )
    return raw[:180]


def _container_labels(inspect: dict[str, Any]) -> dict[str, str]:
    config = inspect.get("Config")
    if not isinstance(config, dict):
        return {}
    labels = config.get("Labels")
    return labels if isinstance(labels, dict) else {}


def _owner_is_running(owner: dict[str, Any]) -> bool:
    container_name = owner.get("container_name")
    if not isinstance(container_name, str):
        return False
    inspect = docker_inspect(container_name)
    return bool(inspect and inspect.get("State", {}).get("Running"))


def operation_start(request: dict[str, Any]) -> dict[str, Any]:
    paths = attempt_paths(request)
    token = require_hex(request, "fencing_token")
    image = request.get("image")
    if not isinstance(image, str) or not IMAGE_RE.fullmatch(image):
        raise RunnerError("Invalid image")
    stage_start = _validate_stage(request.get("stage_start"), "stage_start")
    stage_stop = _validate_stage(request.get("stage_stop"), "stage_stop")
    if STAGES.index(stage_start) > STAGES.index(stage_stop):
        raise RunnerError("stage_start must not be after stage_stop")
    shm_size = request.get("shm_size", "8g")
    if not isinstance(shm_size, str) or not SHM_RE.fullmatch(shm_size):
        raise RunnerError("Invalid shm_size")
    models_root = ensure_root(require_remote_path(request, "models_root"))
    cache_root = ensure_root(require_remote_path(request, "cache_root"))
    env_file_value = request.get("env_file")
    env_file = None
    if env_file_value is not None:
        env_file = require_remote_path(request, "env_file")
        if not env_file.is_file() or env_file.is_symlink():
            raise RunnerError("env_file must be a regular file")

    metadata = read_json(paths["job"])
    if metadata.get("fencing_token") != token:
        raise RunnerError("Fencing token does not match this attempt")
    if not paths["data"].is_dir() or metadata.get("state") not in {
        "READY",
        "STARTING",
        "RUNNING",
        "COMPLETED",
    }:
        raise RunnerError("Input has not been committed")
    if (
        metadata.get("state") == "COMPLETED"
        and paths["success"].is_file()
        and paths["result"].is_file()
    ):
        return read_json(paths["result"])

    container_name = _container_name(metadata)
    spec = {
        "attempt_id": metadata["attempt_id"],
        "config_sha256": metadata["config_sha256"],
        "fencing_token": token,
        "image": image,
        "manifest_sha256": metadata["manifest_sha256"],
        "shm_size": shm_size,
        "stage_start": stage_start,
        "stage_stop": stage_stop,
    }
    spec_sha256 = hashlib.sha256(canonical_json(spec)).hexdigest()
    ensure_directory_chain(paths["work_root"], ("locks",))
    with paths["lock"].open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        existing = docker_inspect(container_name)
        if existing:
            labels = _container_labels(existing)
            if labels.get("io.balalaika.spec-sha256") != spec_sha256:
                raise RunnerError("Container name exists with a different job spec")
            return _status_from_inspect(paths, metadata, existing)

        if paths["owner"].exists():
            owner = read_json(paths["owner"])
            if owner.get("attempt_id") != metadata["attempt_id"] and _owner_is_running(
                owner
            ):
                raise RunnerError(
                    f"GPU 0 is reserved by attempt {owner.get('attempt_id', 'unknown')}"
                )
            paths["owner"].unlink(missing_ok=True)

        uid = os.getuid()
        gid = os.getgid()
        docker_argv = [
            "/usr/bin/docker",
            "run",
            "--detach",
            "--name",
            container_name,
            "--gpus",
            "device=0",
            "--user",
            f"{uid}:{gid}",
            "--shm-size",
            shm_size,
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--log-opt",
            "max-size=100m",
            "--log-opt",
            "max-file=3",
            "--label",
            f"io.balalaika.spec-sha256={spec_sha256}",
            "--label",
            f"io.balalaika.attempt-id={metadata['attempt_id']}",
            "--label",
            f"io.balalaika.fencing-token={token}",
            "--env",
            "CUDA_VISIBLE_DEVICES=0",
            "--env",
            "BALALAIKA_CONFIG_PATH=/config/config.yaml",
            "--env",
            "BALALAIKA_DATA_ROOT=/data",
            "--env",
            "BALALAIKA_MODELS_ROOT=/models",
            "--env",
            "BALALAIKA_CACHE_ROOT=/cache/balalaika",
            "--env",
            "BALALAIKA_TRT_CACHE_PATH=/cache/trt",
            "--env",
            "BALALAIKA_LOG_DIR=/logs",
            "--env",
            "BALALAIKA_OUTPUT_ROOT=/output",
            "--mount",
            f"type=bind,source={paths['data']},target=/data",
            "--mount",
            f"type=bind,source={paths['config']},target=/config/config.yaml,readonly",
            "--mount",
            f"type=bind,source={models_root},target=/models,readonly",
            "--mount",
            f"type=bind,source={cache_root},target=/cache",
            "--mount",
            f"type=bind,source={paths['logs']},target=/logs",
            "--mount",
            f"type=bind,source={paths['output']},target=/output",
        ]
        if env_file:
            docker_argv.extend(["--env-file", str(env_file)])
        docker_argv.extend(
            [
                image,
                "pipeline",
                "--stage",
                stage_start,
                "--stop_stage",
                stage_stop,
                "--strict",
            ]
        )
        metadata.update(
            {
                "state": "STARTING",
                "updated_at": utc_now(),
                "container_name": container_name,
                "spec_sha256": spec_sha256,
                "stage_start": stage_start,
                "stage_stop": stage_stop,
                "image": image,
            }
        )
        atomic_json(paths["job"], metadata)
        process = run_command(docker_argv, timeout=120, check=True)
        container_id = process.stdout.strip()
        owner = {
            "attempt_id": metadata["attempt_id"],
            "container_name": container_name,
            "fencing_token": token,
            "spec_sha256": spec_sha256,
            "updated_at": utc_now(),
        }
        atomic_json(paths["owner"], owner)
        metadata.update(
            {
                "state": "RUNNING",
                "container_id": container_id,
                "started_at": metadata.get("started_at") or utc_now(),
                "updated_at": utc_now(),
            }
        )
        atomic_json(paths["job"], metadata)
        return {
            "ok": True,
            "state": "RUNNING",
            "container_name": container_name,
            "container_id": container_id,
            "attempt_root": str(paths["root"]),
        }


def _stage_progress(paths: dict[str, Path], metadata: dict[str, Any]) -> dict[str, Any]:
    start = metadata.get("stage_start")
    stop = metadata.get("stage_stop")
    if start not in STAGES or stop not in STAGES:
        return {"current_stage": None, "progress_percent": 0.0, "files_processed": 0}
    start_index = STAGES.index(start)
    stop_index = STAGES.index(stop) + 1
    selected = STAGES[start_index:stop_index]
    completed: list[str] = []
    processed = 0
    errors = 0
    for stage in selected:
        status_path = paths["logs"] / f"stage_{stage}_status.json"
        if not status_path.is_file():
            continue
        try:
            status_value = read_json(status_path)
        except RunnerError:
            continue
        completed.append(stage)
        processed = max(
            processed,
            int(status_value.get("processed", status_value.get("files_out", 0)) or 0),
        )
        errors += int(status_value.get("errors", 0) or 0)
    completed_count = len(completed)
    current_stage = completed[-1] if completed else selected[0]
    if completed_count < len(selected):
        current_stage = selected[completed_count]
    return {
        "current_stage": current_stage,
        "progress_percent": round(100.0 * completed_count / max(len(selected), 1), 2),
        "files_processed": processed,
        "stage_errors": errors,
    }


def _capture_container_log(paths: dict[str, Path], container_name: str) -> None:
    process = run_command(
        [
            "/usr/bin/docker",
            "logs",
            "--timestamps",
            "--tail",
            "10000",
            container_name,
        ],
        timeout=60,
    )
    log_path = paths["logs"] / "container.log"
    temporary = log_path.with_suffix(".log.tmp")
    with temporary.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write(process.stdout)
        if process.stderr:
            handle.write("\n[stderr]\n")
            handle.write(process.stderr)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, log_path)


def _status_from_inspect(
    paths: dict[str, Path], metadata: dict[str, Any], inspect: dict[str, Any]
) -> dict[str, Any]:
    state = inspect.get("State") if isinstance(inspect.get("State"), dict) else {}
    running = bool(state.get("Running"))
    exit_code = state.get("ExitCode")
    container_name = metadata.get("container_name") or _container_name(metadata)
    progress = _stage_progress(paths, metadata)
    result = {
        "ok": True,
        "attempt_id": metadata["attempt_id"],
        "attempt_root": str(paths["root"]),
        "container_name": container_name,
        "started_at": state.get("StartedAt") or metadata.get("started_at"),
        "updated_at": utc_now(),
        **progress,
    }
    if running:
        metadata.update({"state": "RUNNING", "updated_at": utc_now()})
        atomic_json(paths["job"], metadata)
        result["state"] = "RUNNING"
        return result

    if not isinstance(exit_code, int):
        result.update({"state": "UNKNOWN", "error": "Container state has no exit code"})
        return result
    _capture_container_log(paths, container_name)
    metadata.update(
        {
            "state": "COMPLETED" if exit_code == 0 else "FAILED",
            "exit_code": exit_code,
            "finished_at": state.get("FinishedAt") or utc_now(),
            "updated_at": utc_now(),
        }
    )
    atomic_json(paths["job"], metadata)
    result.update(
        {
            "state": metadata["state"],
            "exit_code": exit_code,
            "finished_at": metadata["finished_at"],
            "manifest_sha256": metadata["manifest_sha256"],
            "config_sha256": metadata["config_sha256"],
            "fencing_token": metadata["fencing_token"],
        }
    )
    if exit_code == 0:
        result["progress_percent"] = 100.0
        atomic_json(paths["result"], result)
        atomic_marker(paths["success"], f"{metadata['fencing_token']}\n")
    else:
        result["error"] = f"Pipeline container exited with code {exit_code}"
    return result


def operation_status(request: dict[str, Any]) -> dict[str, Any]:
    paths = attempt_paths(request)
    token = require_hex(request, "fencing_token")
    if not paths["job"].exists():
        return {
            "ok": True,
            "state": "MISSING",
            "attempt_id": request["attempt_id"],
            "attempt_root": str(paths["root"]),
        }
    metadata = read_json(paths["job"])
    if metadata.get("fencing_token") != token:
        raise RunnerError("Fencing token does not match this attempt")
    if metadata.get("state") == "CANCELLED":
        return {
            "ok": True,
            "state": "CANCELLED",
            "attempt_id": metadata["attempt_id"],
            "attempt_root": str(paths["root"]),
        }
    if (
        metadata.get("state") == "COMPLETED"
        and paths["success"].is_file()
        and paths["result"].is_file()
    ):
        return read_json(paths["result"])
    container_name = metadata.get("container_name")
    if not isinstance(container_name, str):
        return {
            "ok": True,
            "state": metadata.get("state", "UNKNOWN"),
            "attempt_id": metadata["attempt_id"],
            "attempt_root": str(paths["root"]),
            **_stage_progress(paths, metadata),
        }
    inspect = docker_inspect(container_name)
    if inspect is None:
        if paths["success"].is_file() and paths["result"].is_file():
            return read_json(paths["result"])
        return {
            "ok": False,
            "state": "UNKNOWN",
            "attempt_id": metadata["attempt_id"],
            "attempt_root": str(paths["root"]),
            "error": "Container no longer exists and no success marker was found",
        }
    return _status_from_inspect(paths, metadata, inspect)


def operation_cancel(request: dict[str, Any]) -> dict[str, Any]:
    paths = attempt_paths(request)
    token = require_hex(request, "fencing_token")
    if not paths["job"].exists():
        return {
            "ok": True,
            "state": "CANCELLED",
            "attempt_id": request["attempt_id"],
        }
    metadata = read_json(paths["job"])
    if metadata.get("fencing_token") != token:
        raise RunnerError("Fencing token does not match this attempt")
    if metadata.get("state") == "CANCELLED":
        return {
            "ok": True,
            "state": "CANCELLED",
            "attempt_id": metadata["attempt_id"],
        }
    container_name = metadata.get("container_name")
    if metadata.get("state") == "COMPLETED":
        if paths["result"].is_file() and paths["success"].is_file():
            return read_json(paths["result"])
        raise RunnerError("Completed attempt is missing its result marker")
    if isinstance(container_name, str) and docker_inspect(container_name):
        run_command(
            ["/usr/bin/docker", "stop", "--time", "30", container_name],
            timeout=45,
            check=True,
        )
    metadata.update({"state": "CANCELLED", "updated_at": utc_now()})
    atomic_json(paths["job"], metadata)
    return {"ok": True, "state": "CANCELLED", "attempt_id": metadata["attempt_id"]}


def operation_cleanup(request: dict[str, Any]) -> dict[str, Any]:
    paths = attempt_paths(request)
    token = require_hex(request, "fencing_token")
    metadata = read_json(paths["job"])
    if metadata.get("fencing_token") != token:
        raise RunnerError("Fencing token does not match this attempt")
    container_name = metadata.get("container_name")
    if isinstance(container_name, str):
        inspect = docker_inspect(container_name)
        if inspect and inspect.get("State", {}).get("Running"):
            raise RunnerError("Cannot clean up a running attempt")
        if inspect:
            run_command(
                ["/usr/bin/docker", "rm", container_name], timeout=30, check=True
            )
    if request.get("remove_data") is True:
        shutil.rmtree(paths["root"])
    return {"ok": True, "state": "CLEANED", "attempt_id": metadata["attempt_id"]}


def dispatch(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise RunnerError("RPC request must be a JSON object")
    operation = request.get("operation")
    if operation not in ALLOWED_OPERATIONS:
        raise RunnerError("Unsupported operation")
    handlers = {
        "probe": operation_probe,
        "prepare": operation_prepare,
        "validate_input": operation_validate_input,
        "start": operation_start,
        "status": operation_status,
        "cancel": operation_cancel,
        "cleanup": operation_cleanup,
    }
    if operation == "probe":
        return handlers[operation](request)
    paths = attempt_paths(request)
    ensure_directory_chain(paths["work_root"], ("locks",))
    with paths["attempt_lock"].open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        return handlers[operation](request)


def rpc_main() -> int:
    try:
        raw = sys.stdin.buffer.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise RunnerError("RPC request exceeds 1 MiB")
        request = json.loads(raw)
        response = dispatch(request)
        response.setdefault("ok", True)
        sys.stdout.buffer.write(canonical_json(response) + b"\n")
        return 0
    except (RunnerError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        response = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        sys.stdout.buffer.write(canonical_json(response) + b"\n")
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("rpc",))
    args = parser.parse_args()
    if args.command == "rpc":
        return rpc_main()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
