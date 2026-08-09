#!/usr/bin/env python3
"""Restricted host-side lifecycle runner for one Balalaika GPU node.

The controller invokes fixed subcommands and sends requests on stdin. Keeping
request values out of the SSH command line avoids treating cluster metadata as
remote shell syntax.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

SCHEMA_VERSION = 2
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
HEX_RE = re.compile(r"^[a-f0-9]{32,64}$")
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,255}$")
STAGE_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]{1,2})?$")
SHM_RE = re.compile(r"^[1-9][0-9]*[kKmMgG]?[bB]?$")
REMOTE_PATH_RE = re.compile(r"^/[A-Za-z0-9_./-]+$")
GPU_UUID_RE = re.compile(r"^(?:GPU|MIG)-[A-Za-z0-9._-]{8,128}$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
DIRECT_ENV_DENY = {
    "BASH_ENV",
    "CDPATH",
    "ENV",
    "GLOBIGNORE",
    "LD_AUDIT",
    "LD_PRELOAD",
    "PYTHONHOME",
    "PYTHONPATH",
    "SHELLOPTS",
}
DIRECT_AMBIENT_ENV = {
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LD_LIBRARY_PATH",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
    "TZ",
}
RUNTIME_MODES = ("docker", "direct")
STREAM_PROTOCOL_VERSION = 1
STREAM_MAGIC = b"BLKSTRM1"
STREAM_HEADER_LIMIT = 256 * 1024 * 1024
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


def require_runtime(request: dict[str, Any]) -> str:
    value = request.get("runtime", "docker")
    if value not in RUNTIME_MODES:
        raise RunnerError(f"Invalid runtime: {value!r}")
    return str(value)


def require_gpu_devices(request: dict[str, Any]) -> list[int]:
    value = request.get("gpu_devices", [0])
    if not isinstance(value, list) or not value:
        raise RunnerError("gpu_devices must be a non-empty list")
    devices: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255:
            raise RunnerError("gpu_devices must contain indices from 0 to 255")
        if item in devices:
            raise RunnerError("gpu_devices contains a duplicate index")
        devices.append(item)
    return devices


def require_gpu_uuids(request: dict[str, Any], count: int) -> list[str]:
    value = request.get("gpu_uuids")
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(not isinstance(item, str) or not GPU_UUID_RE.fullmatch(item) for item in value)
        or len(set(value)) != len(value)
    ):
        raise RunnerError("gpu_uuids must contain one unique UUID per GPU device")
    return list(value)


def require_global_rank(request: dict[str, Any]) -> int:
    value = request.get("global_rank")
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**31:
        raise RunnerError("global_rank must be an integer from 0 to 2147483647")
    return value


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
        "runtime_config": root / "control" / "runtime.config.yaml",
        "direct_spec": root / "control" / "direct-spec.json",
        "direct_process": root / "control" / "direct-process.json",
        "direct_exit": root / "control" / "direct-exit.json",
        "direct_log": root / "logs" / "direct.log",
        "result": root / "result.json",
        "success": root / "_SUCCESS",
        "lock": work_root / "locks" / "gpu0.lock",
        "owner": work_root / "locks" / "gpu0.owner.json",
        "attempt_lock": work_root
        / "locks"
        / f"attempt-{run_id}-{partition_id}-{attempt_id}.lock",
    }


def _stream_read_exact(handle: BinaryIO, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            raise RunnerError("Truncated stream protocol frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _stream_read_header(handle: BinaryIO) -> dict[str, Any]:
    if _stream_read_exact(handle, len(STREAM_MAGIC)) != STREAM_MAGIC:
        raise RunnerError("Invalid stream protocol magic")
    length = struct.unpack("!I", _stream_read_exact(handle, 4))[0]
    if length < 2 or length > STREAM_HEADER_LIMIT:
        raise RunnerError("Invalid stream protocol header length")
    try:
        value = json.loads(_stream_read_exact(handle, length))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerError(f"Invalid stream protocol header: {exc}") from exc
    if not isinstance(value, dict):
        raise RunnerError("Stream protocol header must be a JSON object")
    return value


def _stream_write_header(handle: BinaryIO, value: dict[str, Any]) -> None:
    payload = canonical_json(value)
    if len(payload) > STREAM_HEADER_LIMIT:
        raise RunnerError("Stream protocol header is too large")
    handle.write(STREAM_MAGIC)
    handle.write(struct.pack("!I", len(payload)))
    handle.write(payload)


def _stream_attempt_paths(value: Any) -> tuple[dict[str, Path], dict[str, Any]]:
    root = require_remote_path({"attempt_root": value}, "attempt_root")
    if str(PurePosixPath(str(root))) != str(root) or len(root.parents) < 5:
        raise RunnerError("Invalid attempt_root")
    job_path = root / "control" / "job.json"
    try:
        job_info = job_path.lstat()
    except FileNotFoundError as exc:
        raise RunnerError("Stream target has no prepared job") from exc
    if not stat.S_ISREG(job_info.st_mode) or job_path.is_symlink():
        raise RunnerError("Stream target job metadata is unsafe")
    metadata = read_json(job_path)
    request = {**metadata, "work_root": str(root.parents[4])}
    paths = attempt_paths(request)
    if paths["root"] != root or paths["job"] != job_path:
        raise RunnerError("Stream target does not match its prepared job")
    verified_root = ensure_directory_chain(
        paths["work_root"],
        (
            "runs",
            metadata["run_id"],
            "partitions",
            metadata["partition_id"],
            root.name,
        ),
    )
    if verified_root != root:
        raise RunnerError("Stream target directory chain does not match its job")
    for directory in (paths["root"], paths["control"]):
        try:
            info = directory.lstat()
        except FileNotFoundError as exc:
            raise RunnerError("Stream target directory is missing") from exc
        if not stat.S_ISDIR(info.st_mode) or directory.is_symlink():
            raise RunnerError("Stream target directory is unsafe")
    return paths, metadata


def _stream_manifest(
    paths: dict[str, Path], metadata: dict[str, Any]
) -> dict[str, Any]:
    try:
        info = paths["manifest"].lstat()
    except FileNotFoundError as exc:
        raise RunnerError("Immutable manifest has not been transferred") from exc
    if not stat.S_ISREG(info.st_mode) or paths["manifest"].is_symlink():
        raise RunnerError("Immutable manifest is unsafe")
    if sha256_file(paths["manifest"]) != metadata.get("manifest_sha256"):
        raise RunnerError("Immutable manifest SHA256 does not match the prepared job")
    return read_json(paths["manifest"])


def _stream_data_entries(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, list) or manifest.get("files_total") != len(files):
        raise RunnerError("Immutable manifest contains an invalid files list")
    expected: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for item in files:
        if not isinstance(item, dict):
            raise RunnerError("Immutable manifest file entry must be an object")
        name = validate_relative_path(item.get("path")).as_posix()
        if name in expected:
            raise RunnerError(f"Duplicate manifest path: {name}")
        size = item.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RunnerError(f"Invalid size for {name}")
        digest = item.get("sha256")
        if digest is not None and (
            not isinstance(digest, str) or not SHA256_RE.fullmatch(digest)
        ):
            raise RunnerError(f"Invalid SHA256 for {name}")
        expected[name] = {"size": size, "sha256": digest}
        total_bytes += size
    if manifest.get("input_bytes") != total_bytes:
        raise RunnerError("Immutable manifest input_bytes does not match its files")
    return expected


def _stream_state_entries(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    state_name = manifest.get("state_fragment")
    state_metadata = manifest.get("state_fragment_metadata")
    if not state_name or not isinstance(state_metadata, dict):
        raise RunnerError("Immutable manifest has no state fragment")
    name = validate_relative_path(state_name).as_posix()
    size = state_metadata.get("bytes")
    digest = state_metadata.get("sha256")
    if (
        state_metadata.get("path") != name
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(digest, str)
        or not SHA256_RE.fullmatch(digest)
    ):
        raise RunnerError("Immutable state fragment metadata is invalid")
    return {name: {"size": size, "sha256": digest}}


def _stream_push_target(
    paths: dict[str, Path], metadata: dict[str, Any], role: Any
) -> tuple[Path, dict[str, dict[str, Any]]]:
    if metadata.get("state") != "STAGING":
        raise RunnerError("Stream input is accepted only while the job is STAGING")
    if role == "manifest":
        return paths["control"], {
            "manifest.json": {
                "size": None,
                "sha256": metadata.get("manifest_sha256"),
            }
        }
    if role == "config":
        return paths["control"], {
            "config.yaml": {
                "size": None,
                "sha256": metadata.get("config_sha256"),
            }
        }
    manifest = _stream_manifest(paths, metadata)
    if role == "data":
        return paths["data_partial"], _stream_data_entries(manifest)
    if role == "state":
        return paths["data_partial"], _stream_state_entries(manifest)
    raise RunnerError("Unsupported stream push role")


def _stream_safe_directory(root: Path, relative: Path) -> Path:
    try:
        root_info = root.lstat()
    except FileNotFoundError as exc:
        raise RunnerError("Stream destination root is missing") from exc
    if not stat.S_ISDIR(root_info.st_mode) or root.is_symlink():
        raise RunnerError("Stream destination root is unsafe")
    return ensure_directory_chain(root, tuple(relative.parts[:-1]))


def _stream_receive_tar(
    handle: BinaryIO, root: Path, expected: dict[str, dict[str, Any]]
) -> None:
    seen: set[str] = set()
    try:
        with tarfile.open(fileobj=handle, mode="r|") as archive:
            for member in archive:
                name = validate_relative_path(member.name).as_posix()
                if not member.isfile() or member.sparse is not None:
                    raise RunnerError(f"Stream member is not a regular file: {name}")
                if name in seen:
                    raise RunnerError(f"Duplicate stream member: {name}")
                entry = expected.get(name)
                if entry is None:
                    raise RunnerError(f"Unexpected stream member: {name}")
                expected_size = entry.get("size")
                if expected_size is not None and member.size != expected_size:
                    raise RunnerError(f"Stream member size does not match: {name}")
                source = archive.extractfile(member)
                if source is None:
                    raise RunnerError(f"Cannot read stream member: {name}")
                relative = Path(*PurePosixPath(name).parts)
                parent = _stream_safe_directory(root, relative)
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{relative.name}.stream-", dir=parent
                )
                digest = hashlib.sha256()
                try:
                    with os.fdopen(descriptor, "wb") as destination:
                        remaining = member.size
                        while remaining:
                            chunk = source.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise RunnerError(f"Truncated stream member: {name}")
                            destination.write(chunk)
                            digest.update(chunk)
                            remaining -= len(chunk)
                        destination.flush()
                        os.fsync(destination.fileno())
                    expected_digest = entry.get("sha256")
                    if (
                        expected_digest is not None
                        and digest.hexdigest() != expected_digest
                    ):
                        raise RunnerError(
                            f"Stream member SHA256 does not match: {name}"
                        )
                    target = root.joinpath(*relative.parts)
                    os.replace(temporary_name, target)
                    directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                except Exception:
                    try:
                        os.unlink(temporary_name)
                    except FileNotFoundError:
                        pass
                    raise
                seen.add(name)
    except (tarfile.TarError, OSError) as exc:
        raise RunnerError(f"Invalid tar stream: {exc}") from exc
    missing = set(expected) - seen
    if missing:
        raise RunnerError(f"Stream is missing expected member: {min(missing)!r}")


def stream_push(handle: BinaryIO) -> dict[str, Any]:
    header = _stream_read_header(handle)
    if set(header) != {"protocol", "attempt_root", "role"}:
        raise RunnerError("Invalid stream push header fields")
    if header.get("protocol") != STREAM_PROTOCOL_VERSION:
        raise RunnerError("Unsupported stream protocol version")
    paths, metadata = _stream_attempt_paths(header.get("attempt_root"))
    root, expected = _stream_push_target(paths, metadata, header.get("role"))
    _stream_receive_tar(handle, root, expected)
    return {"ok": True, "files_total": len(expected)}


def _stream_walk_regular_files(root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        try:
            children = sorted(
                directory.iterdir(), key=lambda item: os.fsencode(item.name)
            )
        except OSError as exc:
            raise RunnerError(f"Cannot enumerate stream result: {exc}") from exc
        for child in children:
            try:
                info = child.lstat()
            except OSError as exc:
                raise RunnerError(f"Cannot inspect stream result: {exc}") from exc
            relative = child.relative_to(root)
            name = validate_relative_path(relative.as_posix()).as_posix()
            if stat.S_ISDIR(info.st_mode) and not child.is_symlink():
                visit(child)
            elif stat.S_ISREG(info.st_mode) and not child.is_symlink():
                result.append(
                    {
                        "path": child,
                        "name": name,
                        "size": info.st_size,
                        "mtime_ns": info.st_mtime_ns,
                        "device": info.st_dev,
                        "inode": info.st_ino,
                    }
                )
            else:
                raise RunnerError(f"Result contains an unsafe filesystem entry: {name}")

    visit(root)
    return result


def _stream_open_unchanged(entry: dict[str, Any]) -> tuple[BinaryIO, os.stat_result]:
    path = entry["path"]
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        info = os.fstat(descriptor)
    except OSError as exc:
        raise RunnerError(f"Cannot open stream result file {path}: {exc}") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_size != entry["size"]
        or info.st_mtime_ns != entry["mtime_ns"]
        or info.st_dev != entry["device"]
        or info.st_ino != entry["inode"]
    ):
        os.close(descriptor)
        raise RunnerError(f"Stream result file changed while collecting: {path}")
    return os.fdopen(descriptor, "rb"), info


def stream_pull(request_handle: BinaryIO, output_handle: BinaryIO) -> dict[str, Any]:
    header = _stream_read_header(request_handle)
    if set(header) != {"protocol", "attempt_root"}:
        raise RunnerError("Invalid stream pull header fields")
    if header.get("protocol") != STREAM_PROTOCOL_VERSION:
        raise RunnerError("Unsupported stream protocol version")
    paths, metadata = _stream_attempt_paths(header.get("attempt_root"))
    if metadata.get("state") != "COMPLETED":
        raise RunnerError("Stream result is available only for a COMPLETED job")
    for marker in (paths["result"], paths["success"]):
        try:
            info = marker.lstat()
        except FileNotFoundError as exc:
            raise RunnerError("Completed stream result is missing its markers") from exc
        if not stat.S_ISREG(info.st_mode) or marker.is_symlink():
            raise RunnerError("Completed stream result marker is unsafe")
    entries = _stream_walk_regular_files(paths["root"])
    response = {
        "protocol": STREAM_PROTOCOL_VERSION,
        "attempt_id": metadata.get("attempt_id"),
        "files": [
            {"path": entry["name"], "size": entry["size"]} for entry in entries
        ],
    }
    _stream_write_header(output_handle, response)
    with tarfile.open(
        fileobj=output_handle, mode="w|", format=tarfile.PAX_FORMAT
    ) as archive:
        for entry in entries:
            source, before = _stream_open_unchanged(entry)
            try:
                member = tarfile.TarInfo(entry["name"])
                member.size = entry["size"]
                member.mtime = entry["mtime_ns"] // 1_000_000_000
                member.mode = 0o600
                archive.addfile(member, source)
                after = os.fstat(source.fileno())
            finally:
                source.close()
            if (
                after.st_size != before.st_size
                or after.st_mtime_ns != before.st_mtime_ns
                or after.st_ino != before.st_ino
                or after.st_dev != before.st_dev
            ):
                raise RunnerError(
                    f"Stream result file changed while collecting: {entry['path']}"
                )
    output_handle.flush()
    return {"ok": True, "files_total": len(entries)}


def run_command(
    argv: list[str],
    *,
    timeout: int = 45,
    check: bool = False,
    env: dict[str, str] | None = None,
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
            env=env
            or {
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


def _normalize_gpu_uuid(value: Any) -> str:
    text = str(value)
    if not text.startswith(("GPU-", "MIG-")):
        text = f"GPU-{text}"
    if not GPU_UUID_RE.fullmatch(text):
        raise RunnerError(f"GPU returned an invalid UUID: {text!r}")
    return text


def _docker_gpu_probe(devices: list[int]) -> tuple[list[dict[str, Any]], str | None]:
    details: list[dict[str, Any]] = []
    errors: list[str] = []
    for device in devices:
        process = run_command(
            [
                "/usr/bin/nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
                f"--id={device}",
            ]
        )
        if process.returncode != 0:
            errors.append(process.stderr.strip()[-500:] or f"GPU {device} is missing")
            continue
        fields = [field.strip() for field in process.stdout.strip().split(",")]
        if len(fields) != 6:
            errors.append(f"GPU {device} returned malformed nvidia-smi output")
            continue
        try:
            details.append(
                {
                    "index": device,
                    "name": fields[1],
                    "uuid": _normalize_gpu_uuid(fields[2]),
                    "memory_total_mib": int(fields[3]),
                    "memory_used_mib": int(fields[4]),
                    "utilization_percent": int(fields[5]),
                }
            )
        except (ValueError, RunnerError) as exc:
            errors.append(str(exc))
    return details, "; ".join(errors) or None


def _direct_gpu_probe(
    venv_path: Path, devices: list[int]
) -> tuple[list[dict[str, Any]], str | None]:
    python = venv_path / "bin" / "python"
    if not python.is_file() or python.is_symlink():
        return [], f"Virtualenv Python is missing: {python}"
    details, gpu_error = _docker_gpu_probe(devices)
    if gpu_error is not None:
        return details, gpu_error
    script = r"""
import json
import torch
if torch.version.cuda is None:
    raise RuntimeError("PyTorch virtualenv has no CUDA runtime")
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda}))
"""
    environment = dict(os.environ)
    environment["PATH"] = f"{venv_path}/bin:" + environment.get(
        "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    environment["VIRTUAL_ENV"] = str(venv_path)
    process = run_command(
        [str(python), "-c", script],
        timeout=60,
        env=environment,
    )
    if process.returncode != 0:
        return [], process.stderr.strip()[-1000:] or "CUDA probe failed"
    try:
        runtime = json.loads(process.stdout)
        if not isinstance(runtime, dict) or not runtime.get("cuda"):
            raise ValueError("virtualenv returned no CUDA runtime")
        return details, None
    except (json.JSONDecodeError, ValueError, RunnerError) as exc:
        return [], f"CUDA probe returned invalid JSON: {exc}"


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def operation_probe(request: dict[str, Any]) -> dict[str, Any]:
    work_root = ensure_root(require_remote_path(request, "work_root"))
    runtime = require_runtime(request)
    devices = require_gpu_devices(request)
    pipeline_root = require_remote_path(request, "pipeline_root")
    venv_path = require_remote_path(request, "venv_path")
    models_root = require_remote_path(request, "models_root")
    cache_root = ensure_root(require_remote_path(request, "cache_root"))
    standalone_renderer = work_root / "bin" / "balalaika-prepare-config.py"
    checkout_renderer = pipeline_root / "docker" / "prepare_config.py"
    renderer_path = (
        standalone_renderer if _regular_file(standalone_renderer) else checkout_renderer
    )
    image = request.get("image")
    if image is not None and (
        not isinstance(image, str) or not IMAGE_RE.fullmatch(image)
    ):
        raise RunnerError("Invalid image")

    models_ok = bool(
        models_root.is_dir()
        and not models_root.is_symlink()
        and next(models_root.iterdir(), None) is not None
    )
    pipeline_ok = bool(
        pipeline_root.is_dir()
        and not pipeline_root.is_symlink()
        and _regular_file(pipeline_root / "base.sh")
    )
    renderer_ok = _regular_file(renderer_path)
    venv_ok = bool(
        venv_path.is_dir()
        and not venv_path.is_symlink()
        and _regular_file(venv_path / "bin" / "python")
        and _regular_file(venv_path / "bin" / "activate")
    )

    docker_ok: bool | None = None
    docker_error: str | None = None
    image_id = None
    if runtime == "docker":
        docker = run_command(
            ["/usr/bin/docker", "version", "--format", "{{json .Server}}"]
        )
        docker_ok = docker.returncode == 0
        docker_error = docker.stderr.strip()[-500:] if docker.returncode else None
        gpu_details, gpu_error = _docker_gpu_probe(devices)
        if image and docker_ok:
            inspected = run_command(
                [
                    "/usr/bin/docker",
                    "image",
                    "inspect",
                    image,
                    "--format",
                    "{{.Id}}",
                ],
                timeout=20,
            )
            if inspected.returncode == 0:
                image_id = inspected.stdout.strip()
    else:
        gpu_details, gpu_error = _direct_gpu_probe(venv_path, devices)

    gpu_ok = gpu_error is None and len(gpu_details) == len(devices)
    runtime_ok = (
        bool(docker_ok and image_id)
        if runtime == "docker"
        else pipeline_ok and renderer_ok and venv_ok
    )
    disk = shutil.disk_usage(work_root)
    return {
        "ok": bool(runtime_ok and gpu_ok and models_ok),
        "hostname": socket.gethostname(),
        "runtime": runtime,
        "docker_ok": docker_ok,
        "docker_error": docker_error,
        "gpu_ok": gpu_ok,
        "gpu_error": gpu_error,
        "gpu_devices": devices,
        "gpu_uuids": [item["uuid"] for item in gpu_details],
        "gpu": gpu_details[0] if gpu_details else {},
        "gpus": gpu_details,
        "disk": {"total_bytes": disk.total, "free_bytes": disk.free},
        "image": image,
        "image_id": image_id,
        "models_ok": models_ok,
        "models_root": str(models_root),
        "cache_root": str(cache_root),
        "pipeline_ok": pipeline_ok if runtime == "direct" else None,
        "pipeline_root": str(pipeline_root),
        "renderer_ok": renderer_ok if runtime == "direct" else None,
        "renderer_path": str(renderer_path),
        "venv_ok": venv_ok if runtime == "direct" else None,
        "venv_path": str(venv_path),
        "runner_schema": SCHEMA_VERSION,
        "rsync_ok": shutil.which("rsync") is not None,
        "stream_protocol": STREAM_PROTOCOL_VERSION,
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
    if owner.get("runtime") == "direct":
        return _direct_process_alive(owner) or _direct_group_alive(owner)
    container_name = owner.get("container_name")
    if not isinstance(container_name, str):
        return False
    inspect = docker_inspect(container_name)
    return bool(inspect and inspect.get("State", {}).get("Running"))


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text("ascii").strip()
    except OSError as exc:
        raise RunnerError(f"Cannot read kernel boot ID: {exc}") from exc


def _process_start_ticks(pid: int) -> int | None:
    try:
        value = Path(f"/proc/{pid}/stat").read_text("ascii")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        raise RunnerError(f"Cannot inspect process {pid}: {exc}") from exc
    closing = value.rfind(")")
    if closing < 0:
        raise RunnerError(f"Process {pid} returned malformed stat data")
    fields = value[closing + 2:].split()
    if len(fields) <= 19:
        raise RunnerError(f"Process {pid} returned truncated stat data")
    if fields[0] in {"Z", "X"}:
        return None
    return int(fields[19])


def _direct_process_alive(metadata: dict[str, Any]) -> bool:
    pid = metadata.get("supervisor_pid")
    start_ticks = metadata.get("supervisor_start_ticks")
    boot_id = metadata.get("boot_id")
    if not isinstance(pid, int) or not isinstance(start_ticks, int):
        return False
    if boot_id != _boot_id():
        return False
    return _process_start_ticks(pid) == start_ticks


def _direct_group_alive(metadata: dict[str, Any]) -> bool:
    pgid = metadata.get("pipeline_pgid")
    return bool(
        isinstance(pgid, int)
        and metadata.get("boot_id") == _boot_id()
        and _process_group_alive(pgid)
    )


def _process_identity_alive(
    metadata: dict[str, Any], pid_key: str, start_ticks_key: str
) -> bool:
    pid = metadata.get(pid_key)
    start_ticks = metadata.get(start_ticks_key)
    if not isinstance(pid, int) or not isinstance(start_ticks, int):
        return False
    if metadata.get("boot_id") != _boot_id():
        return False
    return _process_start_ticks(pid) == start_ticks


def _process_group_alive(pgid: int) -> bool:
    if not isinstance(pgid, int) or pgid <= 1:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_stop(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_group_alive(pgid):
            return True
        time.sleep(0.1)
    return not _process_group_alive(pgid)


def _load_env_file(path: Path) -> dict[str, str]:
    try:
        if path.stat().st_size > 1024 * 1024:
            raise RunnerError("env_file exceeds 1 MiB")
        lines = path.read_text("utf-8").splitlines()
    except OSError as exc:
        raise RunnerError(f"Cannot read env_file: {exc}") from exc
    result: dict[str, str] = {}
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise RunnerError(f"env_file line {number} has no '='")
        name, value = line.split("=", 1)
        if not ENV_NAME_RE.fullmatch(name):
            raise RunnerError(f"env_file line {number} has an invalid name")
        if name in DIRECT_ENV_DENY:
            raise RunnerError(
                f"env_file line {number} uses forbidden variable {name}"
            )
        if "\x00" in value:
            raise RunnerError(f"env_file line {number} contains NUL")
        result[name] = value
    return result


def _direct_environment(
    paths: dict[str, Path],
    venv_path: Path,
    models_root: Path,
    cache_root: Path,
    gpu_uuids: list[str],
    env_file: Path | None,
    partition_id: str,
    global_rank: int,
) -> dict[str, str]:
    environment = {
        key: os.environ[key]
        for key in DIRECT_AMBIENT_ENV
        if key in os.environ and "\x00" not in os.environ[key]
    }
    if env_file is not None:
        environment.update(_load_env_file(env_file))
    cache_directories = {
        "BALALAIKA_CACHE_ROOT": cache_root / "balalaika",
        "BALALAIKA_TRT_CACHE_PATH": cache_root / "trt",
        "BALALAIKA_RUACCENT_WORKDIR": cache_root / "ruaccent",
        "HF_HOME": cache_root / "huggingface",
        "HF_HUB_CACHE": cache_root / "huggingface" / "hub",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "TORCH_HOME": cache_root / "torch",
        "NUMBA_CACHE_DIR": cache_root / "numba",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "HOME": cache_root / "home",
    }
    for directory in cache_directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "PATH": f"{venv_path}/bin:"
            + environment.get(
                "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            ),
            "VIRTUAL_ENV": str(venv_path),
            "BALALAIKA_VENV": str(venv_path),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": ",".join(gpu_uuids),
            "BALALAIKA_CONFIG_PATH": str(paths["runtime_config"]),
            "BALALAIKA_DATA_ROOT": str(paths["data"]),
            "BALALAIKA_MODELS_ROOT": str(models_root),
            "BALALAIKA_LOG_DIR": str(paths["logs"]),
            "BALALAIKA_OUTPUT_ROOT": str(paths["output"]),
            "BALALAIKA_PARTITION_ID": partition_id,
            "BALALAIKA_GLOBAL_RANK": str(global_rank),
            "BALALAIKA_NODE_PROFILE": str(
                cache_directories["BALALAIKA_CACHE_ROOT"] / "node_profile.json"
            ),
            "BALALAIKA_DISABLE_CUDF": environment.get(
                "BALALAIKA_DISABLE_CUDF", "1"
            ),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            **{key: str(value) for key, value in cache_directories.items()},
        }
    )
    return environment


def _resolve_gpu_details(
    runtime: str, devices: list[int], venv_path: Path
) -> list[dict[str, Any]]:
    if runtime == "docker":
        details, error = _docker_gpu_probe(devices)
    else:
        details, error = _direct_gpu_probe(venv_path, devices)
    if error or len(details) != len(devices):
        raise RunnerError(error or "Could not resolve configured GPUs")
    return details


def _require_gpu_idle_limits(request: dict[str, Any]) -> tuple[int, int]:
    memory_limit = request.get("max_gpu_memory_used_mib", 1024)
    utilization_limit = request.get("max_gpu_utilization_percent", 20)
    if (
        isinstance(memory_limit, bool)
        or not isinstance(memory_limit, int)
        or not 0 <= memory_limit <= 1024 * 1024
    ):
        raise RunnerError("Invalid max_gpu_memory_used_mib")
    if (
        isinstance(utilization_limit, bool)
        or not isinstance(utilization_limit, int)
        or not 0 <= utilization_limit <= 100
    ):
        raise RunnerError("Invalid max_gpu_utilization_percent")
    return memory_limit, utilization_limit


def _gpu_busy_error(
    details: list[dict[str, Any]], memory_limit: int, utilization_limit: int
) -> str | None:
    for item in details:
        used = item.get("memory_used_mib")
        utilization = item.get("utilization_percent")
        if not isinstance(used, int) or not isinstance(utilization, int):
            return f"GPU {item.get('index')} returned incomplete occupancy metrics"
        if used > memory_limit:
            return (
                f"GPU {item.get('index')} is busy: {used} MiB used exceeds "
                f"limit {memory_limit} MiB"
            )
        if utilization > utilization_limit:
            return (
                f"GPU {item.get('index')} is busy: {utilization}% utilization "
                f"exceeds limit {utilization_limit}%"
            )
    return None


def direct_worker_main(spec_path: Path) -> int:
    exit_path: Path | None = None
    process: subprocess.Popen[bytes] | None = None
    spec_sha256: str | None = None
    fencing_token: str | None = None

    def forward_signal(signum: int, _frame: Any) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    try:
        spec = read_json(spec_path)
        spec_path.unlink(missing_ok=True)
        argv = spec.get("argv")
        environment = spec.get("env")
        cwd = spec.get("cwd")
        log_path = spec.get("log_path")
        exit_path_value = spec.get("exit_path")
        process_path_value = spec.get("process_path")
        spec_sha256 = spec.get("spec_sha256")
        fencing_token = spec.get("fencing_token")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(value, str) for value in argv)
            or not isinstance(environment, dict)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in environment.items()
            )
            or not isinstance(cwd, str)
            or not isinstance(log_path, str)
            or not isinstance(exit_path_value, str)
            or not isinstance(process_path_value, str)
            or not isinstance(spec_sha256, str)
            or not SHA256_RE.fullmatch(spec_sha256)
            or not isinstance(fencing_token, str)
            or not HEX_RE.fullmatch(fencing_token)
        ):
            raise RunnerError("Direct worker spec is invalid")
        exit_path = Path(exit_path_value)
        process_path = Path(process_path_value)
        signal.signal(signal.SIGINT, forward_signal)
        signal.signal(signal.SIGTERM, forward_signal)
        log = Path(log_path)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab", buffering=0) as handle:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                env=environment,
                shell=False,
                close_fds=True,
                start_new_session=True,
            )
            process_record = {
                "supervisor_pid": os.getpid(),
                "supervisor_start_ticks": _process_start_ticks(os.getpid()),
                "pipeline_pid": process.pid,
                "pipeline_pgid": os.getpgid(process.pid),
                "pipeline_start_ticks": _process_start_ticks(process.pid),
                "boot_id": _boot_id(),
                "spec_sha256": spec_sha256,
                "fencing_token": fencing_token,
                "started_at": utc_now(),
            }
            atomic_json(process_path, process_record)
            exit_code = process.wait()
            pgid = process_record["pipeline_pgid"]
            if _process_group_alive(pgid):
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                if not _wait_for_process_group_stop(pgid, 5.0):
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    if not _wait_for_process_group_stop(pgid, 5.0):
                        raise RunnerError(
                            "Direct pipeline left a process group alive after SIGKILL"
                        )
        atomic_json(
            exit_path,
            {
                "exit_code": exit_code,
                "finished_at": utc_now(),
                "pipeline_pid": process.pid,
                "spec_sha256": spec_sha256,
                "fencing_token": fencing_token,
            },
        )
        return int(exit_code)
    except BaseException as exc:
        try:
            if exit_path is None:
                return 255
            record = {
                "exit_code": 255,
                "finished_at": utc_now(),
                "error": str(exc),
            }
            if isinstance(spec_sha256, str) and SHA256_RE.fullmatch(spec_sha256):
                record["spec_sha256"] = spec_sha256
            if isinstance(fencing_token, str) and HEX_RE.fullmatch(fencing_token):
                record["fencing_token"] = fencing_token
            atomic_json(exit_path, record)
        except Exception:
            pass
        return 255


def operation_start(request: dict[str, Any]) -> dict[str, Any]:
    paths = attempt_paths(request)
    token = require_hex(request, "fencing_token")
    global_rank = require_global_rank(request)
    runtime = require_runtime(request)
    devices = require_gpu_devices(request)
    expected_gpu_uuids = require_gpu_uuids(request, len(devices))
    memory_limit, utilization_limit = _require_gpu_idle_limits(request)
    pipeline_root = require_remote_path(request, "pipeline_root")
    venv_path = require_remote_path(request, "venv_path")
    image = request.get("image")
    if runtime == "docker" and (
        not isinstance(image, str) or not IMAGE_RE.fullmatch(image)
    ):
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

    gpu_details = _resolve_gpu_details(runtime, devices, venv_path)
    resolved_gpu_uuids = [item["uuid"] for item in gpu_details]
    if resolved_gpu_uuids != expected_gpu_uuids:
        raise RunnerError(
            "Configured GPU mapping changed after claim: "
            f"expected {expected_gpu_uuids}, got {resolved_gpu_uuids}"
        )
    if runtime == "direct":
        standalone_renderer = (
            paths["work_root"] / "bin" / "balalaika-prepare-config.py"
        )
        checkout_renderer = pipeline_root / "docker" / "prepare_config.py"
        renderer_path = (
            standalone_renderer
            if _regular_file(standalone_renderer)
            else checkout_renderer
        )
        if (
            not pipeline_root.is_dir()
            or pipeline_root.is_symlink()
            or not _regular_file(pipeline_root / "base.sh")
        ):
            raise RunnerError(f"Direct pipeline root is invalid: {pipeline_root}")
        if not _regular_file(renderer_path):
            raise RunnerError(
                f"Direct config renderer is missing; bootstrap the node: {renderer_path}"
            )
        if (
            not venv_path.is_dir()
            or venv_path.is_symlink()
            or not _regular_file(venv_path / "bin" / "python")
            or not _regular_file(venv_path / "bin" / "activate")
        ):
            raise RunnerError(f"Direct virtualenv is invalid: {venv_path}")

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

    spec = {
        "attempt_id": metadata["attempt_id"],
        "partition_id": metadata["partition_id"],
        "global_rank": global_rank,
        "config_sha256": metadata["config_sha256"],
        "fencing_token": token,
        "runtime": runtime,
        "gpu_devices": devices,
        "gpu_uuids": expected_gpu_uuids,
        "image": image if runtime == "docker" else None,
        "manifest_sha256": metadata["manifest_sha256"],
        "shm_size": shm_size,
        "stage_start": stage_start,
        "stage_stop": stage_stop,
        "pipeline_root": str(pipeline_root),
        "venv_path": str(venv_path),
        "models_root": str(models_root),
        "cache_root": str(cache_root),
    }
    spec_sha256 = hashlib.sha256(canonical_json(spec)).hexdigest()
    if metadata.get("spec_sha256") == spec_sha256:
        if runtime == "direct" and metadata.get("runtime") == "direct":
            has_process_identity = (
                paths["direct_process"].is_file()
                or paths["direct_exit"].is_file()
                or isinstance(metadata.get("supervisor_pid"), int)
            )
            if has_process_identity:
                return _status_direct(paths, metadata)
        if runtime == "docker":
            container_name = metadata.get("container_name")
            if isinstance(container_name, str):
                existing = docker_inspect(container_name)
                if existing:
                    return _status_from_inspect(paths, metadata, existing)

    ensure_directory_chain(paths["work_root"], ("locks",))
    with paths["lock"].open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerError("Worker GPU set is reserved by another attempt") from exc

        container_name = _container_name(metadata)
        if runtime == "docker":
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
                    "Worker GPU set is reserved by attempt "
                    f"{owner.get('attempt_id', 'unknown')}"
                )
            paths["owner"].unlink(missing_ok=True)

        busy_error = _gpu_busy_error(gpu_details, memory_limit, utilization_limit)
        if busy_error is not None:
            raise RunnerError(busy_error)

        metadata.update(
            {
                "state": "STARTING",
                "updated_at": utc_now(),
                "spec_sha256": spec_sha256,
                "stage_start": stage_start,
                "stage_stop": stage_stop,
                "runtime": runtime,
                "global_rank": global_rank,
                "gpu_devices": devices,
                "gpu_uuids": expected_gpu_uuids,
                "pipeline_root": str(pipeline_root),
                "venv_path": str(venv_path),
                "models_root": str(models_root),
                "cache_root": str(cache_root),
            }
        )
        atomic_json(paths["job"], metadata)

        if runtime == "direct":
            environment = _direct_environment(
                paths,
                venv_path,
                models_root,
                cache_root,
                expected_gpu_uuids,
                env_file,
                metadata["partition_id"],
                global_rank,
            )
            paths["direct_exit"].unlink(missing_ok=True)
            paths["direct_process"].unlink(missing_ok=True)
            run_command(
                [
                    str(venv_path / "bin" / "python"),
                    str(renderer_path),
                    "--input",
                    str(paths["config"]),
                    "--output",
                    str(paths["runtime_config"]),
                ],
                timeout=120,
                check=True,
                env=environment,
            )
            direct_spec = {
                "argv": [
                    "/usr/bin/bash",
                    str(pipeline_root / "base.sh"),
                    "--config_path",
                    str(paths["runtime_config"]),
                    "--stage",
                    stage_start,
                    "--stop_stage",
                    stage_stop,
                    "--strict",
                ],
                "cwd": str(pipeline_root),
                "env": environment,
                "log_path": str(paths["direct_log"]),
                "exit_path": str(paths["direct_exit"]),
                "process_path": str(paths["direct_process"]),
                "spec_sha256": spec_sha256,
                "fencing_token": token,
            }
            atomic_json(paths["direct_spec"], direct_spec)
            supervisor = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "direct-worker",
                    str(paths["direct_spec"]),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd="/",
                env={
                    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                    "PYTHONUNBUFFERED": "1",
                },
                shell=False,
                close_fds=True,
                pass_fds=(lock_handle.fileno(),),
                start_new_session=True,
            )
            process_record: dict[str, Any] | None = None
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if paths["direct_process"].exists():
                    process_record = read_json(paths["direct_process"])
                    break
                if paths["direct_exit"].exists() or supervisor.poll() is not None:
                    break
                time.sleep(0.05)
            if process_record is None:
                if supervisor.poll() is None:
                    supervisor.terminate()
                    try:
                        supervisor.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        supervisor.kill()
                        supervisor.wait(timeout=5)
                detail = ""
                if paths["direct_exit"].exists():
                    detail = str(read_json(paths["direct_exit"]).get("error") or "")
                paths["direct_exit"].unlink(missing_ok=True)
                paths["direct_process"].unlink(missing_ok=True)
                paths["direct_spec"].unlink(missing_ok=True)
                metadata.update({"state": "READY", "updated_at": utc_now()})
                atomic_json(paths["job"], metadata)
                raise RunnerError(
                    "Direct pipeline supervisor did not publish its startup handshake"
                    + (f": {detail}" if detail else "")
                )
            if (
                process_record.get("supervisor_pid") != supervisor.pid
                or process_record.get("spec_sha256") != spec_sha256
                or process_record.get("fencing_token") != token
                or (
                    not paths["direct_exit"].exists()
                    and not _process_identity_alive(
                        process_record, "supervisor_pid", "supervisor_start_ticks"
                    )
                )
            ):
                if supervisor.poll() is None:
                    supervisor.terminate()
                    try:
                        supervisor.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        supervisor.kill()
                        supervisor.wait(timeout=5)
                raise RunnerError("Direct pipeline supervisor handshake is invalid")
            metadata.update(
                {
                    "state": "RUNNING",
                    "supervisor_pid": process_record["supervisor_pid"],
                    "supervisor_start_ticks": process_record[
                        "supervisor_start_ticks"
                    ],
                    "pipeline_pid": process_record["pipeline_pid"],
                    "pipeline_pgid": process_record["pipeline_pgid"],
                    "pipeline_start_ticks": process_record["pipeline_start_ticks"],
                    "boot_id": process_record["boot_id"],
                    "started_at": metadata.get("started_at")
                    or process_record["started_at"],
                    "updated_at": utc_now(),
                }
            )
            owner = {
                "attempt_id": metadata["attempt_id"],
                "runtime": "direct",
                "supervisor_pid": process_record["supervisor_pid"],
                "supervisor_start_ticks": process_record[
                    "supervisor_start_ticks"
                ],
                "pipeline_pid": process_record["pipeline_pid"],
                "pipeline_pgid": process_record["pipeline_pgid"],
                "pipeline_start_ticks": process_record["pipeline_start_ticks"],
                "boot_id": process_record["boot_id"],
                "fencing_token": token,
                "spec_sha256": spec_sha256,
                "gpu_uuids": expected_gpu_uuids,
                "updated_at": utc_now(),
            }
            atomic_json(paths["owner"], owner)
            atomic_json(paths["job"], metadata)
            return _status_direct(paths, metadata)

        uid = os.getuid()
        gid = os.getgid()
        docker_device = ",".join(expected_gpu_uuids)
        if len(expected_gpu_uuids) > 1:
            docker_device = f'"device={docker_device}"'
        else:
            docker_device = f"device={docker_device}"
        logical_devices = ",".join(str(index) for index in range(len(devices)))
        docker_argv = [
            "/usr/bin/docker",
            "run",
            "--detach",
            "--name",
            container_name,
            "--gpus",
            docker_device,
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
            f"CUDA_VISIBLE_DEVICES={logical_devices}",
            "--env",
            f"BALALAIKA_PIPELINE_ROOT={pipeline_root}",
            "--env",
            f"VIRTUAL_ENV={venv_path}",
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
                "--env",
                f"BALALAIKA_PARTITION_ID={metadata['partition_id']}",
                "--env",
                f"BALALAIKA_GLOBAL_RANK={global_rank}",
            ]
        )
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
        metadata.update({"container_name": container_name, "image": image})
        atomic_json(paths["job"], metadata)
        process = run_command(docker_argv, timeout=120, check=True)
        container_id = process.stdout.strip()
        owner = {
            "attempt_id": metadata["attempt_id"],
            "runtime": "docker",
            "container_name": container_name,
            "fencing_token": token,
            "spec_sha256": spec_sha256,
            "gpu_uuids": expected_gpu_uuids,
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


def _status_direct(
    paths: dict[str, Path], metadata: dict[str, Any]
) -> dict[str, Any]:
    progress = _stage_progress(paths, metadata)
    result = {
        "ok": True,
        "runtime": "direct",
        "attempt_id": metadata["attempt_id"],
        "attempt_root": str(paths["root"]),
        "started_at": metadata.get("started_at"),
        "updated_at": utc_now(),
        **progress,
    }
    if paths["direct_exit"].is_file():
        exit_record = read_json(paths["direct_exit"])
        if (
            exit_record.get("spec_sha256") != metadata.get("spec_sha256")
            or exit_record.get("fencing_token") != metadata.get("fencing_token")
        ):
            raise RunnerError("Direct pipeline exit marker has stale job identity")
        exit_code = exit_record.get("exit_code")
        if not isinstance(exit_code, int):
            raise RunnerError("Direct pipeline exit marker has no integer exit code")
        finished_at = exit_record.get("finished_at") or utc_now()
        metadata.update(
            {
                "state": "COMPLETED" if exit_code == 0 else "FAILED",
                "exit_code": exit_code,
                "finished_at": finished_at,
                "updated_at": utc_now(),
            }
        )
        atomic_json(paths["job"], metadata)
        result.update(
            {
                "state": metadata["state"],
                "exit_code": exit_code,
                "finished_at": finished_at,
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
            detail = exit_record.get("error")
            result["error"] = (
                str(detail)
                if detail
                else f"Direct pipeline exited with code {exit_code}"
            )
        return result

    if _direct_process_alive(metadata):
        metadata.update({"state": "RUNNING", "updated_at": utc_now()})
        atomic_json(paths["job"], metadata)
        result.update(
            {
                "state": "RUNNING",
                "supervisor_pid": metadata.get("supervisor_pid"),
                "pipeline_pid": metadata.get("pipeline_pid"),
            }
        )
        return result

    process_record = (
        read_json(paths["direct_process"])
        if paths["direct_process"].is_file()
        else metadata
    )
    pipeline_alive = _direct_group_alive(process_record)
    if (
        metadata.get("state") == "STARTING"
        and not pipeline_alive
        and not paths["direct_process"].exists()
    ):
        metadata.update({"state": "READY", "updated_at": utc_now()})
        atomic_json(paths["job"], metadata)
        return {**result, "state": "READY"}
    return {
        **result,
        "ok": False,
        "state": "UNKNOWN",
        "error": (
            "Direct supervisor disappeared while the pipeline process is still alive"
            if pipeline_alive
            else "Direct supervisor disappeared without an exit marker"
        ),
    }


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
    if metadata.get("runtime") == "direct":
        return _status_direct(paths, metadata)
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


def _signal_direct_pipeline(metadata: dict[str, Any], signum: int) -> bool:
    pipeline_pid = metadata["pipeline_pid"]
    pipeline_pgid = metadata.get("pipeline_pgid")
    if pipeline_pgid != pipeline_pid:
        raise RunnerError("Refusing to signal an unexpected direct process group")
    if not _process_identity_alive(
        metadata, "pipeline_pid", "pipeline_start_ticks"
    ):
        if _direct_group_alive(metadata):
            raise RunnerError(
                "Refusing to signal a direct process group with stale PID identity"
            )
        return False
    if not _process_group_alive(pipeline_pgid):
        return False
    try:
        os.killpg(pipeline_pgid, signum)
    except ProcessLookupError:
        return False
    return True


def _wait_for_process_stop(
    metadata: dict[str, Any], pid_key: str, start_ticks_key: str, timeout: float
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_identity_alive(metadata, pid_key, start_ticks_key):
            return True
        time.sleep(0.1)
    return not _process_identity_alive(metadata, pid_key, start_ticks_key)


def _cancel_direct(paths: dict[str, Path], metadata: dict[str, Any]) -> None:
    process_record = (
        read_json(paths["direct_process"])
        if paths["direct_process"].is_file()
        else metadata
    )
    if (
        process_record.get("spec_sha256") != metadata.get("spec_sha256")
        or process_record.get("fencing_token") != metadata.get("fencing_token")
    ):
        raise RunnerError("Direct process marker has stale job identity")

    pipeline_pgid = process_record.get("pipeline_pgid")
    if not isinstance(pipeline_pgid, int):
        raise RunnerError("Direct process marker has no pipeline process group")
    if _signal_direct_pipeline(process_record, signal.SIGINT):
        stopped = _wait_for_process_group_stop(pipeline_pgid, 30.0)
        if not stopped:
            _signal_direct_pipeline(process_record, signal.SIGTERM)
            stopped = _wait_for_process_group_stop(pipeline_pgid, 10.0)
        if not stopped:
            _signal_direct_pipeline(process_record, signal.SIGKILL)
            stopped = _wait_for_process_group_stop(pipeline_pgid, 5.0)
        if not stopped:
            raise RunnerError("Direct pipeline did not stop after SIGKILL")

    if not _wait_for_process_stop(
        process_record, "supervisor_pid", "supervisor_start_ticks", 5.0
    ):
        supervisor_pid = process_record["supervisor_pid"]
        try:
            os.kill(supervisor_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if not _wait_for_process_stop(
            process_record, "supervisor_pid", "supervisor_start_ticks", 2.0
        ):
            try:
                os.kill(supervisor_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if not _wait_for_process_stop(
                process_record, "supervisor_pid", "supervisor_start_ticks", 2.0
            ):
                raise RunnerError("Direct supervisor did not stop after SIGKILL")


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
    if metadata.get("runtime") == "direct":
        _cancel_direct(paths, metadata)
    elif isinstance(container_name, str) and docker_inspect(container_name):
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
    if metadata.get("runtime") == "direct":
        if _direct_process_alive(metadata) or _direct_group_alive(metadata):
            raise RunnerError("Cannot clean up a running direct attempt")
    elif isinstance(container_name, str):
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


def stream_push_main() -> int:
    try:
        response = stream_push(sys.stdin.buffer)
        sys.stdout.buffer.write(canonical_json(response) + b"\n")
        return 0
    except (RunnerError, KeyError, TypeError, ValueError) as exc:
        sys.stderr.write(f"stream-push: {exc}\n")
        return 2


def stream_pull_main() -> int:
    try:
        stream_pull(sys.stdin.buffer, sys.stdout.buffer)
        return 0
    except (RunnerError, KeyError, TypeError, ValueError) as exc:
        sys.stderr.write(f"stream-pull: {exc}\n")
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("rpc", "stream-push", "stream-pull", "direct-worker")
    )
    parser.add_argument("spec_path", nargs="?", type=Path)
    args = parser.parse_args()
    if args.command == "rpc":
        if args.spec_path is not None:
            parser.error("rpc does not accept a spec path")
        return rpc_main()
    if args.command == "stream-push":
        if args.spec_path is not None:
            parser.error("stream-push does not accept a path argument")
        return stream_push_main()
    if args.command == "stream-pull":
        if args.spec_path is not None:
            parser.error("stream-pull does not accept a path argument")
        return stream_pull_main()
    if args.spec_path is None or not args.spec_path.is_absolute():
        parser.error("direct-worker requires an absolute spec path")
    try:
        info = args.spec_path.lstat()
    except OSError as exc:
        parser.error(f"cannot inspect direct-worker spec: {exc}")
    if not stat.S_ISREG(info.st_mode) or args.spec_path.is_symlink():
        parser.error("direct-worker spec must be a regular file")
    return direct_worker_main(args.spec_path)


if __name__ == "__main__":
    raise SystemExit(main())
