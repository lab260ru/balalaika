"""Pinned-host SSH RPC with rsync and runner-stream data transports."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import stat
import struct
import subprocess
import tarfile
import tempfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, BinaryIO, Iterable

from .config import ClusterConfig, NodeConfig

STREAM_PROTOCOL_VERSION = 1
STREAM_MAGIC = b"BLKSTRM1"
STREAM_HEADER_LIMIT = 256 * 1024 * 1024


class TransportError(RuntimeError):
    """A remote command or transfer failed."""


def _bounded_error(value: str, limit: int = 4000) -> str:
    value = value.strip()
    return value[-limit:] if value else "no diagnostic output"


class ClusterTransport:
    """Transport with no dependency on a resident service on worker nodes."""

    def __init__(self, config: ClusterConfig):
        self.config = config
        self.ssh_binary = Path("/usr/bin/ssh")
        self.rsync_binary = Path("/usr/bin/rsync")
        self._auto_transfer_modes: dict[str, str] = {}

    def _safe_environment(self) -> dict[str, str]:
        environment = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
        auth_sock = os.environ.get("SSH_AUTH_SOCK")
        if auth_sock:
            environment["SSH_AUTH_SOCK"] = auth_sock
        return environment

    def _ssh_options(self, node: NodeConfig) -> list[str]:
        options = [
            str(self.ssh_binary),
            "-F",
            "/dev/null",
            "-l",
            node.user,
            "-p",
            str(node.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={self.config.known_hosts}",
            "-o",
            "UpdateHostKeys=no",
            "-o",
            f"ConnectTimeout={self.config.connect_timeout}",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=2",
            "-o",
            "LogLevel=ERROR",
        ]
        if self.config.ssh_identity:
            options[1:1] = ["-i", str(self.config.ssh_identity)]
        return options

    def _check_ssh_files(self) -> None:
        if not self.ssh_binary.is_file():
            raise TransportError(f"OpenSSH client is missing: {self.ssh_binary}")
        if not self.config.known_hosts.is_file():
            raise TransportError(
                f"Pinned known_hosts file is missing: {self.config.known_hosts}"
            )
        if self.config.ssh_identity and not self.config.ssh_identity.is_file():
            raise TransportError(
                f"SSH identity file is missing: {self.config.ssh_identity}"
            )

    def _ssh_command(self, node: NodeConfig, remote_command: str) -> list[str]:
        return [*self._ssh_options(node), "--", node.host, remote_command]

    def rpc(
        self,
        node: NodeConfig,
        request: dict[str, Any],
        *,
        timeout: int = 60,
    ) -> dict[str, Any]:
        self._check_ssh_files()
        command = shlex.join(["python3", node.runner_path, "rpc"])
        payload = json.dumps(
            request, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        try:
            process = subprocess.run(
                self._ssh_command(node, command),
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
                shell=False,
                env=self._safe_environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TransportError(f"SSH RPC to {node.id} failed: {exc}") from exc
        try:
            response = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise TransportError(
                f"Node {node.id} returned invalid RPC JSON: "
                f"{_bounded_error(process.stdout.decode('utf-8', errors='replace'))}; "
                f"SSH: {_bounded_error(process.stderr.decode('utf-8', errors='replace'))}"
            ) from exc
        if not isinstance(response, dict):
            raise TransportError(f"Node {node.id} returned a non-object RPC response")
        if process.returncode != 0 or response.get("ok") is not True:
            message = response.get("error") or process.stderr.decode(
                "utf-8", errors="replace"
            )
            raise TransportError(f"Node {node.id}: {_bounded_error(str(message))}")
        return response

    def _bootstrap_python_file(
        self, node: NodeConfig, source: Path, destination: Path
    ) -> None:
        self._check_ssh_files()
        temporary = destination.with_suffix(destination.suffix + ".partial")
        # Every path component came through the remote-path allowlist in config.py.
        remote_command = " && ".join(
            (
                "umask 077",
                shlex.join(["mkdir", "-p", str(destination.parent)]),
                f"cat > {shlex.quote(str(temporary))}",
                shlex.join(["python3", "-m", "py_compile", str(temporary)]),
                shlex.join(["chmod", "700", str(temporary)]),
                shlex.join(["mv", "-f", str(temporary), str(destination)]),
            )
        )
        try:
            with source.open("rb") as handle:
                process = subprocess.run(
                    self._ssh_command(node, remote_command),
                    stdin=handle,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=90,
                    check=False,
                    shell=False,
                    env=self._safe_environment(),
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TransportError(f"Cannot bootstrap node {node.id}: {exc}") from exc
        if process.returncode != 0:
            raise TransportError(
                f"Cannot bootstrap node {node.id}: "
                f"{_bounded_error(process.stderr.decode('utf-8', errors='replace'))}"
            )

    def bootstrap(self, node: NodeConfig) -> dict[str, Any]:
        """Install standalone control files without changing the pipeline checkout."""
        self._bootstrap_python_file(
            node,
            Path(__file__).with_name("node_runner.py"),
            Path(node.runner_path),
        )
        self._bootstrap_python_file(
            node,
            Path(__file__).parents[1] / "docker" / "prepare_config.py",
            Path(node.renderer_path),
        )
        return {
            "ok": True,
            "runner_path": node.runner_path,
            "renderer_path": node.renderer_path,
        }

    def probe(self, node: NodeConfig) -> dict[str, Any]:
        result = self.rpc(
            node,
            {
                "operation": "probe",
                "work_root": node.work_root,
                "runtime": node.runtime,
                "gpu_devices": list(node.gpu_devices),
                "pipeline_root": node.pipeline_root,
                "venv_path": node.venv_path,
                "image": node.image or self.config.image,
                "models_root": node.models_root,
                "cache_root": node.cache_root,
            },
        )
        result["selected_transfer"] = self._record_transfer_capabilities(node, result)
        return result

    def _record_transfer_capabilities(
        self, node: NodeConfig, result: dict[str, Any]
    ) -> str:
        if node.transfer != "auto":
            return node.transfer
        if result.get("rsync_ok") is True and self.rsync_binary.is_file():
            selected = "rsync"
        elif result.get("stream_protocol") == STREAM_PROTOCOL_VERSION:
            selected = "stream"
        else:
            # A runner from before the stream protocol did not report either field.
            selected = "rsync"
        self._auto_transfer_modes[node.id] = selected
        return selected

    def _transfer_mode(self, node: NodeConfig) -> str:
        if node.transfer in {"rsync", "stream"}:
            return node.transfer
        if node.transfer != "auto":
            raise TransportError(f"Node {node.id} has invalid transfer mode")
        # Reconciliation may resume STAGING immediately after a controller
        # restart, before another probe. The bootstrapped runner always supports
        # stream mode, while remote rsync availability is unknown at this point.
        return self._auto_transfer_modes.get(node.id, "stream")

    def _rsync_ssh(self, node: NodeConfig) -> str:
        return shlex.join(self._ssh_options(node)[0:])

    @staticmethod
    def _remote(node: NodeConfig, path: str) -> str:
        host = f"[{node.host}]" if ":" in node.host else node.host
        return f"{node.user}@{host}:{path}"

    def _check_rsync(self) -> None:
        self._check_ssh_files()
        if not self.rsync_binary.is_file():
            raise TransportError(
                "rsync is required on the controller and every node; "
                f"missing controller binary: {self.rsync_binary}"
            )

    def _run_rsync(
        self,
        node: NodeConfig,
        arguments: list[str],
        *,
        stdin: BinaryIO | int | None = subprocess.DEVNULL,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        self._check_rsync()
        argv = [
            str(self.rsync_binary),
            "--protect-args",
            "--no-links",
            "--partial",
            "--partial-dir=.rsync-partial",
            "--rsh",
            self._rsync_ssh(node),
            *arguments,
        ]
        try:
            process = subprocess.run(
                argv,
                stdin=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
                shell=False,
                env=self._safe_environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TransportError(f"rsync with node {node.id} failed: {exc}") from exc
        if process.returncode != 0:
            raise TransportError(
                f"rsync with node {node.id} failed ({process.returncode}): "
                f"{_bounded_error(process.stderr.decode('utf-8', errors='replace'))}"
            )
        return process

    @staticmethod
    def _stream_write_header(handle: BinaryIO, value: dict[str, Any]) -> None:
        payload = json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        if len(payload) > STREAM_HEADER_LIMIT:
            raise TransportError("Stream protocol header is too large")
        handle.write(STREAM_MAGIC)
        handle.write(struct.pack("!I", len(payload)))
        handle.write(payload)

    @staticmethod
    def _stream_read_exact(handle: BinaryIO, length: int) -> bytes:
        chunks: list[bytes] = []
        remaining = length
        while remaining:
            chunk = handle.read(remaining)
            if not chunk:
                raise TransportError("Truncated stream protocol frame")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    @classmethod
    def _stream_read_header(cls, handle: BinaryIO) -> dict[str, Any]:
        if cls._stream_read_exact(handle, len(STREAM_MAGIC)) != STREAM_MAGIC:
            raise TransportError("Invalid stream protocol magic")
        length = struct.unpack("!I", cls._stream_read_exact(handle, 4))[0]
        if length < 2 or length > STREAM_HEADER_LIMIT:
            raise TransportError("Invalid stream protocol header length")
        try:
            value = json.loads(cls._stream_read_exact(handle, length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransportError(f"Invalid stream protocol header: {exc}") from exc
        if not isinstance(value, dict):
            raise TransportError("Stream protocol header must be a JSON object")
        return value

    @staticmethod
    def _safe_stream_path(value: str, label: str) -> PurePosixPath:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise TransportError(f"Invalid {label}")
        if any(ord(character) < 32 for character in value):
            raise TransportError(f"Invalid {label}")
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts or str(path) != value:
            raise TransportError(f"Invalid {label}")
        return path

    @staticmethod
    def _safe_archive_name(value: str) -> str:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise TransportError("Invalid stream member path")
        if any(ord(character) < 32 for character in value):
            raise TransportError("Invalid stream member path")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or path == PurePosixPath(".")
            or ".." in path.parts
            or str(path) != value
            or value.startswith("-")
        ):
            raise TransportError(f"Unsafe stream member path: {value!r}")
        return path.as_posix()

    @classmethod
    def _stream_push_location(cls, remote_path: str) -> tuple[str, str]:
        path = cls._safe_stream_path(remote_path, "remote stream path")
        if path.parent.name == "control" and path.name in {
            "manifest.json",
            "config.yaml",
        }:
            role = "manifest" if path.name == "manifest.json" else "config"
            return str(path.parent.parent), role
        if path.parent.name == "data.partial":
            return str(path.parent.parent), "state"
        raise TransportError(
            "Stream push_file only accepts prepared manifest, config, or state paths"
        )

    @classmethod
    def _stream_data_location(cls, remote_path: str) -> str:
        path = cls._safe_stream_path(remote_path, "remote data path")
        if path.name != "data.partial":
            raise TransportError("Stream partition destination must be data.partial")
        return str(path.parent)

    @staticmethod
    def _stderr_tail(handle: BinaryIO) -> str:
        handle.seek(0, os.SEEK_END)
        length = handle.tell()
        handle.seek(max(0, length - 4000))
        return _bounded_error(handle.read().decode("utf-8", errors="replace"))

    def _stream_command(self, node: NodeConfig, command: str) -> list[str]:
        if command not in {"stream-push", "stream-pull"}:
            raise TransportError("Invalid runner stream command")
        remote_command = shlex.join(["python3", node.runner_path, command])
        return self._ssh_command(node, remote_command)

    @staticmethod
    def _open_regular_source(path: Path) -> tuple[BinaryIO, os.stat_result]:
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            info = os.fstat(descriptor)
        except OSError as exc:
            raise TransportError(f"Cannot open transfer source {path}: {exc}") from exc
        if not stat.S_ISREG(info.st_mode):
            os.close(descriptor)
            raise TransportError(f"Transfer source is not a regular file: {path}")
        return os.fdopen(descriptor, "rb"), info

    @staticmethod
    def _add_stream_file(
        archive: tarfile.TarFile, name: str, path: Path
    ) -> None:
        source, before = ClusterTransport._open_regular_source(path)
        try:
            member = tarfile.TarInfo(name)
            member.size = before.st_size
            member.mtime = before.st_mtime_ns // 1_000_000_000
            member.mode = 0o600
            archive.addfile(member, source)
            after = os.fstat(source.fileno())
        finally:
            source.close()
        if (
            after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
        ):
            raise TransportError(f"Transfer source changed while streaming: {path}")

    def _run_stream_push(
        self,
        node: NodeConfig,
        header: dict[str, Any],
        entries: Iterable[tuple[str, Path]],
    ) -> None:
        self._check_ssh_files()
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                process = subprocess.Popen(
                    self._stream_command(node, "stream-push"),
                    stdin=subprocess.PIPE,
                    stdout=stdout,
                    stderr=stderr,
                    shell=False,
                    env=self._safe_environment(),
                )
            except OSError as exc:
                raise TransportError(
                    f"Cannot start stream transfer to node {node.id}: {exc}"
                ) from exc
            write_error: Exception | None = None
            try:
                if process.stdin is None:
                    raise TransportError("Stream transfer has no stdin pipe")
                self._stream_write_header(process.stdin, header)
                with tarfile.open(
                    fileobj=process.stdin, mode="w|", format=tarfile.PAX_FORMAT
                ) as archive:
                    seen: set[str] = set()
                    for raw_name, path in entries:
                        name = self._safe_archive_name(raw_name)
                        if name in seen:
                            raise TransportError(f"Duplicate transfer path: {name}")
                        seen.add(name)
                        self._add_stream_file(archive, name, path)
            except (BrokenPipeError, OSError, tarfile.TarError, TransportError) as exc:
                write_error = exc
            finally:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
            returncode = process.wait()
            diagnostic = self._stderr_tail(stderr)
            if returncode != 0:
                raise TransportError(
                    f"Stream transfer to node {node.id} failed ({returncode}): "
                    f"{diagnostic}"
                ) from write_error
            if write_error is not None:
                raise TransportError(
                    f"Stream transfer to node {node.id} failed: {write_error}"
                ) from write_error

    @classmethod
    def _partition_entries(
        cls, source_root: Path, files_list_path: Path
    ) -> list[tuple[str, Path]]:
        if not source_root.is_dir() or source_root.is_symlink():
            raise TransportError(f"Source root is unsafe or missing: {source_root}")
        if not files_list_path.is_file() or files_list_path.is_symlink():
            raise TransportError(f"Partition files list is unsafe: {files_list_path}")
        entries: list[tuple[str, Path]] = []
        for encoded in files_list_path.read_bytes().split(b"\0"):
            if not encoded:
                continue
            name = cls._safe_archive_name(os.fsdecode(encoded))
            relative = PurePosixPath(name)
            current = source_root
            for part in relative.parts[:-1]:
                current /= part
                try:
                    info = current.lstat()
                except OSError as exc:
                    raise TransportError(
                        f"Cannot inspect transfer source directory {current}: {exc}"
                    ) from exc
                if not stat.S_ISDIR(info.st_mode) or current.is_symlink():
                    raise TransportError(f"Unsafe transfer source directory: {current}")
            entries.append((name, source_root.joinpath(*relative.parts)))
        return entries

    @staticmethod
    def _safe_local_directory(root: Path, name: str) -> Path:
        relative = PurePosixPath(name)
        current = root
        for part in relative.parts[:-1]:
            current /= part
            try:
                info = current.lstat()
            except FileNotFoundError:
                current.mkdir(mode=0o700)
                info = current.lstat()
            if not stat.S_ISDIR(info.st_mode) or current.is_symlink():
                raise TransportError(f"Unsafe local result directory: {current}")
        return current

    @classmethod
    def _extract_stream_result(
        cls,
        handle: BinaryIO,
        root: Path,
        expected: dict[str, int],
    ) -> None:
        seen: set[str] = set()
        try:
            with tarfile.open(fileobj=handle, mode="r|") as archive:
                for member in archive:
                    name = cls._safe_archive_name(member.name)
                    if not member.isfile() or member.sparse is not None:
                        raise TransportError(
                            f"Result stream member is not a regular file: {name}"
                        )
                    if name in seen:
                        raise TransportError(f"Duplicate result stream member: {name}")
                    expected_size = expected.get(name)
                    if expected_size is None:
                        raise TransportError(f"Unexpected result stream member: {name}")
                    if member.size != expected_size:
                        raise TransportError(
                            f"Result stream member size does not match: {name}"
                        )
                    source = archive.extractfile(member)
                    if source is None:
                        raise TransportError(
                            f"Cannot read result stream member: {name}"
                        )
                    parent = cls._safe_local_directory(root, name)
                    target = root.joinpath(*PurePosixPath(name).parts)
                    descriptor, temporary_name = tempfile.mkstemp(
                        prefix=f".{target.name}.stream-", dir=parent
                    )
                    try:
                        with os.fdopen(descriptor, "wb") as destination:
                            remaining = member.size
                            while remaining:
                                chunk = source.read(min(1024 * 1024, remaining))
                                if not chunk:
                                    raise TransportError(
                                        f"Truncated result stream member: {name}"
                                    )
                                destination.write(chunk)
                                remaining -= len(chunk)
                            destination.flush()
                            os.fsync(destination.fileno())
                        os.replace(temporary_name, target)
                    except Exception:
                        try:
                            os.unlink(temporary_name)
                        except FileNotFoundError:
                            pass
                        raise
                    seen.add(name)
        except (tarfile.TarError, OSError) as exc:
            raise TransportError(f"Invalid result tar stream: {exc}") from exc
        missing = set(expected) - seen
        if missing:
            raise TransportError(
                f"Result stream is missing expected member: {min(missing)!r}"
            )

    def _run_stream_pull(
        self, node: NodeConfig, remote_attempt_root: str, local_partial: Path
    ) -> None:
        self._check_ssh_files()
        attempt_root = str(
            self._safe_stream_path(remote_attempt_root, "remote attempt root")
        )
        if local_partial.is_symlink():
            raise TransportError(f"Local result path is a symlink: {local_partial}")
        if local_partial.exists():
            if not local_partial.is_dir():
                raise TransportError(
                    f"Local result path is not a directory: {local_partial}"
                )
            shutil.rmtree(local_partial)
        local_partial.mkdir(parents=True, mode=0o700)
        with tempfile.TemporaryFile() as stderr:
            try:
                process = subprocess.Popen(
                    self._stream_command(node, "stream-pull"),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    shell=False,
                    env=self._safe_environment(),
                )
            except OSError as exc:
                raise TransportError(
                    f"Cannot start result stream from node {node.id}: {exc}"
                ) from exc
            read_error: Exception | None = None
            try:
                if process.stdin is None or process.stdout is None:
                    raise TransportError("Result stream pipes are unavailable")
                self._stream_write_header(
                    process.stdin,
                    {
                        "protocol": STREAM_PROTOCOL_VERSION,
                        "attempt_root": attempt_root,
                    },
                )
                process.stdin.close()
                response = self._stream_read_header(process.stdout)
                if set(response) != {"protocol", "attempt_id", "files"}:
                    raise TransportError("Invalid result stream header fields")
                if response.get("protocol") != STREAM_PROTOCOL_VERSION:
                    raise TransportError("Unsupported result stream protocol version")
                files = response.get("files")
                if not isinstance(files, list):
                    raise TransportError("Result stream header has no files list")
                expected: dict[str, int] = {}
                for item in files:
                    if not isinstance(item, dict) or set(item) != {"path", "size"}:
                        raise TransportError("Invalid result stream file metadata")
                    name = self._safe_archive_name(item.get("path"))
                    size = item.get("size")
                    if (
                        name in expected
                        or isinstance(size, bool)
                        or not isinstance(size, int)
                        or size < 0
                    ):
                        raise TransportError("Invalid result stream file metadata")
                    expected[name] = size
                self._extract_stream_result(process.stdout, local_partial, expected)
            except (BrokenPipeError, OSError, TransportError) as exc:
                read_error = exc
            finally:
                if process.stdin is not None and not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
                if process.stdout is not None:
                    process.stdout.close()
            returncode = process.wait()
            diagnostic = self._stderr_tail(stderr)
            if returncode != 0:
                raise TransportError(
                    f"Result stream from node {node.id} failed ({returncode}): "
                    f"{diagnostic}"
                ) from read_error
            if read_error is not None:
                raise TransportError(
                    f"Result stream from node {node.id} failed: {read_error}"
                ) from read_error

    def push_partition_files(
        self,
        node: NodeConfig,
        source_root: Path,
        files_list_path: Path,
        remote_data_partial: str,
    ) -> None:
        if not source_root.is_dir():
            raise TransportError(f"Source root is missing: {source_root}")
        if not files_list_path.is_file():
            raise TransportError(f"Partition files list is missing: {files_list_path}")
        if self._transfer_mode(node) == "stream":
            attempt_root = self._stream_data_location(remote_data_partial)
            self._run_stream_push(
                node,
                {
                    "protocol": STREAM_PROTOCOL_VERSION,
                    "attempt_root": attempt_root,
                    "role": "data",
                },
                self._partition_entries(source_root, files_list_path),
            )
            return
        with files_list_path.open("rb") as files_list:
            self._run_rsync(
                node,
                [
                    "--recursive",
                    "--times",
                    "--from0",
                    "--files-from=-",
                    "--relative",
                    "--",
                    f"{source_root}/",
                    f"{self._remote(node, remote_data_partial)}/",
                ],
                stdin=files_list,
                timeout=None,
            )

    def push_file(self, node: NodeConfig, source: Path, remote_path: str) -> None:
        if not source.is_file() or source.is_symlink():
            raise TransportError(f"Transfer source is not a regular file: {source}")
        if self._transfer_mode(node) == "stream":
            attempt_root, role = self._stream_push_location(remote_path)
            self._run_stream_push(
                node,
                {
                    "protocol": STREAM_PROTOCOL_VERSION,
                    "attempt_root": attempt_root,
                    "role": role,
                },
                ((PurePosixPath(remote_path).name, source),),
            )
            return
        self._run_rsync(
            node,
            ["--times", "--", str(source), self._remote(node, remote_path)],
            timeout=None,
        )

    def pull_attempt(
        self, node: NodeConfig, remote_attempt_root: str, local_partial: Path
    ) -> None:
        if self._transfer_mode(node) == "stream":
            self._run_stream_pull(node, remote_attempt_root, local_partial)
            return
        local_partial.mkdir(parents=True, exist_ok=True)
        try:
            self._run_rsync(
                node,
                [
                    "--recursive",
                    "--times",
                    "--",
                    f"{self._remote(node, remote_attempt_root)}/",
                    f"{local_partial}/",
                ],
                timeout=None,
            )
        except Exception:
            # The partial directory is intentionally retained for rsync resume.
            raise


def executable_prerequisites() -> dict[str, str | None]:
    return {
        "ssh": shutil.which("ssh"),
        "rsync": shutil.which("rsync"),
    }
