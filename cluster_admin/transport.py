"""Pinned-host SSH RPC and rsync data transport."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, BinaryIO

from .config import ClusterConfig, NodeConfig


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

    def bootstrap(self, node: NodeConfig) -> dict[str, Any]:
        """Install the standalone runner through one constrained SSH command."""
        self._check_ssh_files()
        source = Path(__file__).with_name("node_runner.py")
        runner = Path(node.runner_path)
        temporary = runner.with_suffix(runner.suffix + ".partial")
        # Every path component came through the remote-path allowlist in config.py.
        remote_command = " && ".join(
            (
                "umask 077",
                shlex.join(["mkdir", "-p", str(runner.parent)]),
                f"cat > {shlex.quote(str(temporary))}",
                shlex.join(["python3", "-m", "py_compile", str(temporary)]),
                shlex.join(["chmod", "700", str(temporary)]),
                shlex.join(["mv", "-f", str(temporary), str(runner)]),
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
        return {"ok": True, "runner_path": node.runner_path}

    def probe(self, node: NodeConfig) -> dict[str, Any]:
        return self.rpc(
            node,
            {
                "operation": "probe",
                "work_root": node.work_root,
                "image": node.image or self.config.image,
                "models_root": node.models_root,
                "cache_root": node.cache_root,
            },
        )

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
        self._run_rsync(
            node,
            ["--times", "--", str(source), self._remote(node, remote_path)],
            timeout=None,
        )

    def pull_attempt(
        self, node: NodeConfig, remote_attempt_root: str, local_partial: Path
    ) -> None:
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
