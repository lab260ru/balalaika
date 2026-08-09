"""Local HTTP dashboard for the cluster controller database."""

from __future__ import annotations

import ipaddress
import json
import mimetypes
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .config import ClusterConfig, validate_slug
from .db import StateDB, TERMINAL_PARTITION_STATES
from .scheduler import ClusterScheduler

MAX_REQUEST_BYTES = 4096


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "BalalaikaCluster/0.1"

    @property
    def state_db(self) -> StateDB:
        return self.server.state_db  # type: ignore[attr-defined]

    @property
    def static_root(self) -> Path:
        return self.server.static_root  # type: ignore[attr-defined]

    @property
    def scheduler(self) -> ClusterScheduler:
        return self.server.scheduler  # type: ignore[attr-defined]

    @property
    def csrf_token(self) -> str:
        return self.server.csrf_token  # type: ignore[attr-defined]

    def log_message(self, format: str, *args) -> None:
        return

    def _headers(self, status: HTTPStatus, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self._headers(status, "application/json; charset=utf-8", len(payload))
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _overview(self, run_id: str | None) -> dict[str, Any]:
        overview = self.state_db.overview(run_id)
        controller = dict(overview.get("controller") or {})
        controller["csrf_token"] = self.csrf_token
        overview["controller"] = controller
        return overview

    def _same_origin(self) -> bool:
        origin = self.headers.get("Origin", "")
        host = self.headers.get("Host", "")
        parsed = urlsplit(origin)
        return bool(
            parsed.scheme in {"http", "https"}
            and parsed.netloc
            and parsed.netloc.lower() == host.lower()
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
            and parsed.username is None
            and parsed.password is None
        )

    def _authorize_post(self) -> bool:
        supplied_token = self.headers.get("X-CSRF-Token", "")
        if not self._same_origin() or not secrets.compare_digest(
            supplied_token, self.csrf_token
        ):
            self._json(
                {"error": "same-origin request with a valid CSRF token required"},
                HTTPStatus.FORBIDDEN,
            )
            return False
        return True

    def _read_json_object(self) -> dict[str, Any] | None:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
        if content_type.lower() != "application/json":
            self._json(
                {"error": "Content-Type must be application/json"},
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            )
            return None
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._json(
                {"error": "valid Content-Length required"}, HTTPStatus.BAD_REQUEST
            )
            return None
        if length <= 0 or length > MAX_REQUEST_BYTES:
            if length > MAX_REQUEST_BYTES:
                self.close_connection = True
            self._json(
                {"error": f"request body must be 1-{MAX_REQUEST_BYTES} bytes"},
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE
                if length > MAX_REQUEST_BYTES
                else HTTPStatus.BAD_REQUEST,
            )
            return None
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json(
                {"error": "request body must be valid JSON"}, HTTPStatus.BAD_REQUEST
            )
            return None
        if not isinstance(value, dict):
            self._json(
                {"error": "request body must be a JSON object"},
                HTTPStatus.BAD_REQUEST,
            )
            return None
        return value

    def _cancel_partition(
        self, run_id: str, partition_id: str, body: dict[str, Any]
    ) -> None:
        try:
            run_id = validate_slug(run_id, "run id")
            partition_id = validate_slug(partition_id, "partition id")
            node_id = validate_slug(body.get("node_id"), "node id")
        except (TypeError, ValueError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if set(body) != {"node_id"}:
            self._json(
                {"error": "request body must contain only node_id"},
                HTTPStatus.BAD_REQUEST,
            )
            return

        partition = self.state_db.get_partition(run_id, partition_id)
        if partition is None:
            self._json({"error": "partition not found"}, HTTPStatus.NOT_FOUND)
            return
        if partition["state"] in TERMINAL_PARTITION_STATES:
            if partition["state"] == "CANCELLED":
                node = self.state_db.get_node(node_id)
                drained = bool(node and node.get("drained"))
                if drained:
                    self._json(
                        {
                            "ok": True,
                            "run_id": run_id,
                            "partition_id": partition_id,
                            "node_id": node_id,
                            "state": "CANCELLED",
                            "cancellation_requested": True,
                            "node_drained": True,
                            "drained_node_id": node_id,
                        }
                    )
                else:
                    self._json(
                        {
                            "error": "partition is cancelled but node is not drained; "
                            "refresh controller state"
                        },
                        HTTPStatus.CONFLICT,
                    )
            else:
                self._json(
                    {"error": f"partition is already {partition['state']}"},
                    HTTPStatus.CONFLICT,
                )
            return
        if partition.get("node_id") != node_id:
            self._json(
                {"error": "partition is no longer assigned to this node"},
                HTTPStatus.CONFLICT,
            )
            return
        node = next(
            (item for item in self.state_db.list_nodes() if item["id"] == node_id),
            None,
        )
        if node is None:
            self._json({"error": "node not found"}, HTTPStatus.NOT_FOUND)
            return
        if node.get("current_partition_id") != partition_id:
            self._json(
                {"error": "node is no longer running this partition"},
                HTTPStatus.CONFLICT,
            )
            return

        try:
            drained_node = self.state_db.set_node_drained(
                node_id,
                True,
                expected_run_id=run_id,
                expected_partition_id=partition_id,
            )
        except KeyError:
            self._json({"error": "node not found"}, HTTPStatus.NOT_FOUND)
            return
        except RuntimeError as exc:
            self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            return

        try:
            affected = self.state_db.request_cancel_partition(run_id, partition_id)
        except KeyError:
            self._json({"error": "partition not found"}, HTTPStatus.NOT_FOUND)
            return

        current = self.state_db.get_partition(run_id, partition_id) or partition
        response: dict[str, Any] = {
            "ok": True,
            "run_id": run_id,
            "partition_id": partition_id,
            "node_id": node_id,
            "state": current["state"],
            "desired_state": current.get("desired_state"),
            "cancellation_requested": True,
            "affected": affected,
            "node_drained": bool(drained_node["drained"]),
            "drained_node_id": node_id,
        }
        self._json(response, HTTPStatus.ACCEPTED)

    def _set_node_drained(
        self, node_id: str, drained: bool, body: dict[str, Any]
    ) -> None:
        try:
            node_id = validate_slug(node_id, "node id")
        except (TypeError, ValueError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if body:
            self._json(
                {"error": "request body must be an empty JSON object"},
                HTTPStatus.BAD_REQUEST,
            )
            return
        try:
            node = self.state_db.set_node_drained(node_id, drained)
        except KeyError:
            self._json({"error": "node not found"}, HTTPStatus.NOT_FOUND)
            return
        self._json(
            {
                "ok": True,
                "node_id": node_id,
                "drained": bool(node["drained"]),
                "enabled": bool(node["enabled"]),
            }
        )

    def _static(self, relative: str) -> None:
        allowed = {
            "index.html": "index.html",
            "app.js": "app.js",
            "styles.css": "styles.css",
        }
        filename = allowed.get(relative)
        if filename is None:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        path = self.static_root / filename
        try:
            payload = path.read_bytes()
        except OSError:
            self._json({"error": "dashboard asset is missing"}, HTTPStatus.NOT_FOUND)
            return
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if media_type.startswith("text/") or media_type == "application/javascript":
            media_type += "; charset=utf-8"
        self._headers(HTTPStatus.OK, media_type, len(payload))
        if self.command != "HEAD":
            self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/api/v1/health":
            self._json({"ok": True})
            return
        if parsed.path == "/api/v1/overview":
            run_id = parse_qs(parsed.query).get("run_id", [None])[0]
            if run_id is not None:
                try:
                    validate_slug(run_id, "run id")
                except ValueError as exc:
                    self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
            self._json(self._overview(run_id))
            return
        if parsed.path in {"/", "/index.html"}:
            self._static("index.html")
            return
        if parsed.path in {"/app.js", "/static/app.js"}:
            self._static("app.js")
            return
        if parsed.path in {"/styles.css", "/static/styles.css"}:
            self._static("styles.css")
            return
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        cancel_partition = (
            len(parts) == 7
            and parts[:3] == ["api", "v1", "runs"]
            and parts[4] == "partitions"
            and parts[6] == "cancel"
        )
        update_node = (
            len(parts) == 5
            and parts[:3] == ["api", "v1", "nodes"]
            and parts[4] in {"drain", "resume"}
        )
        if parsed.query or not (cancel_partition or update_node):
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        if not self._authorize_post():
            return
        body = self._read_json_object()
        if body is None:
            return
        if cancel_partition:
            self._cancel_partition(parts[3], parts[5], body)
        else:
            self._set_node_drained(parts[3], parts[4] == "drain", body)


def build_server(
    config: ClusterConfig,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    scheduler: ClusterScheduler | None = None,
) -> ThreadingHTTPServer:
    try:
        bind_address = ipaddress.ip_address(host)
    except ValueError:
        if host != "localhost":
            raise ValueError(
                "Dashboard may bind only to localhost; use an SSH tunnel"
            )
    else:
        if not bind_address.is_loopback:
            raise ValueError(
                "Dashboard may bind only to a loopback address; use an SSH tunnel"
            )
    database = StateDB(config.db_path)
    database.initialize()
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    server.state_db = database  # type: ignore[attr-defined]
    server.scheduler = scheduler or ClusterScheduler(  # type: ignore[attr-defined]
        config, database=database
    )
    server.csrf_token = secrets.token_urlsafe(32)  # type: ignore[attr-defined]
    server.static_root = Path(__file__).with_name(  # type: ignore[attr-defined]
        "static"
    )
    return server


def serve_panel(
    config: ClusterConfig, host: str = "127.0.0.1", port: int = 8765
) -> None:
    server = build_server(config, host, port)
    try:
        print(f"Balalaika cluster dashboard: http://{host}:{server.server_port}")
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
