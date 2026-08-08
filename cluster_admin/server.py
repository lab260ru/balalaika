"""Read-only HTTP dashboard for the cluster controller database."""

from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .config import ClusterConfig, validate_slug
from .db import StateDB


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "BalalaikaCluster/0.1"

    @property
    def state_db(self) -> StateDB:
        return self.server.state_db  # type: ignore[attr-defined]

    @property
    def static_root(self) -> Path:
        return self.server.static_root  # type: ignore[attr-defined]

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
            self._json(self.state_db.overview(run_id))
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


def build_server(
    config: ClusterConfig, host: str = "127.0.0.1", port: int = 8765
) -> ThreadingHTTPServer:
    database = StateDB(config.db_path)
    database.initialize()
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    server.state_db = database  # type: ignore[attr-defined]
    server.static_root = Path(__file__).with_name("static")  # type: ignore[attr-defined]
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
