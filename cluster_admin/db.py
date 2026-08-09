from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ACTIVE_ATTEMPT_STATES = (
    "ASSIGNED",
    "STAGING",
    "READY",
    "STARTING",
    "RUNNING",
    "COLLECTING",
    "VERIFYING",
    "UNKNOWN",
)
TERMINAL_PARTITION_STATES = ("SUCCEEDED", "FAILED", "CANCELLED")
TERMINAL_ATTEMPT_STATES = ("COMPLETED", "FAILED", "CANCELLED")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA synchronous=FULL;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    host TEXT NOT NULL,
    user TEXT NOT NULL,
    port INTEGER NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    drained INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'UNKNOWN',
    last_seen_at TEXT,
    current_partition_id TEXT,
    details_json TEXT NOT NULL DEFAULT '{}',
    error TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    state TEXT NOT NULL,
    desired_state TEXT NOT NULL DEFAULT 'RUNNING',
    source_root TEXT NOT NULL,
    image TEXT NOT NULL,
    config_path TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    execution_config_sha256 TEXT,
    execution_config_version INTEGER,
    execution_config_json TEXT,
    stage_start TEXT NOT NULL,
    stage_stop TEXT NOT NULL,
    partitions_total INTEGER NOT NULL,
    input_bytes INTEGER NOT NULL,
    audio_seconds REAL NOT NULL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS partitions (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    state TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    files_list_path TEXT NOT NULL,
    state_fragment_path TEXT,
    files_total INTEGER NOT NULL,
    input_bytes INTEGER NOT NULL,
    audio_seconds REAL NOT NULL,
    weight REAL NOT NULL,
    desired_state TEXT NOT NULL DEFAULT 'RUNNING',
    node_id TEXT REFERENCES nodes(id),
    current_attempt_id TEXT,
    current_stage TEXT,
    progress_percent REAL NOT NULL DEFAULT 0,
    files_processed INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    updated_at TEXT NOT NULL,
    error TEXT,
    PRIMARY KEY (run_id, id)
);

CREATE TABLE IF NOT EXISTS attempts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    partition_id TEXT NOT NULL,
    node_id TEXT NOT NULL REFERENCES nodes(id),
    ordinal INTEGER NOT NULL,
    fencing_token TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL,
    remote_root TEXT NOT NULL,
    node_host TEXT,
    node_user TEXT,
    node_port INTEGER,
    gpu_uuids_json TEXT,
    container_name TEXT,
    started_at TEXT,
    updated_at TEXT NOT NULL,
    exit_code INTEGER,
    result_sha256 TEXT,
    error TEXT,
    FOREIGN KEY (run_id, partition_id)
      REFERENCES partitions(run_id, id) ON DELETE CASCADE
);

DROP INDEX IF EXISTS one_active_attempt_per_node;
CREATE UNIQUE INDEX one_active_attempt_per_node
ON attempts(node_id)
WHERE state IN (
  'ASSIGNED', 'STAGING', 'READY', 'STARTING', 'RUNNING',
  'COLLECTING', 'VERIFYING', 'UNKNOWN'
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    run_id TEXT,
    partition_id TEXT,
    node_id TEXT,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}'
);
"""


class StateDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            node_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(nodes)")
            }
            if "drained" not in node_columns:
                connection.execute(
                    "ALTER TABLE nodes ADD COLUMN drained INTEGER NOT NULL DEFAULT 0"
                )
            run_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(runs)")
            }
            if "desired_state" not in run_columns:
                connection.execute(
                    "ALTER TABLE runs ADD COLUMN desired_state TEXT NOT NULL "
                    "DEFAULT 'RUNNING'"
                )
            if "execution_config_sha256" not in run_columns:
                connection.execute(
                    "ALTER TABLE runs ADD COLUMN execution_config_sha256 TEXT"
                )
            if "execution_config_version" not in run_columns:
                connection.execute(
                    "ALTER TABLE runs ADD COLUMN execution_config_version INTEGER"
                )
            if "execution_config_json" not in run_columns:
                connection.execute(
                    "ALTER TABLE runs ADD COLUMN execution_config_json TEXT"
                )
            partition_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(partitions)")
            }
            if "desired_state" not in partition_columns:
                connection.execute(
                    "ALTER TABLE partitions ADD COLUMN desired_state TEXT NOT NULL "
                    "DEFAULT 'RUNNING'"
                )
            attempt_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(attempts)")
            }
            for column, definition in (
                ("node_host", "TEXT"),
                ("node_user", "TEXT"),
                ("node_port", "INTEGER"),
                ("gpu_uuids_json", "TEXT"),
            ):
                if column not in attempt_columns:
                    connection.execute(
                        f"ALTER TABLE attempts ADD COLUMN {column} {definition}"
                    )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def sync_nodes(self, nodes) -> None:
        now = utc_now()
        with self.connect() as connection:
            for node in nodes:
                connection.execute(
                    """
                    INSERT INTO nodes(id, host, user, port, enabled, state, details_json)
                    VALUES (?, ?, ?, ?, ?, 'UNKNOWN', '{}')
                    ON CONFLICT(id) DO UPDATE SET
                      host=excluded.host, user=excluded.user, port=excluded.port,
                      enabled=excluded.enabled
                    """,
                    (node.id, node.host, node.user, node.port, int(node.enabled)),
                )
            connection.execute(
                "INSERT INTO events(created_at, level, message) VALUES (?, 'INFO', ?)",
                (now, f"Synchronized {len(nodes)} configured nodes"),
            )

    def create_run(self, run: dict[str, Any], partitions: list[dict[str, Any]]) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO runs(
                  id, created_at, updated_at, state, desired_state, source_root, image,
                  config_path, config_sha256, execution_config_sha256,
                  execution_config_version, execution_config_json,
                  stage_start, stage_stop,
                  partitions_total, input_bytes, audio_seconds
                ) VALUES (
                  ?, ?, ?, 'PLANNED', 'RUNNING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    run["id"],
                    now,
                    now,
                    run["source_root"],
                    run["image"],
                    run["config_path"],
                    run["config_sha256"],
                    run.get("execution_config_sha256"),
                    run.get("execution_config_version"),
                    run.get("execution_config_json"),
                    run["stage_start"],
                    run["stage_stop"],
                    len(partitions),
                    run["input_bytes"],
                    run["audio_seconds"],
                ),
            )
            for partition in partitions:
                connection.execute(
                    """
                    INSERT INTO partitions(
                      run_id, id, ordinal, state, manifest_path, manifest_sha256,
                      files_list_path, state_fragment_path, files_total,
                      input_bytes, audio_seconds, weight, updated_at
                    ) VALUES (?, ?, ?, 'QUEUED', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run["id"],
                        partition["id"],
                        partition["ordinal"],
                        partition["manifest_path"],
                        partition["manifest_sha256"],
                        partition["files_list_path"],
                        partition.get("state_fragment_path"),
                        partition["files_total"],
                        partition["input_bytes"],
                        partition["audio_seconds"],
                        partition["weight"],
                        now,
                    ),
                )
            connection.execute(
                """INSERT INTO events(created_at, run_id, level, message, details_json)
                   VALUES (?, ?, 'INFO', ?, ?)""",
                (
                    now,
                    run["id"],
                    f"Created run with {len(partitions)} partitions",
                    json.dumps({"input_bytes": run["input_bytes"]}),
                ),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE id=?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_runs(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_nodes(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM nodes ORDER BY id").fetchall()
        return [self._decode_node(row) for row in rows]

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM nodes WHERE id=?", (node_id,)
            ).fetchone()
        return self._decode_node(row) if row else None

    @staticmethod
    def _decode_node(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["enabled"] = bool(item["enabled"])
        item["drained"] = bool(item["drained"])
        item["details"] = json.loads(item.pop("details_json") or "{}")
        return item

    def set_node_drained(
        self,
        node_id: str,
        drained: bool,
        *,
        expected_run_id: str | None = None,
        expected_partition_id: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT drained, current_partition_id FROM nodes WHERE id=?",
                (node_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown node: {node_id}")
            if expected_run_id is not None and expected_partition_id is not None:
                partition = connection.execute(
                    """SELECT state, node_id FROM partitions
                       WHERE run_id=? AND id=?""",
                    (expected_run_id, expected_partition_id),
                ).fetchone()
                if (
                    partition is None
                    or partition["state"] in TERMINAL_PARTITION_STATES
                    or partition["node_id"] != node_id
                ):
                    raise RuntimeError(
                        f"Partition {expected_run_id}/{expected_partition_id} "
                        f"is no longer active on node {node_id}"
                    )
            if (
                expected_partition_id is not None
                and row["current_partition_id"] != expected_partition_id
            ):
                raise RuntimeError(
                    f"Node {node_id} is no longer running partition "
                    f"{expected_partition_id}"
                )
            changed = bool(row["drained"]) != drained
            connection.execute(
                "UPDATE nodes SET drained=? WHERE id=?", (int(drained), node_id)
            )
            if changed:
                connection.execute(
                    """INSERT INTO events(
                         created_at, node_id, level, message, details_json
                       ) VALUES (?, ?, 'INFO', ?, ?)""",
                    (
                        now,
                        node_id,
                        "Node drained" if drained else "Node resumed",
                        json.dumps({"drained": drained}),
                    ),
                )
            updated = connection.execute(
                "SELECT * FROM nodes WHERE id=?", (node_id,)
            ).fetchone()
        if updated is None:  # pragma: no cover - protected by the write transaction
            raise KeyError(f"Unknown node: {node_id}")
        return self._decode_node(updated)

    def list_partitions(self, run_id: str | None = None) -> list[dict[str, Any]]:
        query = (
            "SELECT partitions.*, attempts.ordinal AS attempt "
            "FROM partitions LEFT JOIN attempts "
            "ON attempts.id=partitions.current_attempt_id"
        )
        params: tuple[Any, ...] = ()
        if run_id:
            query += " WHERE partitions.run_id=?"
            params = (run_id,)
        query += " ORDER BY partitions.run_id DESC, partitions.ordinal"
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_partition(self, run_id: str, partition_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM partitions WHERE run_id=? AND id=?",
                (run_id, partition_id),
            ).fetchone()
        return dict(row) if row else None

    def next_queued_partition(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """SELECT * FROM partitions WHERE run_id=? AND state='QUEUED'
                   AND desired_state='RUNNING'
                   ORDER BY weight DESC, ordinal LIMIT 1""",
                (run_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_active_attempts(self, run_id: str | None = None) -> list[dict[str, Any]]:
        placeholders = ",".join("?" for _ in ACTIVE_ATTEMPT_STATES)
        parameters: list[Any] = list(ACTIVE_ATTEMPT_STATES)
        query = f"SELECT * FROM attempts WHERE state IN ({placeholders})"
        if run_id:
            query += " AND run_id=?"
            parameters.append(run_id)
        query += " ORDER BY updated_at, id"
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def get_attempt(self, attempt_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM attempts WHERE id=?", (attempt_id,)
            ).fetchone()
        return dict(row) if row else None

    def active_attempt_for_node(self, node_id: str) -> dict[str, Any] | None:
        placeholders = ",".join("?" for _ in ACTIVE_ATTEMPT_STATES)
        with self.connect() as connection:
            row = connection.execute(
                f"SELECT * FROM attempts WHERE node_id=? AND state IN ({placeholders}) "
                "ORDER BY updated_at DESC LIMIT 1",
                (node_id, *ACTIVE_ATTEMPT_STATES),
            ).fetchone()
        return dict(row) if row else None

    def claim_next(
        self,
        run_id: str,
        node_id: str,
        remote_root: str,
        *,
        node_host: str | None = None,
        node_user: str | None = None,
        node_port: int | None = None,
        gpu_uuids: list[str] | None = None,
    ) -> dict[str, Any] | None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            node = connection.execute(
                "SELECT enabled, drained FROM nodes WHERE id=?", (node_id,)
            ).fetchone()
            if node is None or not bool(node["enabled"]) or bool(node["drained"]):
                return None
            active = connection.execute(
                f"SELECT 1 FROM attempts WHERE node_id=? AND state IN "
                f"({','.join('?' for _ in ACTIVE_ATTEMPT_STATES)})",
                (node_id, *ACTIVE_ATTEMPT_STATES),
            ).fetchone()
            if active:
                return None
            desired = connection.execute(
                "SELECT desired_state FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if not desired or desired["desired_state"] != "RUNNING":
                return None
            partition = connection.execute(
                """SELECT * FROM partitions WHERE run_id=? AND state='QUEUED'
                   AND desired_state='RUNNING'
                   ORDER BY weight DESC, ordinal LIMIT 1""",
                (run_id,),
            ).fetchone()
            if not partition:
                return None
            previous = connection.execute(
                "SELECT COUNT(*) AS count FROM attempts WHERE run_id=? AND partition_id=?",
                (run_id, partition["id"]),
            ).fetchone()["count"]
            attempt_id = uuid.uuid4().hex
            token = uuid.uuid4().hex
            ordinal = int(previous) + 1
            connection.execute(
                """INSERT INTO attempts(
                     id, run_id, partition_id, node_id, ordinal, fencing_token,
                     state, remote_root, node_host, node_user, node_port,
                     gpu_uuids_json, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, 'ASSIGNED', ?, ?, ?, ?, ?, ?)""",
                (
                    attempt_id,
                    run_id,
                    partition["id"],
                    node_id,
                    ordinal,
                    token,
                    remote_root,
                    node_host,
                    node_user,
                    node_port,
                    json.dumps(gpu_uuids) if gpu_uuids is not None else None,
                    now,
                ),
            )
            connection.execute(
                """UPDATE partitions SET state='ASSIGNED', node_id=?,
                   current_attempt_id=?, updated_at=?, error=NULL
                   WHERE run_id=? AND id=?""",
                (node_id, attempt_id, now, run_id, partition["id"]),
            )
            connection.execute(
                "UPDATE nodes SET current_partition_id=? WHERE id=?",
                (partition["id"], node_id),
            )
            connection.execute(
                """INSERT INTO events(
                     created_at, run_id, partition_id, node_id, level, message,
                     details_json
                   ) VALUES (?, ?, ?, ?, 'INFO', ?, ?)""",
                (
                    now,
                    run_id,
                    partition["id"],
                    node_id,
                    f"Assigned partition to {node_id}",
                    json.dumps({"attempt_id": attempt_id, "attempt": ordinal}),
                ),
            )
            result = dict(partition)
            result.update(
                {
                    "attempt_id": attempt_id,
                    "attempt_ordinal": ordinal,
                    "fencing_token": token,
                    "node_id": node_id,
                    "remote_root": remote_root,
                }
            )
            return result

    def update_node(
        self,
        node_id: str,
        state: str,
        details: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self.connect() as connection:
            current = connection.execute(
                "SELECT details_json FROM nodes WHERE id=?", (node_id,)
            ).fetchone()
            merged: dict[str, Any] = {}
            if current:
                try:
                    merged.update(json.loads(current["details_json"] or "{}"))
                except json.JSONDecodeError:
                    pass
            if details:
                merged.update(details)
            connection.execute(
                """UPDATE nodes SET state=?, last_seen_at=?, details_json=?, error=?
                   WHERE id=?""",
                (state, utc_now(), json.dumps(merged), error, node_id),
            )

    def update_attempt(
        self,
        attempt_id: str,
        state: str,
        *,
        container_name: str | None = None,
        exit_code: int | None = None,
        error: str | None = None,
        current_stage: str | None = None,
        progress_percent: float | None = None,
        files_processed: int | None = None,
    ) -> bool:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT attempts.run_id, attempts.partition_id, attempts.node_id,
                          attempts.state AS attempt_state,
                          partitions.current_attempt_id
                   FROM attempts JOIN partitions
                     ON partitions.run_id=attempts.run_id
                    AND partitions.id=attempts.partition_id
                   WHERE attempts.id=?""",
                (attempt_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"Unknown attempt: {attempt_id}")
            if row["current_attempt_id"] != attempt_id:
                raise RuntimeError(f"Attempt is no longer current: {attempt_id}")
            current_state = row["attempt_state"]
            if current_state in TERMINAL_ATTEMPT_STATES:
                return current_state == state
            allowed = {
                "ASSIGNED": {"ASSIGNED", "STAGING", "UNKNOWN", "CANCELLED"},
                "STAGING": {"STAGING", "READY", "UNKNOWN", "CANCELLED"},
                "READY": {"READY", "STARTING", "UNKNOWN", "CANCELLED"},
                "STARTING": {
                    "STARTING",
                    "RUNNING",
                    "COLLECTING",
                    "UNKNOWN",
                    "CANCELLED",
                },
                "RUNNING": {"RUNNING", "COLLECTING", "UNKNOWN", "CANCELLED"},
                "COLLECTING": {"COLLECTING", "VERIFYING", "UNKNOWN", "CANCELLED"},
                "VERIFYING": {"VERIFYING", "COMPLETED", "UNKNOWN", "CANCELLED"},
                "UNKNOWN": {
                    "UNKNOWN",
                    "STAGING",
                    "READY",
                    "STARTING",
                    "RUNNING",
                    "COLLECTING",
                    "VERIFYING",
                    "COMPLETED",
                    "CANCELLED",
                },
            }
            if state not in allowed.get(current_state, set()):
                raise RuntimeError(
                    f"Invalid attempt transition: {current_state} -> {state}"
                )
            started = now if state == "RUNNING" else None
            connection.execute(
                """UPDATE attempts SET state=?, container_name=COALESCE(?, container_name),
                   started_at=COALESCE(started_at, ?), updated_at=?, exit_code=?, error=?
                   WHERE id=?""",
                (state, container_name, started, now, exit_code, error, attempt_id),
            )
            partition_state = state
            if state == "COMPLETED":
                partition_state = "SUCCEEDED"
            connection.execute(
                """UPDATE partitions SET state=?, current_stage=COALESCE(?, current_stage),
                   progress_percent=COALESCE(?, progress_percent),
                   files_processed=COALESCE(?, files_processed),
                   started_at=COALESCE(started_at, ?), updated_at=?, error=?
                   WHERE run_id=? AND id=? AND current_attempt_id=?""",
                (
                    partition_state,
                    current_stage,
                    progress_percent,
                    files_processed,
                    started,
                    now,
                    error,
                    row["run_id"],
                    row["partition_id"],
                    attempt_id,
                ),
            )
            if partition_state in TERMINAL_PARTITION_STATES:
                connection.execute(
                    "UPDATE nodes SET current_partition_id=NULL WHERE id=?",
                    (row["node_id"],),
                )
            if current_state != state or error:
                connection.execute(
                    """INSERT INTO events(
                         created_at, run_id, partition_id, node_id, level, message,
                         details_json
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        now,
                        row["run_id"],
                        row["partition_id"],
                        row["node_id"],
                        "ERROR" if error else "INFO",
                        f"Attempt state changed to {state}",
                        json.dumps({"attempt_id": attempt_id, "error": error}),
                    ),
                )
            self._refresh_run_state(connection, row["run_id"], now)
            return True

    def requeue_attempt(
        self, attempt_id: str, error: str, *, max_attempts: int | None = None
    ) -> str:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT attempts.run_id, attempts.partition_id, attempts.node_id,
                          attempts.ordinal, attempts.state AS attempt_state,
                          partitions.current_attempt_id
                   FROM attempts JOIN partitions
                     ON partitions.run_id=attempts.run_id
                    AND partitions.id=attempts.partition_id
                   WHERE attempts.id=?""",
                (attempt_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"Unknown attempt: {attempt_id}")
            if row["current_attempt_id"] != attempt_id:
                raise RuntimeError(f"Attempt is no longer current: {attempt_id}")
            if row["attempt_state"] in TERMINAL_ATTEMPT_STATES:
                raise RuntimeError(f"Attempt is already terminal: {attempt_id}")
            connection.execute(
                "UPDATE attempts SET state='FAILED', updated_at=?, error=? WHERE id=?",
                (now, error, attempt_id),
            )
            partition_state = (
                "FAILED"
                if max_attempts is not None and int(row["ordinal"]) >= max_attempts
                else "QUEUED"
            )
            connection.execute(
                """UPDATE partitions SET state=?, node_id=NULL,
                   current_attempt_id=NULL, updated_at=?, error=?
                   WHERE run_id=? AND id=? AND current_attempt_id=?""",
                (
                    partition_state,
                    now,
                    error,
                    row["run_id"],
                    row["partition_id"],
                    attempt_id,
                ),
            )
            connection.execute(
                "UPDATE nodes SET current_partition_id=NULL WHERE id=?",
                (row["node_id"],),
            )
            connection.execute(
                """INSERT INTO events(
                     created_at, run_id, partition_id, node_id, level, message,
                     details_json
                   ) VALUES (?, ?, ?, ?, 'ERROR', ?, ?)""",
                (
                    now,
                    row["run_id"],
                    row["partition_id"],
                    row["node_id"],
                    f"Attempt failed; partition is {partition_state}",
                    json.dumps({"attempt_id": attempt_id, "error": error}),
                ),
            )
            self._refresh_run_state(connection, row["run_id"], now)
        return partition_state

    def events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def record_event(
        self,
        *,
        level: str,
        message: str,
        run_id: str | None = None,
        partition_id: str | None = None,
        node_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO events(
                     created_at, run_id, partition_id, node_id, level, message,
                     details_json
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    utc_now(),
                    run_id,
                    partition_id,
                    node_id,
                    level,
                    message,
                    json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
                ),
            )

    def cancel_queued(self, run_id: str, partition_id: str | None = None) -> int:
        now = utc_now()
        query = (
            "UPDATE partitions SET state='CANCELLED', desired_state='CANCELLED', "
            "updated_at=?, error=NULL "
            "WHERE run_id=? AND state='QUEUED'"
        )
        parameters: list[Any] = [now, run_id]
        if partition_id:
            query += " AND id=?"
            parameters.append(partition_id)
        with self.connect() as connection:
            cursor = connection.execute(query, parameters)
            self._refresh_run_state(connection, run_id, now)
            return cursor.rowcount

    def request_cancel_run(self, run_id: str) -> None:
        with self.connect() as connection:
            now = utc_now()
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE runs SET desired_state='CANCELLED', updated_at=? WHERE id=?",
                (now, run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown run: {run_id}")
            connection.execute(
                """UPDATE partitions SET desired_state='CANCELLED', updated_at=?
                   WHERE run_id=? AND state NOT IN ('SUCCEEDED', 'FAILED', 'CANCELLED')""",
                (now, run_id),
            )

    def request_cancel_partition(self, run_id: str, partition_id: str) -> int:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state FROM partitions WHERE run_id=? AND id=?",
                (run_id, partition_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown partition: {run_id}/{partition_id}")
            if row["state"] in TERMINAL_PARTITION_STATES:
                return 0
            queued = row["state"] == "QUEUED"
            connection.execute(
                """UPDATE partitions SET desired_state='CANCELLED',
                   state=CASE WHEN state='QUEUED' THEN 'CANCELLED' ELSE state END,
                   updated_at=?, error=NULL WHERE run_id=? AND id=?""",
                (now, run_id, partition_id),
            )
            connection.execute(
                """INSERT INTO events(
                     created_at, run_id, partition_id, level, message
                   ) VALUES (?, ?, ?, 'INFO', 'Partition cancellation requested')""",
                (now, run_id, partition_id),
            )
            self._refresh_run_state(connection, run_id, now)
            return int(queued)

    def retry_failed_partition(self, run_id: str, partition_id: str) -> bool:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """UPDATE partitions SET state='QUEUED', desired_state='RUNNING',
                   node_id=NULL, error=NULL, updated_at=?
                   WHERE run_id=? AND id=? AND state='FAILED'
                   AND current_attempt_id IS NULL""",
                (now, run_id, partition_id),
            )
            self._refresh_run_state(connection, run_id, now)
            return cursor.rowcount == 1

    @staticmethod
    def _refresh_run_state(
        connection: sqlite3.Connection, run_id: str, now: str
    ) -> None:
        rows = connection.execute(
            "SELECT state, COUNT(*) AS count FROM partitions WHERE run_id=? GROUP BY state",
            (run_id,),
        ).fetchall()
        counts = {row["state"]: row["count"] for row in rows}
        total = sum(counts.values())
        if counts and counts.get("SUCCEEDED", 0) == total:
            state = "SUCCEEDED"
        elif any(
            counts.get(item, 0)
            for item in (
                "ASSIGNED",
                "STAGING",
                "READY",
                "STARTING",
                "RUNNING",
                "COLLECTING",
                "VERIFYING",
                "UNKNOWN",
            )
        ):
            state = "RUNNING"
        elif counts.get("QUEUED", 0):
            state = "PLANNED"
        elif (
            total
            and sum(counts.get(item, 0) for item in TERMINAL_PARTITION_STATES) == total
        ):
            state = "FAILED" if counts.get("FAILED", 0) else "CANCELLED"
        else:
            state = "PLANNED"
        connection.execute(
            "UPDATE runs SET state=?, updated_at=? WHERE id=?", (state, now, run_id)
        )

    def overview(self, run_id: str | None = None) -> dict[str, Any]:
        runs = self.list_runs()
        if run_id is None and runs:
            active = next(
                (run for run in runs if run["state"] not in ("SUCCEEDED", "CANCELLED")),
                runs[0],
            )
            run_id = active["id"]
        active_run = self.get_run(run_id) if run_id else None
        partitions = self.list_partitions(run_id) if run_id else []
        if active_run:
            completed_weight = 0.0
            for row in partitions:
                if row["state"] == "SUCCEEDED":
                    progress = 100.0
                else:
                    progress = min(100.0, max(0.0, float(row["progress_percent"])))
                completed_weight += float(row["weight"]) * progress / 100.0
            total_weight = sum(float(row["weight"]) for row in partitions)
            active_run["progress_percent"] = round(
                100.0 * completed_weight / max(total_weight, 1.0), 2
            )
        return {
            "generated_at": utc_now(),
            "controller": {"status": "online", "version": "0.1.0"},
            "active_run": active_run,
            "runs_summary": runs,
            "nodes": self.list_nodes(),
            "partitions": partitions,
            "events": self.events(50),
        }
