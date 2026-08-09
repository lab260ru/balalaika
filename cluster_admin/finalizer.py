"""Publish a completed multinode run as one immutable dataset artifact."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .config import ClusterConfig, validate_slug
from .db import StateDB

FINALIZER_SCHEMA_VERSION = 1
FINAL_DIRNAME = "dataset"
MANIFEST_FILENAME = "manifest.json"
SUCCESS_FILENAME = "_SUCCESS"
STATE_FILENAME = "balalaika.parquet"
AUDIT_FILENAME = "filter_summary.csv"
SHARD_MANIFEST_FILENAME = "shards.jsonl"
PARQUET_BATCH_SIZE = 100_000

AUDIT_HEADERS = (
    "timestamp",
    "stage",
    "files_in",
    "files_out",
    "hours_in",
    "hours_out",
    "hours_removed",
    "params",
)


class FinalizerError(RuntimeError):
    """A run cannot be finalized without risking a corrupt dataset."""


@dataclass(frozen=True)
class _PartitionResult:
    partition: dict[str, Any]
    attempt: dict[str, Any]
    root: Path
    data_root: Path
    remote_data_roots: tuple[PurePosixPath, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise FinalizerError(f"Missing {label}: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise FinalizerError(f"{label} must be a regular file: {path}")
    return metadata


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json_file(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalizerError(f"Invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalizerError(f"{label} must contain a JSON object: {path}")
    return value


def _partition_results(
    config: ClusterConfig,
    database: StateDB,
    run_id: str,
) -> tuple[dict[str, Any], list[_PartitionResult]]:
    run = database.get_run(run_id)
    if run is None:
        raise FinalizerError(f"Unknown run: {run_id}")
    if run.get("state") != "SUCCEEDED":
        raise FinalizerError(
            f"Run {run_id} is {run.get('state')}, expected SUCCEEDED"
        )

    partitions = database.list_partitions(run_id)
    expected_total = int(run.get("partitions_total") or 0)
    if expected_total <= 0 or len(partitions) != expected_total:
        raise FinalizerError(
            f"Run {run_id} has {len(partitions)} partitions, expected {expected_total}"
        )
    partitions.sort(key=lambda item: (int(item["ordinal"]), item["id"]))
    ordinals = [int(item["ordinal"]) for item in partitions]
    if ordinals != list(range(expected_total)):
        raise FinalizerError(f"Run {run_id} partition ordinals are not contiguous")

    run_root = config.results_dir / run_id
    results: list[_PartitionResult] = []
    for partition in partitions:
        partition_id = validate_slug(str(partition["id"]), "partition id")
        if partition.get("state") != "SUCCEEDED":
            raise FinalizerError(
                f"Partition {run_id}/{partition_id} is {partition.get('state')}, "
                "expected SUCCEEDED"
            )
        attempt_id = partition.get("current_attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise FinalizerError(
                f"Partition {run_id}/{partition_id} has no current attempt"
            )
        attempt = database.get_attempt(attempt_id)
        if attempt is None:
            raise FinalizerError(f"Missing current attempt {attempt_id}")
        if (
            attempt.get("state") != "COMPLETED"
            or attempt.get("run_id") != run_id
            or attempt.get("partition_id") != partition_id
            or attempt.get("id") != attempt_id
        ):
            raise FinalizerError(
                f"Current attempt {attempt_id} is not the completed owner of "
                f"{run_id}/{partition_id}"
            )

        root = run_root / "partitions" / partition_id
        result = _read_json_file(root / "result.json", "partition result")
        _require_regular_file(root / SUCCESS_FILENAME, "partition success marker")
        try:
            marker = (root / SUCCESS_FILENAME).read_text("ascii").strip()
        except (OSError, UnicodeDecodeError) as exc:
            raise FinalizerError(
                f"Invalid partition success marker: {root / SUCCESS_FILENAME}"
            ) from exc
        checks = {
            "attempt_id": attempt_id,
            "fencing_token": attempt["fencing_token"],
            "manifest_sha256": partition["manifest_sha256"],
            "config_sha256": run["config_sha256"],
            "exit_code": 0,
        }
        for key, expected in checks.items():
            if result.get(key) != expected:
                raise FinalizerError(
                    f"Partition {run_id}/{partition_id} has invalid result {key}"
                )
        if marker != attempt["fencing_token"]:
            raise FinalizerError(
                f"Partition {run_id}/{partition_id} has a stale success marker"
            )
        result_digest = attempt.get("result_sha256")
        if result_digest and result_digest != _sha256_file(root / "result.json"):
            raise FinalizerError(
                f"Partition {run_id}/{partition_id} result digest does not match"
            )

        attempt_name = (
            f"attempt-{int(attempt['ordinal']):03d}-{attempt['id'][:12]}"
        )
        remote_attempt_root = (
            PurePosixPath(attempt["remote_root"])
            / "runs"
            / run_id
            / "partitions"
            / partition_id
            / attempt_name
        )
        remote_roots: list[PurePosixPath] = [
            PurePosixPath("/data"),
            remote_attempt_root / "data",
        ]
        job_path = root / "control" / "job.json"
        if job_path.exists():
            job = _read_json_file(job_path, "partition job metadata")
            attempt_root = job.get("attempt_root")
            if isinstance(attempt_root, str) and attempt_root.startswith("/"):
                remote_roots.append(PurePosixPath(attempt_root) / "data")
        results.append(
            _PartitionResult(
                partition=partition,
                attempt=attempt,
                root=root,
                data_root=root / "data",
                remote_data_roots=tuple(remote_roots),
            )
        )
    return run, results


def _safe_relative_filepath(
    raw_value: object,
    remote_data_roots: Iterable[PurePosixPath],
) -> PurePosixPath:
    if not isinstance(raw_value, str) or not raw_value:
        raise FinalizerError("Parquet filepath must be a non-empty string")
    if "\x00" in raw_value:
        raise FinalizerError("Parquet filepath contains a NUL byte")
    value = raw_value
    path = PurePosixPath(value)
    if path.is_absolute():
        relative = None
        for root in remote_data_roots:
            try:
                relative = path.relative_to(root)
                break
            except ValueError:
                continue
        if relative is None:
            raise FinalizerError(
                f"Parquet filepath is outside the partition data root: {value!r}"
            )
    else:
        relative = path
    if not relative.parts or relative == PurePosixPath("."):
        raise FinalizerError(f"Invalid empty parquet filepath: {value!r}")
    if any(part in ("", ".", "..") for part in relative.parts):
        raise FinalizerError(f"Unsafe parquet filepath: {value!r}")
    return relative


def _merge_parquet(
    partitions: list[_PartitionResult],
    destination: Path,
) -> tuple[int, str]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise FinalizerError("PyArrow is required to finalize balalaika.parquet") from exc

    writer = None
    output_schema = None
    rows_total = 0
    try:
        for item in partitions:
            source = item.data_root / STATE_FILENAME
            _require_regular_file(source, "partition balalaika.parquet")
            try:
                parquet = pq.ParquetFile(source)
            except Exception as exc:
                raise FinalizerError(f"Cannot read partition parquet {source}: {exc}") from exc
            schema = parquet.schema_arrow
            path_index = schema.get_field_index("filepath")
            if path_index < 0:
                raise FinalizerError(f"Partition parquet has no filepath column: {source}")
            path_type = schema.field(path_index).type
            if not (pa.types.is_string(path_type) or pa.types.is_large_string(path_type)):
                raise FinalizerError(
                    f"Partition parquet filepath has unsupported type {path_type}: {source}"
                )
            if output_schema is None:
                output_schema = schema
                writer = pq.ParquetWriter(
                    destination,
                    output_schema,
                    compression="snappy",
                )
            elif not schema.equals(output_schema, check_metadata=False):
                raise FinalizerError(
                    f"Partition parquet schema differs from the first partition: {source}"
                )

            for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_SIZE):
                table = pa.Table.from_batches([batch])
                portable_paths: list[str] = []
                for raw_path in table.column(path_index).to_pylist():
                    relative = _safe_relative_filepath(
                        raw_path,
                        item.remote_data_roots,
                    )
                    portable_paths.append(
                        (
                            PurePosixPath("partitions")
                            / item.partition["id"]
                            / "data"
                            / relative
                        ).as_posix()
                    )
                field = table.schema.field(path_index)
                table = table.set_column(
                    path_index,
                    field,
                    pa.array(portable_paths, type=field.type),
                )
                writer.write_table(table)
                rows_total += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if output_schema is None:
        raise FinalizerError("Run has no partition parquet files")
    _fsync_file(destination)
    return rows_total, _sha256_file(destination)


def _float(value: str, label: str, source: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FinalizerError(f"Invalid {label} in {source}: {value!r}") from exc
    if not math.isfinite(result) or result < 0:
        raise FinalizerError(f"Invalid {label} in {source}: {value!r}")
    return result


def _integer(value: str, label: str, source: Path) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise FinalizerError(f"Invalid {label} in {source}: {value!r}") from exc
    if result < 0:
        raise FinalizerError(f"Invalid {label} in {source}: {value!r}")
    return result


def _read_latest_audit_rows(source: Path) -> dict[str, dict[str, str]]:
    _require_regular_file(source, "partition filter summary")
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(AUDIT_HEADERS):
                raise FinalizerError(
                    f"Unexpected filter summary columns in {source}: {reader.fieldnames}"
                )
            latest: dict[str, dict[str, str]] = {}
            for row in reader:
                stage = (row.get("stage") or "").strip()
                if not stage:
                    raise FinalizerError(f"Empty stage in filter summary {source}")
                _integer(row["files_in"], "files_in", source)
                _integer(row["files_out"], "files_out", source)
                _float(row["hours_in"], "hours_in", source)
                _float(row["hours_out"], "hours_out", source)
                _float(row["hours_removed"], "hours_removed", source)
                try:
                    parsed_params = json.loads(row["params"] or "{}")
                except json.JSONDecodeError as exc:
                    raise FinalizerError(f"Invalid params JSON in {source}") from exc
                row["params"] = json.dumps(
                    parsed_params,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                latest[stage] = row
            return latest
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise FinalizerError(f"Cannot read filter summary {source}: {exc}") from exc


def _merge_filter_summaries(
    partitions: list[_PartitionResult],
    destination: Path,
) -> tuple[int, str] | None:
    grouped: dict[str, list[tuple[str, dict[str, str]]]] = {}
    stage_order: list[str] = []
    for item in partitions:
        source = item.data_root / AUDIT_FILENAME
        if not source.exists() and not source.is_symlink():
            continue
        for stage, row in _read_latest_audit_rows(source).items():
            if stage not in grouped:
                grouped[stage] = []
                stage_order.append(stage)
            grouped[stage].append((item.partition["id"], row))
    if not grouped:
        return None

    with destination.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_HEADERS)
        writer.writeheader()
        for stage in stage_order:
            values = grouped[stage]
            params_by_partition = {
                partition_id: json.loads(row["params"])
                for partition_id, row in values
            }
            unique_params = {
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                for value in params_by_partition.values()
            }
            params: object
            if len(unique_params) == 1:
                params = next(iter(params_by_partition.values()))
            else:
                params = {"partitions": params_by_partition}
            writer.writerow(
                {
                    "timestamp": max(row["timestamp"] for _, row in values),
                    "stage": stage,
                    "files_in": sum(
                        _integer(row["files_in"], "files_in", destination)
                        for _, row in values
                    ),
                    "files_out": sum(
                        _integer(row["files_out"], "files_out", destination)
                        for _, row in values
                    ),
                    "hours_in": round(
                        sum(
                            _float(row["hours_in"], "hours_in", destination)
                            for _, row in values
                        ),
                        4,
                    ),
                    "hours_out": round(
                        sum(
                            _float(row["hours_out"], "hours_out", destination)
                            for _, row in values
                        ),
                        4,
                    ),
                    "hours_removed": round(
                        sum(
                            _float(
                                row["hours_removed"], "hours_removed", destination
                            )
                            for _, row in values
                        ),
                        4,
                    ),
                    "params": json.dumps(
                        params,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    return len(grouped), _sha256_file(destination)


def _publish_webdataset_shards(
    partitions: list[_PartitionResult],
    destination: Path,
) -> tuple[list[dict[str, Any]], str]:
    destination.mkdir()
    entries: list[dict[str, Any]] = []
    for item in partitions:
        output_root = item.root / "output"
        if not output_root.exists() and not output_root.is_symlink():
            continue
        if not output_root.is_dir() or output_root.is_symlink():
            raise FinalizerError(f"Partition output must be a directory: {output_root}")
        sources = sorted(
            output_root.rglob("*.tar"),
            key=lambda path: os.fsencode(path.relative_to(output_root).as_posix()),
        )
        for local_index, source in enumerate(sources):
            metadata = _require_regular_file(source, "WebDataset shard")
            if metadata.st_size <= 0:
                raise FinalizerError(f"WebDataset shard is empty: {source}")
            global_rank = int(item.partition["ordinal"])
            name = (
                f"{item.partition['id']}-r{global_rank:06d}-"
                f"s{local_index:06d}.tar"
            )
            target = destination / name
            try:
                os.link(source, target, follow_symlinks=False)
            except OSError as exc:
                raise FinalizerError(
                    f"Cannot hard-link WebDataset shard {source} to {target}; "
                    "the finalizer never copies or repacks audio"
                ) from exc
            linked = _require_regular_file(target, "published WebDataset shard")
            if (
                linked.st_dev != metadata.st_dev
                or linked.st_ino != metadata.st_ino
                or linked.st_size != metadata.st_size
                or linked.st_mtime_ns != metadata.st_mtime_ns
            ):
                raise FinalizerError(
                    f"WebDataset shard changed while it was published: {source}"
                )
            entries.append(
                {
                    "name": name,
                    "path": (
                        PurePosixPath(FINAL_DIRNAME) / "webdataset" / name
                    ).as_posix(),
                    "partition_id": item.partition["id"],
                    "global_rank": global_rank,
                    "local_shard_index": local_index,
                    "source_path": (
                        PurePosixPath("partitions")
                        / item.partition["id"]
                        / "output"
                        / source.relative_to(output_root).as_posix()
                    ).as_posix(),
                    "bytes": metadata.st_size,
                }
            )

    manifest_path = destination / SHARD_MANIFEST_FILENAME
    payload = b"".join(
        json.dumps(entry, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for entry in entries
    )
    _write_bytes(manifest_path, payload)
    _fsync_directory(destination)
    return entries, _sha256_bytes(payload)


def _verify_existing(final_root: Path, run_id: str) -> dict[str, Any]:
    try:
        root_metadata = final_root.lstat()
    except FileNotFoundError as exc:  # pragma: no cover - caller checks exists()
        raise FinalizerError(f"Missing finalized dataset: {final_root}") from exc
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise FinalizerError(f"Finalized dataset must be a directory: {final_root}")
    manifest_path = final_root / MANIFEST_FILENAME
    success_path = final_root / SUCCESS_FILENAME
    manifest = _read_json_file(manifest_path, "dataset manifest")
    _require_regular_file(success_path, "dataset success marker")
    try:
        marker = success_path.read_text("ascii").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise FinalizerError(f"Invalid dataset success marker: {success_path}") from exc
    digest = _sha256_file(manifest_path)
    if marker != digest:
        raise FinalizerError(f"Dataset manifest digest does not match {success_path}")
    if (
        manifest.get("schema_version") != FINALIZER_SCHEMA_VERSION
        or manifest.get("run_id") != run_id
    ):
        raise FinalizerError(f"Dataset manifest identity is invalid: {manifest_path}")

    parquet = manifest.get("parquet")
    webdataset = manifest.get("webdataset")
    if not isinstance(parquet, dict) or not isinstance(webdataset, dict):
        raise FinalizerError(f"Dataset manifest has invalid artifacts: {manifest_path}")
    parquet_path = final_root / STATE_FILENAME
    shard_manifest_path = final_root / "webdataset" / SHARD_MANIFEST_FILENAME
    _require_regular_file(parquet_path, "finalized parquet")
    _require_regular_file(shard_manifest_path, "WebDataset shard manifest")
    if parquet.get("sha256") != _sha256_file(parquet_path):
        raise FinalizerError(f"Finalized parquet digest does not match: {parquet_path}")
    if webdataset.get("manifest_sha256") != _sha256_file(shard_manifest_path):
        raise FinalizerError(
            f"WebDataset shard manifest digest does not match: {shard_manifest_path}"
        )
    try:
        entries = [
            json.loads(line)
            for line in shard_manifest_path.read_text("utf-8").splitlines()
            if line
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalizerError(f"Invalid WebDataset shard manifest: {exc}") from exc
    if len(entries) != webdataset.get("shards"):
        raise FinalizerError("WebDataset shard count does not match dataset manifest")
    names: set[str] = set()
    bytes_total = 0
    run_root = final_root.parent
    for entry in entries:
        if not isinstance(entry, dict):
            raise FinalizerError("WebDataset shard manifest entry must be an object")
        name = entry.get("name")
        path = entry.get("path")
        size = entry.get("bytes")
        if (
            not isinstance(name, str)
            or name in names
            or not isinstance(path, str)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise FinalizerError("WebDataset shard manifest entry is invalid")
        names.add(name)
        expected_path = (
            PurePosixPath(FINAL_DIRNAME) / "webdataset" / name
        ).as_posix()
        if path != expected_path:
            raise FinalizerError(f"Unexpected WebDataset shard path: {path!r}")
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise FinalizerError(f"Unsafe WebDataset shard path: {path!r}")
        shard = run_root.joinpath(*relative.parts)
        metadata = _require_regular_file(shard, "finalized WebDataset shard")
        if metadata.st_size != size:
            raise FinalizerError(f"WebDataset shard size does not match: {shard}")
        bytes_total += size
    if bytes_total != webdataset.get("bytes"):
        raise FinalizerError("WebDataset byte count does not match dataset manifest")
    webdataset_root = final_root / "webdataset"
    actual_paths = {
        path.relative_to(webdataset_root).as_posix()
        for path in webdataset_root.rglob("*.tar")
    }
    if actual_paths != names:
        raise FinalizerError("WebDataset directory does not match its shard manifest")

    audit = manifest.get("filter_summary")
    if audit is not None:
        if not isinstance(audit, dict):
            raise FinalizerError("Dataset filter summary metadata is invalid")
        audit_path = final_root / AUDIT_FILENAME
        _require_regular_file(audit_path, "finalized filter summary")
        if audit.get("sha256") != _sha256_file(audit_path):
            raise FinalizerError(
                f"Finalized filter summary digest does not match: {audit_path}"
            )
    manifest["manifest_sha256"] = digest
    manifest["path"] = str(final_root)
    manifest["reused"] = True
    return manifest


def finalize_run(
    config: ClusterConfig,
    database: StateDB,
    run_id: str,
) -> dict[str, Any]:
    """Finalize a successful run without mutating partition result trees.

    Publication is an atomic directory rename. Re-running the command verifies
    and returns the already-published artifact instead of replacing it.
    """

    run_id = validate_slug(run_id, "run id")
    database.initialize()
    run_root = config.results_dir / run_id
    final_root = run_root / FINAL_DIRNAME
    if final_root.exists() or final_root.is_symlink():
        return _verify_existing(final_root, run_id)

    run, partitions = _partition_results(config, database, run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    partial_root = run_root / f".{FINAL_DIRNAME}.{uuid.uuid4().hex}.partial"
    partial_root.mkdir(mode=0o755)
    published = False
    try:
        rows_total, parquet_sha256 = _merge_parquet(
            partitions,
            partial_root / STATE_FILENAME,
        )
        audit = _merge_filter_summaries(
            partitions,
            partial_root / AUDIT_FILENAME,
        )
        shards, shard_manifest_sha256 = _publish_webdataset_shards(
            partitions,
            partial_root / "webdataset",
        )
        partition_manifest = []
        for item in partitions:
            partition_manifest.append(
                {
                    "id": item.partition["id"],
                    "ordinal": int(item.partition["ordinal"]),
                    "attempt_id": item.attempt["id"],
                    "fencing_token": item.attempt["fencing_token"],
                    "manifest_sha256": item.partition["manifest_sha256"],
                    "result_path": (
                        PurePosixPath("partitions")
                        / item.partition["id"]
                        / "result.json"
                    ).as_posix(),
                }
            )
        manifest = {
            "schema_version": FINALIZER_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": _utc_now(),
            "config_sha256": run["config_sha256"],
            "execution_config_sha256": run.get("execution_config_sha256"),
            "stage_start": run["stage_start"],
            "stage_stop": run["stage_stop"],
            "input_bytes": int(run["input_bytes"]),
            "input_audio_seconds": float(run["audio_seconds"]),
            "path_base": "run_root",
            "parquet": {
                "path": f"{FINAL_DIRNAME}/{STATE_FILENAME}",
                "rows": rows_total,
                "sha256": parquet_sha256,
                "filepath_base": "run_root",
            },
            "webdataset": {
                "format": "webdataset",
                "root": f"{FINAL_DIRNAME}/webdataset",
                "manifest": (
                    f"{FINAL_DIRNAME}/webdataset/{SHARD_MANIFEST_FILENAME}"
                ),
                "manifest_sha256": shard_manifest_sha256,
                "shards": len(shards),
                "bytes": sum(int(entry["bytes"]) for entry in shards),
                "naming": "<partition-id>-r<global-rank>-s<local-shard>.tar",
            },
            "filter_summary": (
                {
                    "path": f"{FINAL_DIRNAME}/{AUDIT_FILENAME}",
                    "stages": audit[0],
                    "sha256": audit[1],
                }
                if audit is not None
                else None
            ),
            "partitions": partition_manifest,
        }
        manifest_bytes = _json_bytes(manifest)
        manifest_digest = _sha256_bytes(manifest_bytes)
        _write_bytes(partial_root / MANIFEST_FILENAME, manifest_bytes)
        _write_bytes(
            partial_root / SUCCESS_FILENAME,
            (manifest_digest + "\n").encode("ascii"),
        )
        _fsync_directory(partial_root)
        try:
            os.rename(partial_root, final_root)
            published = True
            _fsync_directory(run_root)
        except FileExistsError:
            return _verify_existing(final_root, run_id)

        database.record_event(
            run_id=run_id,
            level="INFO",
            message="Finalized run dataset",
            details={
                "path": str(final_root),
                "manifest_sha256": manifest_digest,
                "rows": rows_total,
                "webdataset_shards": len(shards),
            },
        )
        return {
            **manifest,
            "manifest_sha256": manifest_digest,
            "path": str(final_root),
            "reused": False,
        }
    finally:
        if not published and partial_root.exists():
            shutil.rmtree(partial_root)
