from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from cluster_admin.config import ClusterConfig, validate_slug

MANIFEST_SCHEMA_VERSION = 1
STATE_FILENAME = "balalaika.parquet"
_PARQUET_BATCH_SIZE = 65_536


class UnsafeDatasetError(ValueError):
    """Raised when the source tree cannot be represented safely for transfer."""


@dataclass(frozen=True)
class PartitionPlan:
    """Controller-side artifacts ready to be persisted in ``StateDB``."""

    run: dict[str, Any]
    partitions: list[dict[str, Any]]
    run_manifest_path: Path


@dataclass(frozen=True)
class _FileEntry:
    path: str
    size: int
    mtime_ns: int


@dataclass
class _Unit:
    key: str
    files: list[_FileEntry] = field(default_factory=list)
    input_bytes: int = 0
    audio_bytes: int = 0
    audio_seconds: float = 0.0
    balance_weight: float = 0.0


@dataclass
class _Bin:
    ordinal: int
    units: list[_Unit] = field(default_factory=list)
    weight: float = 0.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> str:
    payload_text = (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    payload = payload_text.encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _validate_source_root(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise UnsafeDatasetError(f"Source root cannot be a symlink: {expanded}")
    try:
        mode = expanded.stat(follow_symlinks=False).st_mode
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Source root does not exist: {expanded}") from error
    if not stat.S_ISDIR(mode):
        raise NotADirectoryError(f"Source root is not a directory: {expanded}")
    return expanded.resolve(strict=True)


def _safe_relative_path(path: Any) -> PurePosixPath:
    if not isinstance(path, str) or not path:
        raise UnsafeDatasetError(f"Dataset path must be a non-empty string: {path!r}")
    if "\x00" in path or any(ord(character) < 32 for character in path):
        raise UnsafeDatasetError(f"Dataset path contains a control character: {path!r}")
    relative = PurePosixPath(path)
    if relative.is_absolute() or not relative.parts:
        raise UnsafeDatasetError(f"Dataset path must be relative: {path!r}")
    if any(part in ("", ".", "..") for part in relative.parts):
        raise UnsafeDatasetError(f"Unsafe dataset path: {path!r}")
    if str(relative) != path or path.startswith("-"):
        raise UnsafeDatasetError(f"Dataset path is not canonical: {path!r}")
    return relative


def _scan_files(
    source_root: Path, excluded_top_level: set[str] | None = None
) -> list[_FileEntry]:
    """Walk without following links and return stable, relative file metadata."""

    files: list[_FileEntry] = []

    def visit(directory: Path, relative_directory: PurePosixPath | None) -> None:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda item: os.fsencode(item.name))
        for entry in entries:
            if relative_directory is None and entry.name in (
                excluded_top_level or set()
            ):
                continue
            relative = (
                PurePosixPath(entry.name)
                if relative_directory is None
                else relative_directory / entry.name
            )
            relative = _safe_relative_path(relative.as_posix())
            if entry.is_symlink():
                raise UnsafeDatasetError(
                    f"Symlinks are not allowed in the dataset: {relative.as_posix()}"
                )
            try:
                metadata = entry.stat(follow_symlinks=False)
            except FileNotFoundError as error:
                raise UnsafeDatasetError(
                    f"Dataset changed while it was being scanned: {relative.as_posix()}"
                ) from error
            if stat.S_ISDIR(metadata.st_mode):
                visit(Path(entry.path), relative)
            elif stat.S_ISREG(metadata.st_mode):
                if relative_directory is None and entry.name == STATE_FILENAME:
                    continue
                files.append(
                    _FileEntry(
                        path=relative.as_posix(),
                        size=metadata.st_size,
                        mtime_ns=metadata.st_mtime_ns,
                    )
                )
            else:
                raise UnsafeDatasetError(
                    "Only regular files and directories are allowed in the dataset: "
                    f"{relative.as_posix()}"
                )

    visit(source_root, None)
    return files


def _group_key(relative_path: str, depth: int) -> str:
    relative = _safe_relative_path(relative_path)
    return PurePosixPath(*relative.parts[:depth]).as_posix()


def _make_units(
    files: Iterable[_FileEntry],
    group_depth: int,
    audio_extensions: tuple[str, ...],
) -> dict[str, _Unit]:
    units: dict[str, _Unit] = {}
    for item in files:
        key = _group_key(item.path, group_depth)
        unit = units.setdefault(key, _Unit(key=key))
        unit.files.append(item)
        unit.input_bytes += item.size
        if PurePosixPath(item.path).suffix.lower() in audio_extensions:
            unit.audio_bytes += item.size
    for unit in units.values():
        unit.files.sort(key=lambda item: os.fsencode(item.path))
    return units


def _import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "PyArrow is required to read or split balalaika.parquet"
        ) from error
    return pa, pq


def _state_relative_path(value: Any, source_root: Path) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = os.fsdecode(value)
    if not isinstance(value, str) or not value:
        return None
    if ".." in PurePosixPath(value).parts:
        raise UnsafeDatasetError(
            f"Parquet filepath contains parent traversal: {value!r}"
        )

    candidate = Path(value)
    if candidate.is_absolute():
        normalized = Path(os.path.normpath(value))
        try:
            relative = normalized.relative_to(source_root)
        except ValueError:
            try:
                # State created inside the worker container uses this mount.
                relative = normalized.relative_to("/data")
            except ValueError:
                return None
        text = relative.as_posix()
    else:
        posix = PurePosixPath(value)
        text = posix.as_posix()

    if text in ("", "."):
        return None
    return _safe_relative_path(text).as_posix()


def _positive_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


def _load_state_durations(
    state_path: Path,
    source_root: Path,
    group_depth: int,
    units: dict[str, _Unit],
    allowed_paths: set[str],
) -> dict[str, float]:
    """Stream only path/duration columns and attach known durations to units."""

    if not state_path.exists():
        return {}
    _, pq = _import_pyarrow()
    parquet = pq.ParquetFile(state_path)
    names = set(parquet.schema_arrow.names)
    if "filepath" not in names or "total_duration" not in names:
        return {}

    duration_by_path: dict[str, float] = {}
    for batch in parquet.iter_batches(
        batch_size=_PARQUET_BATCH_SIZE,
        columns=["filepath", "total_duration"],
    ):
        paths = batch.column(0).to_pylist()
        durations = batch.column(1).to_pylist()
        for raw_path, raw_duration in zip(paths, durations):
            relative = _state_relative_path(raw_path, source_root)
            duration = _positive_float(raw_duration)
            if relative is None or relative not in allowed_paths or not duration:
                continue
            key = _group_key(relative, group_depth)
            unit = units.get(key)
            if unit is None:
                continue
            unit.audio_seconds += duration
            duration_by_path[relative] = duration_by_path.get(relative, 0.0) + duration
    return duration_by_path


def _set_balance_weights(units: Iterable[_Unit]) -> None:
    units = list(units)
    known_seconds = sum(unit.audio_seconds for unit in units)
    known_bytes = sum(unit.audio_bytes for unit in units if unit.audio_seconds > 0)
    if known_seconds <= 0:
        for unit in units:
            unit.balance_weight = float(max(unit.input_bytes, 1))
        return

    bytes_per_second = known_bytes / known_seconds if known_bytes > 0 else 32_000.0
    bytes_per_second = max(bytes_per_second, 1.0)
    for unit in units:
        unit.balance_weight = unit.audio_seconds
        if unit.balance_weight <= 0:
            basis = unit.audio_bytes or unit.input_bytes
            unit.balance_weight = max(basis / bytes_per_second, 0.001)


def _assign_lpt(units: Iterable[_Unit], partition_count: int) -> list[_Bin]:
    """Longest-processing-time assignment with deterministic tie breaking."""

    ordered = sorted(units, key=lambda unit: (-unit.balance_weight, unit.key))
    count = min(partition_count, len(ordered))
    bins = [_Bin(ordinal=index) for index in range(count)]
    for unit in ordered:
        target = min(bins, key=lambda item: (item.weight, item.ordinal))
        target.units.append(unit)
        target.weight += unit.balance_weight
    for item in bins:
        item.units.sort(key=lambda unit: unit.key)
    return bins


def _replace_filepath_column(table, rewritten_paths: list[str], pa):
    index = table.schema.get_field_index("filepath")
    if index < 0:
        raise ValueError(f"{STATE_FILENAME} does not contain a filepath column")
    field = table.schema.field(index)
    try:
        values = pa.array(rewritten_paths, type=field.type)
    except (TypeError, ValueError, pa.ArrowInvalid) as error:
        raise ValueError(
            f"Unsupported filepath type in {STATE_FILENAME}: {field.type}"
        ) from error
    return table.set_column(index, field, values)


def _split_state(
    state_path: Path,
    output_dir: Path,
    source_root: Path,
    group_depth: int,
    group_to_partition: dict[str, int],
    partition_count: int,
    allowed_paths: set[str],
) -> list[dict[str, Any]]:
    pa, pq = _import_pyarrow()
    parquet = pq.ParquetFile(state_path)
    if "filepath" not in parquet.schema_arrow.names:
        raise ValueError(f"{STATE_FILENAME} does not contain a filepath column")

    partial_paths = [
        output_dir / f"part-{index:04d}.{STATE_FILENAME}.partial"
        for index in range(partition_count)
    ]
    final_paths = [
        output_dir / f"part-{index:04d}.{STATE_FILENAME}"
        for index in range(partition_count)
    ]
    writers = [
        pq.ParquetWriter(path, parquet.schema_arrow, compression="snappy")
        for path in partial_paths
    ]
    row_counts = [0] * partition_count
    try:
        for batch in parquet.iter_batches(batch_size=_PARQUET_BATCH_SIZE):
            table = pa.Table.from_batches([batch])
            path_index = table.schema.get_field_index("filepath")
            raw_paths = table.column(path_index).to_pylist()
            routed: list[list[int]] = [[] for _ in range(partition_count)]
            rewritten: list[list[str]] = [[] for _ in range(partition_count)]
            for row_index, raw_path in enumerate(raw_paths):
                relative = _state_relative_path(raw_path, source_root)
                if relative is None or relative not in allowed_paths:
                    continue
                partition = group_to_partition.get(_group_key(relative, group_depth))
                if partition is None:
                    continue
                routed[partition].append(row_index)
                rewritten[partition].append(f"/data/{relative}")
            for partition, indices in enumerate(routed):
                if not indices:
                    continue
                subset = table.take(pa.array(indices, type=pa.int64()))
                subset = _replace_filepath_column(subset, rewritten[partition], pa)
                writers[partition].write_table(subset)
                row_counts[partition] += len(indices)
    finally:
        for writer in writers:
            writer.close()

    result: list[dict[str, Any]] = []
    for index, (partial, final) in enumerate(zip(partial_paths, final_paths)):
        os.replace(partial, final)
        result.append(
            {
                "path": STATE_FILENAME,
                "rows": row_counts[index],
                "bytes": final.stat().st_size,
                "sha256": _sha256_file(final),
            }
        )
    return result


def _files_payload(
    units: Iterable[_Unit], duration_by_path: dict[str, float]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for unit in units:
        for item in unit.files:
            payload: dict[str, Any] = {
                "path": item.path,
                "size": item.size,
                "mtime_ns": item.mtime_ns,
            }
            duration = duration_by_path.get(item.path, 0.0)
            if duration:
                payload["duration_seconds"] = duration
            result.append(payload)
    result.sort(key=lambda item: os.fsencode(item["path"]))
    return result


def _write_files_list(path: Path, files: Iterable[dict[str, Any]]) -> None:
    with path.open("wb") as handle:
        for item in files:
            relative = _safe_relative_path(item["path"]).as_posix()
            handle.write(relative.encode("utf-8", "surrogateescape"))
            handle.write(b"\0")


def validate_source_manifest(
    source_root: str | Path, manifest_path: str | Path
) -> dict[str, int]:
    """Fence an rsync against source changes made after planning.

    This deliberately checks metadata instead of hashing file contents. Callers
    can run it immediately before and after rsync to detect mutations without a
    second full read of a multi-hundred-gigabyte dataset.
    """

    root = _validate_source_root(Path(source_root))
    manifest_file = Path(manifest_path)
    try:
        manifest_info = manifest_file.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Partition manifest is missing: {manifest_file}"
        ) from error
    if not stat.S_ISREG(manifest_info.st_mode) or manifest_file.is_symlink():
        raise UnsafeDatasetError(
            f"Partition manifest must be a regular file: {manifest_file}"
        )
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid partition manifest: {manifest_file}") from error
    if not isinstance(manifest, dict):
        raise ValueError("Partition manifest must contain an object")
    files = manifest.get("files")
    if not isinstance(files, list) or manifest.get("files_total") != len(files):
        raise ValueError("Partition manifest contains an invalid files list")

    seen: set[str] = set()
    verified_directories: set[tuple[str, ...]] = set()
    input_bytes = 0
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Partition manifest file entry must be an object")
        relative = _safe_relative_path(item.get("path"))
        relative_text = relative.as_posix()
        if relative_text in seen:
            raise ValueError(f"Duplicate path in partition manifest: {relative_text}")
        seen.add(relative_text)

        current = root
        directory_parts: list[str] = []
        for part in relative.parts[:-1]:
            directory_parts.append(part)
            directory_key = tuple(directory_parts)
            current /= part
            if directory_key in verified_directories:
                continue
            try:
                directory_info = current.lstat()
            except FileNotFoundError as error:
                raise UnsafeDatasetError(
                    f"Source directory disappeared after planning: {current}"
                ) from error
            if not stat.S_ISDIR(directory_info.st_mode) or current.is_symlink():
                raise UnsafeDatasetError(
                    f"Source directory changed or became a symlink: {current}"
                )
            verified_directories.add(directory_key)

        path = root.joinpath(*relative.parts)
        try:
            info = path.lstat()
        except FileNotFoundError as error:
            raise UnsafeDatasetError(
                f"Source file disappeared after planning: {relative_text}"
            ) from error
        if not stat.S_ISREG(info.st_mode) or path.is_symlink():
            raise UnsafeDatasetError(
                f"Source file changed or became a symlink: {relative_text}"
            )
        expected_size = item.get("size")
        expected_mtime_ns = item.get("mtime_ns")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int):
            raise ValueError(f"Invalid size in manifest for {relative_text}")
        invalid_mtime = isinstance(expected_mtime_ns, bool) or not isinstance(
            expected_mtime_ns, int
        )
        if invalid_mtime:
            raise ValueError(f"Invalid mtime_ns in manifest for {relative_text}")
        if info.st_size != expected_size or info.st_mtime_ns != expected_mtime_ns:
            raise UnsafeDatasetError(
                f"Source file changed after planning: {relative_text}"
            )
        input_bytes += info.st_size

    if manifest.get("input_bytes") != input_bytes:
        raise ValueError("Partition manifest input_bytes does not match its files")
    return {"files_total": len(files), "input_bytes": input_bytes}


def build_partition_plan(
    config: ClusterConfig,
    run_id: str,
    partition_count: int | None = None,
    split_state: bool = True,
) -> PartitionPlan:
    """Build immutable transfer manifests and an LPT-balanced run plan.

    Dataset files are grouped by the first ``config.group_depth`` relative path
    components. The output directory is published with one atomic rename, so a
    failed planning pass cannot leave a seemingly usable partial run.
    """

    validate_slug(run_id, "run id")
    source_root = _validate_source_root(config.source_root)
    state_dir = config.state_dir.expanduser().resolve()
    if state_dir == source_root or state_dir.is_relative_to(source_root):
        raise ValueError("Controller state_dir must be outside source_root")
    if partition_count is None:
        enabled_nodes = sum(node.enabled for node in config.nodes)
        if enabled_nodes == 0:
            raise ValueError("At least one enabled node is required")
        partition_count = enabled_nodes * config.partitions_per_node
    if isinstance(partition_count, bool) or not isinstance(partition_count, int):
        raise ValueError("partition_count must be a positive integer")
    if partition_count < 1:
        raise ValueError("partition_count must be a positive integer")

    files = _scan_files(source_root, set(config.exclude_top_level))
    if not files:
        raise ValueError(f"No dataset files found under {source_root}")
    all_units = _make_units(files, config.group_depth, config.audio_extensions)
    units = {key: unit for key, unit in all_units.items() if unit.audio_bytes > 0}
    if not units:
        raise ValueError(
            "No files with configured audio extensions were found under "
            f"{source_root}"
        )
    allowed_paths = {item.path for unit in units.values() for item in unit.files}
    state_path = source_root / STATE_FILENAME
    duration_by_path = _load_state_durations(
        state_path, source_root, config.group_depth, units, allowed_paths
    )
    _set_balance_weights(units.values())
    bins = _assign_lpt(units.values(), partition_count)

    manifests_root = config.manifests_dir
    manifests_root.mkdir(parents=True, exist_ok=True)
    final_dir = manifests_root / run_id
    if final_dir.exists():
        raise FileExistsError(f"Run artifacts already exist: {final_dir}")
    staging_dir = manifests_root / f".{run_id}.partial-{os.getpid()}-{uuid.uuid4().hex}"
    staging_dir.mkdir(mode=0o700)

    try:
        try:
            config_info = config.pipeline_config.lstat()
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Pipeline config does not exist: {config.pipeline_config}"
            ) from error
        if not stat.S_ISREG(config_info.st_mode) or config.pipeline_config.is_symlink():
            raise ValueError(
                f"Pipeline config must be a regular file: {config.pipeline_config}"
            )
        config_snapshot_name = "pipeline.config.yaml"
        config_snapshot = staging_dir / config_snapshot_name
        shutil.copyfile(config.pipeline_config, config_snapshot)
        config_sha256 = _sha256_file(config_snapshot)

        group_to_partition = {
            unit.key: item.ordinal for item in bins for unit in item.units
        }
        state_metadata: list[dict[str, Any] | None]
        if split_state and state_path.exists():
            state_metadata = _split_state(
                state_path,
                staging_dir,
                source_root,
                config.group_depth,
                group_to_partition,
                len(bins),
                allowed_paths,
            )
        else:
            state_metadata = [None] * len(bins)

        partition_records: list[dict[str, Any]] = []
        run_partitions: list[dict[str, Any]] = []
        for item in bins:
            partition_id = f"part-{item.ordinal:04d}"
            files_payload = _files_payload(item.units, duration_by_path)
            input_bytes = sum(unit.input_bytes for unit in item.units)
            audio_seconds = sum(unit.audio_seconds for unit in item.units)
            manifest: dict[str, Any] = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "run_id": run_id,
                "partition_id": partition_id,
                "ordinal": item.ordinal,
                "group_depth": config.group_depth,
                "groups": [unit.key for unit in item.units],
                "files": files_payload,
                "files_total": len(files_payload),
                "input_bytes": input_bytes,
                "audio_seconds": audio_seconds,
                "weight": item.weight,
                "state_fragment": (
                    STATE_FILENAME if state_metadata[item.ordinal] is not None else None
                ),
                "state_fragment_metadata": state_metadata[item.ordinal],
            }
            manifest_name = f"{partition_id}.manifest.json"
            manifest_path = staging_dir / manifest_name
            manifest_sha256 = _write_json(manifest_path, manifest)
            files_list_name = f"{partition_id}.files0"
            _write_files_list(staging_dir / files_list_name, files_payload)

            state_fragment_name = (
                f"{partition_id}.{STATE_FILENAME}"
                if state_metadata[item.ordinal] is not None
                else None
            )
            partition_records.append(
                {
                    "id": partition_id,
                    "ordinal": item.ordinal,
                    "manifest": manifest_name,
                    "manifest_sha256": manifest_sha256,
                    "files_list": files_list_name,
                    "state_fragment": state_fragment_name,
                    "files_total": len(files_payload),
                    "input_bytes": input_bytes,
                    "audio_seconds": audio_seconds,
                    "weight": item.weight,
                }
            )
            run_partitions.append(
                {
                    "id": partition_id,
                    "ordinal": item.ordinal,
                    "manifest_path": str(final_dir / manifest_name),
                    "manifest_sha256": manifest_sha256,
                    "files_list_path": str(final_dir / files_list_name),
                    "state_fragment_path": (
                        str(final_dir / state_fragment_name)
                        if state_fragment_name
                        else None
                    ),
                    "files_total": len(files_payload),
                    "input_bytes": input_bytes,
                    "audio_seconds": audio_seconds,
                    "weight": item.weight,
                    "units_total": len(item.units),
                }
            )

        run_manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "source_root": str(source_root),
            "config_path": config_snapshot_name,
            "config_sha256": config_sha256,
            "image": config.image,
            "stage_start": config.stage_start,
            "stage_stop": config.stage_stop,
            "group_depth": config.group_depth,
            "partitions": partition_records,
            "partitions_total": len(partition_records),
            "files_total": sum(item["files_total"] for item in partition_records),
            "input_bytes": sum(item["input_bytes"] for item in partition_records),
            "audio_seconds": sum(item["audio_seconds"] for item in partition_records),
        }
        _write_json(staging_dir / "run.manifest.json", run_manifest)
        os.replace(staging_dir, final_dir)

        run = {
            "id": run_id,
            "source_root": str(source_root),
            "image": config.image,
            "config_path": str(final_dir / config_snapshot_name),
            "config_sha256": config_sha256,
            "stage_start": config.stage_start,
            "stage_stop": config.stage_stop,
            "input_bytes": run_manifest["input_bytes"],
            "audio_seconds": run_manifest["audio_seconds"],
            "files_total": run_manifest["files_total"],
            "partitions_total": run_manifest["partitions_total"],
        }
        return PartitionPlan(
            run=run,
            partitions=run_partitions,
            run_manifest_path=final_dir / "run.manifest.json",
        )
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
