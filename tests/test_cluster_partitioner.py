from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cluster_admin.config import ClusterConfig, NodeConfig
from cluster_admin.partitioner import (
    UnsafeDatasetError,
    build_partition_plan,
    validate_source_manifest,
)


def _config(tmp_path: Path, source_root: Path, *, group_depth: int = 1):
    pipeline_config = tmp_path / "pipeline.yaml"
    pipeline_config.write_text("preprocess: {}\n", encoding="utf-8")
    return ClusterConfig(
        path=tmp_path / "cluster.yaml",
        state_dir=tmp_path / "controller-state",
        source_root=source_root,
        pipeline_config=pipeline_config,
        image="balalaika@sha256:" + "1" * 64,
        ssh_identity=None,
        known_hosts=tmp_path / "known_hosts",
        connect_timeout=10,
        poll_seconds=15,
        max_attempts=3,
        partitions_per_node=2,
        group_depth=group_depth,
        audio_extensions=(".flac", ".wav"),
        stage_start="1",
        stage_stop="15",
        shm_size="8g",
        nodes=(
            NodeConfig(
                id="node-1",
                host="node-1.example",
                user="balalaika",
                port=22,
                work_root="/var/lib/balalaika",
                models_root="/var/lib/balalaika/models",
                cache_root="/var/lib/balalaika/cache",
            ),
        ),
    )


def _write(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes([size % 251]) * size)


def _partition_manifests(plan):
    return [
        json.loads(Path(partition["manifest_path"]).read_text(encoding="utf-8"))
        for partition in plan.partitions
    ]


def _assignment(plan):
    return {
        group: manifest["partition_id"]
        for manifest in _partition_manifests(plan)
        for group in manifest["groups"]
    }


def test_lpt_plan_is_deterministic_and_writes_safe_transfer_artifacts(tmp_path):
    source = tmp_path / "dataset"
    _write(source / "group-a" / "audio.flac", 100)
    _write(source / "group-a" / "metadata.json", 3)
    _write(source / "group-b" / "audio.flac", 90)
    _write(source / "group-c" / "audio.flac", 80)
    _write(source / "group-d" / "audio.flac", 70)
    config = _config(tmp_path, source)

    first = build_partition_plan(config, "run-a", partition_count=2)
    second = build_partition_plan(config, "run-b", partition_count=2)

    assert _assignment(first) == _assignment(second)
    assert len(first.partitions) == 2
    assert first.run["input_bytes"] == 343
    assert sum(item["files_total"] for item in first.partitions) == 5

    source_text = str(source.resolve())
    for partition, manifest in zip(first.partitions, _partition_manifests(first)):
        assert "source_root" not in manifest
        assert source_text not in json.dumps(manifest)
        assert manifest["schema_version"] == 1
        assert manifest["files_total"] == len(manifest["files"])
        assert all(not Path(item["path"]).is_absolute() for item in manifest["files"])
        assert all(".." not in Path(item["path"]).parts for item in manifest["files"])

        manifest_bytes = Path(partition["manifest_path"]).read_bytes()
        assert (
            hashlib.sha256(manifest_bytes).hexdigest() == partition["manifest_sha256"]
        )
        files0 = Path(partition["files_list_path"]).read_bytes()
        assert files0.endswith(b"\0")
        assert [part.decode() for part in files0[:-1].split(b"\0")] == [
            item["path"] for item in manifest["files"]
        ]


def test_parquet_durations_drive_lpt_instead_of_file_size(tmp_path):
    source = tmp_path / "dataset"
    _write(source / "large-short" / "audio.flac", 1_000)
    _write(source / "small-long" / "audio.flac", 10)
    _write(source / "small-medium" / "audio.flac", 10)
    table = pa.table(
        {
            "filepath": [
                str(source / "large-short" / "audio.flac"),
                str(source / "small-long" / "audio.flac"),
                str(source / "small-medium" / "audio.flac"),
            ],
            "total_duration": [1.0, 9.0, 8.0],
        }
    )
    pq.write_table(table, source / "balalaika.parquet")

    plan = build_partition_plan(
        _config(tmp_path, source),
        "duration-run",
        partition_count=2,
        split_state=False,
    )
    manifests = _partition_manifests(plan)

    long_partition = next(
        manifest for manifest in manifests if "small-long" in manifest["groups"]
    )
    assert long_partition["groups"] == ["small-long"]
    assert sorted(manifest["audio_seconds"] for manifest in manifests) == [9.0, 9.0]
    assert plan.run["audio_seconds"] == 18.0
    duration_entries = {
        entry["path"]: entry["duration_seconds"]
        for manifest in manifests
        for entry in manifest["files"]
    }
    assert duration_entries["small-long/audio.flac"] == 9.0


def test_split_state_preserves_schema_and_rewrites_paths(tmp_path):
    source = tmp_path / "dataset"
    _write(source / "group-a" / "audio.wav", 20)
    _write(source / "group-b" / "audio.wav", 20)
    original = pa.table(
        {
            "filepath": [
                str(source / "group-a" / "audio.wav"),
                "/data/group-b/audio.wav",
                str(tmp_path / "outside.wav"),
            ],
            "total_duration": [4.0, 6.0, 99.0],
            "speaker_id": ["a", "b", "outside"],
            "score": [0.5, None, 1.0],
        }
    )
    pq.write_table(original, source / "balalaika.parquet")

    plan = build_partition_plan(
        _config(tmp_path, source), "split-run", partition_count=2, split_state=True
    )

    fragments = []
    for partition, manifest in zip(plan.partitions, _partition_manifests(plan)):
        fragment_path = Path(partition["state_fragment_path"])
        fragment = pq.read_table(fragment_path)
        fragments.append(fragment)
        assert manifest["state_fragment"] == "balalaika.parquet"
        metadata = manifest["state_fragment_metadata"]
        assert metadata["path"] == "balalaika.parquet"
        assert metadata["rows"] == fragment.num_rows
        assert metadata["bytes"] == fragment_path.stat().st_size
        assert (
            metadata["sha256"] == hashlib.sha256(fragment_path.read_bytes()).hexdigest()
        )
        assert "balalaika.parquet" not in {item["path"] for item in manifest["files"]}

    combined = pa.concat_tables(fragments)
    assert combined.schema == original.schema
    assert sorted(combined.column("filepath").to_pylist()) == [
        "/data/group-a/audio.wav",
        "/data/group-b/audio.wav",
    ]
    assert sorted(combined.column("speaker_id").to_pylist()) == ["a", "b"]


@pytest.mark.parametrize("link_to_directory", [False, True])
def test_rejects_symlinks_without_publishing_partial_run(tmp_path, link_to_directory):
    source = tmp_path / "dataset"
    _write(source / "group" / "audio.wav", 10)
    target = source / "target"
    if link_to_directory:
        target.mkdir()
    else:
        target.write_text("target", encoding="utf-8")
    (source / "unsafe-link").symlink_to(target, target_is_directory=link_to_directory)
    config = _config(tmp_path, source)

    with pytest.raises(UnsafeDatasetError, match="Symlinks"):
        build_partition_plan(config, "unsafe-run", partition_count=1)

    assert not (config.manifests_dir / "unsafe-run").exists()


def test_rejects_special_files(tmp_path):
    source = tmp_path / "dataset"
    _write(source / "group" / "audio.wav", 10)
    fifo = source / "group" / "input.pipe"
    os.mkfifo(fifo)

    with pytest.raises(UnsafeDatasetError, match="regular files"):
        build_partition_plan(_config(tmp_path, source), "fifo-run", partition_count=1)


@pytest.mark.parametrize("unsafe_name", ["-option.wav", "line\nbreak.wav"])
def test_rejects_paths_not_accepted_by_node_runner(tmp_path, unsafe_name):
    source = tmp_path / "dataset"
    _write(source / unsafe_name, 10)

    with pytest.raises(UnsafeDatasetError):
        build_partition_plan(
            _config(tmp_path, source), "unsafe-path-run", partition_count=1
        )


@pytest.mark.parametrize(
    "unsafe_filepath", ["../outside.wav", "/data/group/../../outside.wav"]
)
def test_rejects_parent_traversal_in_parquet_state(tmp_path, unsafe_filepath):
    source = tmp_path / "dataset"
    _write(source / "group" / "audio.wav", 10)
    pq.write_table(
        pa.table(
            {
                "filepath": [unsafe_filepath],
                "total_duration": [1.0],
            }
        ),
        source / "balalaika.parquet",
    )

    with pytest.raises(UnsafeDatasetError, match="parent traversal"):
        build_partition_plan(
            _config(tmp_path, source), "traversal-run", partition_count=1
        )


def test_partition_count_is_capped_to_number_of_groups(tmp_path):
    source = tmp_path / "dataset"
    _write(source / "only-group" / "audio.wav", 10)

    plan = build_partition_plan(
        _config(tmp_path, source), "small-run", partition_count=20
    )

    assert len(plan.partitions) == 1
    assert plan.partitions[0]["units_total"] == 1


def test_group_depth_keeps_nested_dataset_unit_together(tmp_path):
    source = tmp_path / "dataset"
    _write(source / "20260808" / "video-a" / "audio.wav", 10)
    _write(source / "20260808" / "video-a" / "metadata.json", 5)
    _write(source / "20260808" / "video-b" / "audio.wav", 10)

    plan = build_partition_plan(
        _config(tmp_path, source, group_depth=2),
        "nested-run",
        partition_count=2,
    )
    manifests = _partition_manifests(plan)
    paths_by_group = {
        group: {item["path"] for item in manifest["files"]}
        for manifest in manifests
        for group in manifest["groups"]
    }

    assert paths_by_group["20260808/video-a"] == {
        "20260808/video-a/audio.wav",
        "20260808/video-a/metadata.json",
    }
    assert paths_by_group["20260808/video-b"] == {"20260808/video-b/audio.wav"}


def test_rejects_controller_state_inside_dataset(tmp_path):
    source = tmp_path / "dataset"
    _write(source / "group" / "audio.wav", 10)
    config = replace(_config(tmp_path, source), state_dir=source / ".cluster-state")

    with pytest.raises(ValueError, match="outside source_root"):
        build_partition_plan(config, "recursive-state-run", partition_count=1)


def test_source_manifest_fence_detects_same_size_change(tmp_path):
    source = tmp_path / "dataset"
    audio = source / "group" / "audio.wav"
    _write(audio, 10)
    plan = build_partition_plan(
        _config(tmp_path, source), "fence-run", partition_count=1
    )
    manifest_path = Path(plan.partitions[0]["manifest_path"])

    assert validate_source_manifest(source, manifest_path) == {
        "files_total": 1,
        "input_bytes": 10,
    }
    planned_mtime = audio.stat().st_mtime_ns
    audio.write_bytes(b"z" * 10)
    os.utime(audio, ns=(planned_mtime + 1_000_000, planned_mtime + 1_000_000))

    with pytest.raises(UnsafeDatasetError, match="changed after planning"):
        validate_source_manifest(source, manifest_path)


def test_source_manifest_fence_rejects_symlink_replacement(tmp_path):
    source = tmp_path / "dataset"
    audio = source / "group" / "audio.wav"
    replacement = tmp_path / "replacement.wav"
    _write(audio, 10)
    _write(replacement, 10)
    plan = build_partition_plan(
        _config(tmp_path, source), "fence-link-run", partition_count=1
    )
    audio.unlink()
    audio.symlink_to(replacement)

    with pytest.raises(UnsafeDatasetError, match="became a symlink"):
        validate_source_manifest(source, plan.partitions[0]["manifest_path"])


@pytest.mark.parametrize("partition_count", [0, -1, True, 1.5])
def test_rejects_invalid_partition_count(tmp_path, partition_count):
    source = tmp_path / "dataset"
    _write(source / "group" / "audio.wav", 10)

    with pytest.raises(ValueError, match="positive integer"):
        build_partition_plan(
            _config(tmp_path, source),
            "invalid-count",
            partition_count=partition_count,
        )
