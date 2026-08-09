from __future__ import annotations

import csv
import hashlib
import json
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cluster_admin.config import ClusterConfig, NodeConfig
from cluster_admin.cli import build_parser
from cluster_admin.db import StateDB
from cluster_admin.finalizer import FinalizerError, finalize_run


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _config(tmp_path: Path) -> ClusterConfig:
    source = tmp_path / "source"
    source.mkdir()
    pipeline_config = tmp_path / "pipeline.yaml"
    pipeline_config.write_text("export: {}\n", encoding="utf-8")
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("", encoding="ascii")
    return ClusterConfig(
        path=tmp_path / "cluster.yaml",
        state_dir=tmp_path / "state",
        source_root=source,
        pipeline_config=pipeline_config,
        image="balalaika:test",
        ssh_identity=None,
        known_hosts=known_hosts,
        connect_timeout=5,
        poll_seconds=1,
        max_attempts=2,
        partitions_per_node=1,
        group_depth=1,
        audio_extensions=(".wav",),
        stage_start="1",
        stage_stop="14",
        shm_size="1g",
        nodes=(
            NodeConfig(
                id="node-a",
                host="node-a",
                user="worker",
                port=22,
                work_root="/work",
                models_root="/models",
                cache_root="/cache",
            ),
        ),
    )


def _write_audit(path: Path, *, files: int, hours: float, threshold: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "timestamp",
                "stage",
                "files_in",
                "files_out",
                "hours_in",
                "hours_out",
                "hours_removed",
                "params",
            ),
        )
        writer.writeheader()
        # Only the latest row for a repeated stage contributes to the dataset sum.
        writer.writerow(
            {
                "timestamp": "2026-08-08T00:00:00+00:00",
                "stage": "music_detect",
                "files_in": files + 10,
                "files_out": files + 9,
                "hours_in": hours + 1,
                "hours_out": hours + 0.9,
                "hours_removed": 0.1,
                "params": json.dumps({"threshold": threshold}),
            }
        )
        writer.writerow(
            {
                "timestamp": "2026-08-09T00:00:00+00:00",
                "stage": "music_detect",
                "files_in": files,
                "files_out": files - 1,
                "hours_in": hours,
                "hours_out": hours - 0.25,
                "hours_removed": 0.25,
                "params": json.dumps({"threshold": threshold}),
            }
        )


def _write_tar(path: Path, payload: bytes) -> None:
    member = path.with_suffix(".txt")
    member.write_bytes(payload)
    with tarfile.open(path, "w") as archive:
        archive.add(member, arcname="sample.txt")
    member.unlink()


def _successful_run(
    tmp_path: Path,
    *,
    bad_filepath: bool = False,
    complete_second: bool = True,
) -> tuple[ClusterConfig, StateDB, list[dict]]:
    config = _config(tmp_path)
    database = StateDB(config.db_path)
    database.initialize()
    database.sync_nodes(config.nodes)
    partitions = [
        {
            "id": f"part-{ordinal:04d}",
            "ordinal": ordinal,
            "manifest_path": str(tmp_path / f"part-{ordinal}.json"),
            "manifest_sha256": f"manifest-{ordinal}",
            "files_list_path": str(tmp_path / f"part-{ordinal}.files0"),
            "files_total": 1,
            "input_bytes": 10,
            "audio_seconds": 1.0,
            "weight": 1.0,
        }
        for ordinal in range(2)
    ]
    database.create_run(
        {
            "id": "run-final",
            "source_root": str(config.source_root),
            "image": config.image,
            "config_path": str(config.pipeline_config),
            "config_sha256": "config-digest",
            "execution_config_sha256": "execution-digest",
            "execution_config_version": 2,
            "execution_config_json": "{}",
            "stage_start": "1",
            "stage_stop": "14",
            "input_bytes": 20,
            "audio_seconds": 2.0,
        },
        partitions,
    )

    attempts = []
    for ordinal, partition in enumerate(partitions):
        attempt = database.claim_next(
            "run-final",
            "node-a",
            "/work",
            node_host="node-a",
            node_user="worker",
            node_port=22,
            gpu_uuids=["GPU-test"],
        )
        assert attempt is not None
        attempt = database.get_attempt(attempt["attempt_id"])
        assert attempt is not None
        root = config.results_dir / "run-final" / "partitions" / partition["id"]
        data = root / "data"
        output = root / "output" / "nested"
        control = root / "control"
        data.mkdir(parents=True)
        output.mkdir(parents=True)
        control.mkdir(parents=True)
        relative = Path("podcast") / f"chunk-{ordinal}.wav"
        audio = data / relative
        audio.parent.mkdir(parents=True)
        audio.write_bytes(f"audio-{ordinal}".encode("ascii"))
        filepath = (
            "/outside/chunk.wav"
            if bad_filepath and ordinal == 0
            else (
                (
                    f"/work/runs/run-final/partitions/{partition['id']}/"
                    f"attempt-001-{attempt['id'][:12]}/data/{relative.as_posix()}"
                )
                if ordinal == 0
                else f"/data/{relative.as_posix()}"
            )
        )
        pq.write_table(
            pa.table(
                {
                    "filepath": [filepath],
                    "total_duration": [float(ordinal + 1)],
                }
            ),
            data / "balalaika.parquet",
        )
        _write_audit(
            data / "filter_summary.csv",
            files=10 + ordinal,
            hours=2.0 + ordinal,
            threshold=0.5 + ordinal,
        )
        _write_tar(
            output
            / f"shard_g{ordinal:06d}_{partition['id']}_w000_0000.tar",
            f"sample-{ordinal}".encode(),
        )
        (control / "job.json").write_text(
            json.dumps({"state": "COMPLETED"}),
            encoding="utf-8",
        )
        result = {
            "attempt_id": attempt["id"],
            "fencing_token": attempt["fencing_token"],
            "manifest_sha256": partition["manifest_sha256"],
            "config_sha256": "config-digest",
            "exit_code": 0,
        }
        (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
        (root / "_SUCCESS").write_text(
            attempt["fencing_token"] + "\n", encoding="ascii"
        )
        attempts.append(attempt)
        if ordinal == 1 and not complete_second:
            database.update_attempt(attempt["id"], "UNKNOWN")
        else:
            database.update_attempt(attempt["id"], "UNKNOWN")
            database.update_attempt(attempt["id"], "COMPLETED", exit_code=0)
    return config, database, attempts


def test_finalize_merges_state_audit_and_webdataset_without_repacking(tmp_path):
    config, database, _ = _successful_run(tmp_path)

    result = finalize_run(config, database, "run-final")

    dataset = Path(result["path"])
    assert result["reused"] is False
    assert (dataset / "_SUCCESS").read_text("ascii").strip() == result[
        "manifest_sha256"
    ]
    state = pq.read_table(dataset / "balalaika.parquet").to_pydict()
    assert state["filepath"] == [
        "partitions/part-0000/data/podcast/chunk-0.wav",
        "partitions/part-0001/data/podcast/chunk-1.wav",
    ]

    with (dataset / "filter_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        audit = list(csv.DictReader(handle))
    assert len(audit) == 1
    assert audit[0]["stage"] == "music_detect"
    assert audit[0]["files_in"] == "21"
    assert audit[0]["files_out"] == "19"
    assert audit[0]["hours_in"] == "5.0"
    assert audit[0]["hours_out"] == "4.5"
    assert json.loads(audit[0]["params"]) == {
        "partitions": {
            "part-0000": {"threshold": 0.5},
            "part-0001": {"threshold": 1.5},
        }
    }

    shard_entries = [
        json.loads(line)
        for line in (dataset / "webdataset" / "shards.jsonl")
        .read_text("utf-8")
        .splitlines()
    ]
    assert [entry["name"] for entry in shard_entries] == [
        "part-0000-r000000-s000000.tar",
        "part-0001-r000001-s000000.tar",
    ]
    for entry in shard_entries:
        published = dataset.parent / entry["path"]
        source = dataset.parent / entry["source_path"]
        assert published.stat().st_ino == source.stat().st_ino

    events = database.events()
    assert events[0]["message"] == "Finalized run dataset"


def test_finalize_is_idempotent_and_never_replaces_dataset(tmp_path):
    config, database, _ = _successful_run(tmp_path)
    first = finalize_run(config, database, "run-final")
    manifest = Path(first["path"]) / "manifest.json"
    inode = manifest.stat().st_ino

    second = finalize_run(config, database, "run-final")

    assert second["reused"] is True
    assert second["manifest_sha256"] == first["manifest_sha256"]
    assert manifest.stat().st_ino == inode


def test_finalize_refuses_incomplete_run(tmp_path):
    config, database, _ = _successful_run(tmp_path, complete_second=False)

    with pytest.raises(FinalizerError, match="expected SUCCEEDED"):
        finalize_run(config, database, "run-final")

    assert not (config.results_dir / "run-final" / "dataset").exists()


def test_finalize_refuses_nonportable_parquet_filepath_and_cleans_partial(tmp_path):
    config, database, _ = _successful_run(tmp_path, bad_filepath=True)

    with pytest.raises(FinalizerError, match="outside the partition data root"):
        finalize_run(config, database, "run-final")

    run_root = config.results_dir / "run-final"
    assert not (run_root / "dataset").exists()
    assert not list(run_root.glob(".dataset.*.partial"))


def test_finalize_detects_tampered_immutable_manifest(tmp_path):
    config, database, _ = _successful_run(tmp_path)
    result = finalize_run(config, database, "run-final")
    manifest = Path(result["path"]) / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FinalizerError, match="digest does not match"):
        finalize_run(config, database, "run-final")


def test_finalize_allows_run_before_export_or_audit_stages(tmp_path):
    config, database, _ = _successful_run(tmp_path)
    with database.connect() as connection:
        connection.execute(
            "UPDATE runs SET stage_stop='13' WHERE id='run-final'"
        )
    run_root = config.results_dir / "run-final"
    for path in run_root.glob("partitions/*/data/filter_summary.csv"):
        path.unlink()
    for path in run_root.glob("partitions/*/output/**/*.tar"):
        path.unlink()

    result = finalize_run(config, database, "run-final")

    assert result["filter_summary"] is None
    assert result["webdataset"]["shards"] == 0
    assert (Path(result["path"]) / "webdataset" / "shards.jsonl").read_bytes() == b""


def test_cli_exposes_run_finalize():
    args = build_parser().parse_args(["run", "finalize", "run-final", "--json"])

    assert args.command == "run"
    assert args.run_command == "finalize"
    assert args.run_id == "run-final"
    assert args.json is True
