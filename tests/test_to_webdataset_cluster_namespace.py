"""Multinode shard names remain unique without changing local exports."""

import pytest

from src.to_webdataset import resolve_shard_namespace, shard_pattern, worker_fn


def test_unset_cluster_environment_keeps_legacy_shard_name(tmp_path):
    assert resolve_shard_namespace({}) is None
    assert shard_pattern(tmp_path, 0) == str(tmp_path / "shard_000_%04d.tar")


def test_cluster_namespace_adds_partition_and_global_rank_to_shard_name(tmp_path):
    namespace = resolve_shard_namespace(
        {
            "BALALAIKA_PARTITION_ID": "part-0007",
            "BALALAIKA_GLOBAL_RANK": "7",
        }
    )

    assert namespace == ("part-0007", 7)
    assert shard_pattern(tmp_path, 2, namespace) == str(
        tmp_path / "shard_g000007_part-0007_w002_%04d.tar"
    )


@pytest.mark.parametrize(
    "environment",
    [
        {"BALALAIKA_PARTITION_ID": "part-0001"},
        {"BALALAIKA_GLOBAL_RANK": "1"},
        {
            "BALALAIKA_PARTITION_ID": "../part-0001",
            "BALALAIKA_GLOBAL_RANK": "1",
        },
        {
            "BALALAIKA_PARTITION_ID": "part-0001",
            "BALALAIKA_GLOBAL_RANK": "-1",
        },
        {
            "BALALAIKA_PARTITION_ID": "part-0001",
            "BALALAIKA_GLOBAL_RANK": "01",
        },
        {
            "BALALAIKA_PARTITION_ID": "part-0001",
            "BALALAIKA_GLOBAL_RANK": str(2**31),
        },
    ],
)
def test_cluster_namespace_rejects_partial_or_unsafe_values(environment):
    with pytest.raises(ValueError):
        resolve_shard_namespace(environment)


def test_worker_writes_namespaced_shard_without_overwriting_other_rank(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFFfake-audio")
    output = tmp_path / "output"
    output.mkdir()

    first = worker_fn(
        worker_id=0,
        audio_paths=[str(audio)],
        output_dir=output,
        metadata_dict={},
        max_shard_size=1024 * 1024,
        max_shard_count=100,
        shard_namespace=("part-0000", 0),
    )
    second = worker_fn(
        worker_id=0,
        audio_paths=[str(audio)],
        output_dir=output,
        metadata_dict={},
        max_shard_size=1024 * 1024,
        max_shard_count=100,
        shard_namespace=("part-0001", 1),
    )

    assert first == (1, 0)
    assert second == (1, 0)
    assert sorted(path.name for path in output.glob("*.tar")) == [
        "shard_g000000_part-0000_w000_0000.tar",
        "shard_g000001_part-0001_w000_0000.tar",
    ]
