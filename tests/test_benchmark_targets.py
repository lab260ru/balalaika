"""Contract tests for the benchmarking harness config mutators.

The autotune/warmup sweep drives the harness with ``--batch-size-override``, so
each target's mutator MUST write the batch size into the EXACT config key its
stage reads. These tests apply every mutator to a deep copy of the real
``configs/config.yaml`` and assert the value lands at the hard-coded key path
that the corresponding stage module consumes.

The expected key paths below were verified against the stage source:

* separation.music_detect  -> separation.music_detect.bs        (music_detect.py)
* separation.distillmos    -> separation.distillmos.batch_size  (distillmos_process.py)
* separation.antispoofing  -> separation.antispoofing.batch_size(antispoofing.py)
* denoising                -> denoising.batch_size              (denoising.py)
* transcription            -> transcription.batch_size (FLAT)   (transcription.py)

Run: .dev_venv/bin/python -m pytest tests/test_benchmark_targets.py -q
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from benchmarking.common import REPO_ROOT, load_full_config
from benchmarking.targets import TARGETS

CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
BATCH_OVERRIDE = 17
WORK_DATASET = Path("/tmp/bench_targets_test/dataset")


def make_args(batch_size_override: Optional[int] = BATCH_OVERRIDE) -> argparse.Namespace:
    """Namespace exposing every attribute the mutators read."""
    return argparse.Namespace(
        batch_size_override=batch_size_override,
        cpu_workers_per_gpu=None,
        cpu_workers_total=None,
        model_name_override=None,
        disable_diarization=False,
    )


def dig(config: Dict[str, Any], key_path: Tuple[str, ...]) -> Any:
    current: Any = config
    for key in key_path:
        assert isinstance(current, dict), f"expected dict at {key} in {key_path}"
        assert key in current, f"missing key {key} in {key_path}"
        current = current[key]
    return current


# (target name, key path the stage actually reads for batch size)
BATCH_KEY_PATHS: List[Tuple[str, Tuple[str, ...]]] = [
    ("separation.music_detect", ("separation", "music_detect", "bs")),
    ("separation.distillmos", ("separation", "distillmos", "batch_size")),
    ("separation.antispoofing", ("separation", "antispoofing", "batch_size")),
    ("denoising.stage", ("denoising", "batch_size")),
    ("transcription.stage", ("transcription", "batch_size")),
    ("transcription.gigaam-v3-e2e-ctc", ("transcription", "batch_size")),
    ("transcription.giga_ctc", ("transcription", "batch_size")),
    ("transcription.vosk", ("transcription", "batch_size")),
    ("transcription.tone", ("transcription", "batch_size")),
]


@pytest.fixture(scope="module")
def base_config() -> Dict[str, Any]:
    return load_full_config(CONFIG_PATH)


@pytest.mark.parametrize("target_name,key_path", BATCH_KEY_PATHS)
def test_batch_size_override_lands_on_stage_key(
    base_config: Dict[str, Any], target_name: str, key_path: Tuple[str, ...]
) -> None:
    target = TARGETS[target_name]
    assert target.mutator is not None, f"{target_name} has no mutator"

    config = copy.deepcopy(base_config)
    target.mutator(config, make_args(), WORK_DATASET)

    assert dig(config, key_path) == BATCH_OVERRIDE, (
        f"{target_name}: batch override did not land at {'.'.join(key_path)}"
    )


def test_separation_stage_fans_batch_into_all_model_subsections(
    base_config: Dict[str, Any],
) -> None:
    """The aggregate separation stage must override every batched sub-stage."""
    target = TARGETS["separation.stage"]
    config = copy.deepcopy(base_config)
    target.mutator(config, make_args(), WORK_DATASET)

    assert dig(config, ("separation", "music_detect", "bs")) == BATCH_OVERRIDE
    assert dig(config, ("separation", "distillmos", "batch_size")) == BATCH_OVERRIDE
    assert dig(config, ("separation", "antispoofing", "batch_size")) == BATCH_OVERRIDE


def test_transcription_model_target_pins_single_model(base_config: Dict[str, Any]) -> None:
    target = TARGETS["transcription.giga_ctc"]
    config = copy.deepcopy(base_config)
    target.mutator(config, make_args(), WORK_DATASET)

    section = config["transcription"]
    assert section["model_names"] == ["giga_ctc"]
    assert section["consensus_num"] == 0
    assert section["use_rover"] is False
    # The flat batch_size is what transcription.py reads; no per-model subsection.
    assert section["batch_size"] == BATCH_OVERRIDE
    assert "giga" not in section, "stale per-model batch subsection leaked back in"
    assert "vosk" not in section


def test_cpu_workers_per_gpu_plumbs_num_workers(base_config: Dict[str, Any]) -> None:
    args = make_args()
    args.cpu_workers_per_gpu = 9

    checks = [
        ("separation.music_detect", ("separation", "music_detect", "num_workers")),
        ("separation.distillmos", ("separation", "distillmos", "num_workers")),
        ("separation.antispoofing", ("separation", "antispoofing", "num_workers")),
        ("denoising.stage", ("denoising", "num_workers")),
        ("transcription.stage", ("transcription", "num_workers")),
    ]
    for target_name, key_path in checks:
        config = copy.deepcopy(base_config)
        TARGETS[target_name].mutator(config, args, WORK_DATASET)
        assert dig(config, key_path) == 9, f"{target_name} did not plumb num_workers"


def test_all_pipeline_modules_are_importable_paths(base_config: Dict[str, Any]) -> None:
    """pipeline.base must reference modules that still exist on disk."""
    target = TARGETS["pipeline.base"]
    for module in target.modules:
        rel = Path(*module.split(".")).with_suffix(".py")
        pkg_init = REPO_ROOT / Path(*module.split(".")) / "__init__.py"
        assert (REPO_ROOT / rel).exists() or pkg_init.exists(), (
            f"pipeline.base references missing module {module}"
        )


def test_no_target_references_removed_separation_modules() -> None:
    """Guard against regressing to nisqa/diarization/silence_detect modules."""
    removed = {
        "src.separation.nisqa_process",
        "src.separation.diarization",
        "src.separation.silence_detect",
    }
    for name, target in TARGETS.items():
        leaked = removed.intersection(target.modules)
        assert not leaked, f"{name} still references removed modules: {leaked}"
