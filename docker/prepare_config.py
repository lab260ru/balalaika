#!/usr/bin/env python3
"""Render a host config into a container-local, per-run config copy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

DATA_SECTIONS = (
    "download",
    "preprocess",
    "separation",
    "transcription",
    "punctuation",
    "accent",
    "phonemizer",
    "denoising",
    "export",
)

MODEL_PATHS = (
    ("preprocess", "sortformer_model"),
    ("preprocess", "vad_args", "smart_vad_model"),
    ("separation", "music_detect", "onnx_path"),
    ("separation", "antispoofing", "onnx_path"),
    ("separation", "tts_suitability", "onnx_path"),
    ("transcription", "model_path"),
    ("transcription", "vosk_path"),
    ("denoising", "onnx_path"),
)


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        value = {}
        parent[key] = value
    return value


def _get_nested(root: dict[str, Any], keys: Iterable[str]) -> Any:
    current: Any = root
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _set_nested(root: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    current = root
    for key in keys[:-1]:
        current = _mapping(current, key)
    current[keys[-1]] = value


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value


def render_config(source: Path, destination: Path) -> None:
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level config value must be a mapping")

    runtime = _mapping(data, "runtime")
    runtime["venv_path"] = os.environ.get("VIRTUAL_ENV", "/opt/balalaika/.venv")
    runtime["log_dir"] = _env("BALALAIKA_LOG_DIR", "/logs")
    runtime["trt_cache_path"] = _env("BALALAIKA_TRT_CACHE_PATH", "/cache/trt")

    optional_runtime_env = {
        "io_profile": "BALALAIKA_IO_PROFILE",
        "threads_per_worker": "BALALAIKA_THREADS_PER_WORKER",
    }
    for config_key, env_name in optional_runtime_env.items():
        value = _env(env_name)
        if value is not None:
            runtime[config_key] = value

    data_root = _env("BALALAIKA_DATA_ROOT")
    if data_root is not None:
        for section_name in DATA_SECTIONS:
            section = data.get(section_name)
            if isinstance(section, dict):
                section["podcasts_path"] = data_root

    output_root = _env("BALALAIKA_OUTPUT_ROOT")
    if output_root is not None:
        export = _mapping(data, "export")
        export["output_path"] = output_root

    models_root = _env("BALALAIKA_MODELS_ROOT", "/models")
    if models_root is not None:
        for keys in MODEL_PATHS:
            configured = _get_nested(data, keys)
            if isinstance(configured, str) and configured.strip():
                _set_nested(data, keys, str(Path(models_root) / Path(configured).name))

    cache_root = _env("BALALAIKA_CACHE_ROOT", "/cache/balalaika")
    data["cache_path"] = cache_root
    phonemizer = data.get("phonemizer")
    if (
        cache_root is not None
        and isinstance(phonemizer, dict)
        and phonemizer.get("oov_cache_path")
    ):
        phonemizer["oov_cache_path"] = str(Path(cache_root) / "g2p_oov_cache.pkl")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_config(args.input, args.output)
    print(f"Rendered container config: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
