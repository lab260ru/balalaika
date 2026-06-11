"""Print shell-friendly env exports derived from the ``runtime`` config block.

Used by ``base.sh`` to avoid hard-coding the venv path, CPU affinity, log
directory, and TensorRT cache location in the shell scripts. Bash sources the
output via::

    eval "$(python3 -m src.utils.runtime_env --config_path configs/config.yaml)"

Output keys (printed only when present / non-empty):

* ``BALALAIKA_VENV``           — virtualenv root (defaults to ``.dev_venv``)
* ``BALALAIKA_CPU_AFFINITY``   — argument for ``taskset -c`` (empty disables)
* ``BALALAIKA_LOG_DIR``        — directory for rotating log files
* ``BALALAIKA_LOG_LEVEL``      — minimum log level for loguru sinks
* ``BALALAIKA_TRT_CACHE_PATH`` — TensorRT engine cache root
* ``BALALAIKA_TRT_WORKSPACE``  — TensorRT workspace bytes (per session)
* ``BALALAIKA_TRT_FP16``       — ``1`` / ``0`` toggle for fp16

The Python modules also read the same ``runtime`` block via :func:`runtime_cfg`
so the values stay aligned between shell and Python.
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULTS: Dict[str, Any] = {
    "venv_path": ".dev_venv",
    "cpu_affinity": "",
    "log_dir": "./logs",
    "log_level": "INFO",
    "trt_cache_path": "./cache/trt",
    "trt_workspace_bytes": 4 * 1024 ** 3,
    "trt_fp16": True,
}

ENV_KEYS = {
    "venv_path": "BALALAIKA_VENV",
    "cpu_affinity": "BALALAIKA_CPU_AFFINITY",
    "log_dir": "BALALAIKA_LOG_DIR",
    "log_level": "BALALAIKA_LOG_LEVEL",
    "trt_cache_path": "BALALAIKA_TRT_CACHE_PATH",
    "trt_workspace_bytes": "BALALAIKA_TRT_WORKSPACE",
    "trt_fp16": "BALALAIKA_TRT_FP16",
}


def _load_runtime(config_path: str) -> Dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    block = data.get("runtime", {}) if isinstance(data, dict) else {}
    return block if isinstance(block, dict) else {}


def runtime_cfg(config_path: str | None = None) -> Dict[str, Any]:
    """Return a merged runtime config (defaults overridden by YAML)."""
    cfg = dict(DEFAULTS)
    if config_path:
        for k, v in _load_runtime(config_path).items():
            if v is None:
                continue
            cfg[k] = v
    return cfg


def _format_value(key: str, value: Any) -> str:
    if key == "trt_fp16":
        return "1" if value else "0"
    return str(value if value is not None else "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True, help="Path to balalaika YAML config")
    args = parser.parse_args()

    cfg = runtime_cfg(args.config_path)
    for key, env_name in ENV_KEYS.items():
        value = _format_value(key, cfg.get(key))
        print(f"export {env_name}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
