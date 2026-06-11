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
* ``BALALAIKA_IO_PROFILE``     — ``auto``/``hdd``/``ssd`` reader-concurrency profile
* ``BALALAIKA_THREADS_PER_WORKER`` — intra-op / OMP / BLAS thread cap per
  worker process (empty = unset, i.e. library defaults / no regression)
* ``BALALAIKA_STATE_FORMAT``   — ``csv``/``parquet`` pipeline-state format (from ``csv.state_format``)

The Python modules also read the same ``runtime`` block via :func:`runtime_cfg`
so the values stay aligned between shell and Python.
"""
from __future__ import annotations

import argparse
import shlex
import sys
from functools import lru_cache
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
    "io_profile": "auto",
    # Empty = unset: keep library defaults so single-worker latency does not
    # regress. A positive int caps ORT intra-op pools + OMP/BLAS teams per
    # worker process (see base.sh / make_session_options).
    "threads_per_worker": "",
    # Sourced from the top-level `csv:` block, not `runtime:` (see
    # _load_runtime). "csv" keeps balalaika.csv as the live state; "parquet"
    # uses balalaika.parquet + a CSV export. Exported so every spawned stage
    # worker sees the same format.
    "state_format": "csv",
}

ENV_KEYS = {
    "venv_path": "BALALAIKA_VENV",
    "cpu_affinity": "BALALAIKA_CPU_AFFINITY",
    "log_dir": "BALALAIKA_LOG_DIR",
    "log_level": "BALALAIKA_LOG_LEVEL",
    "trt_cache_path": "BALALAIKA_TRT_CACHE_PATH",
    "trt_workspace_bytes": "BALALAIKA_TRT_WORKSPACE",
    "trt_fp16": "BALALAIKA_TRT_FP16",
    "io_profile": "BALALAIKA_IO_PROFILE",
    "threads_per_worker": "BALALAIKA_THREADS_PER_WORKER",
    "state_format": "BALALAIKA_STATE_FORMAT",
}


def _load_runtime(config_path: str) -> Dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    block = data.get("runtime", {})
    cfg: Dict[str, Any] = dict(block) if isinstance(block, dict) else {}
    # state_format lives in the `csv:` block but is exported alongside the
    # runtime env so base.sh propagates it to every stage/worker.
    csv_block = data.get("csv", {})
    if isinstance(csv_block, dict) and csv_block.get("state_format") is not None:
        cfg.setdefault("state_format", csv_block.get("state_format"))
    return cfg


@lru_cache(maxsize=None)
def _runtime_cfg_cached(config_path: str | None) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    if config_path:
        for k, v in _load_runtime(config_path).items():
            if v is None:
                continue
            cfg[k] = v
    return cfg


def runtime_cfg(config_path: str | None = None) -> Dict[str, Any]:
    """Return a merged runtime config (defaults overridden by YAML).

    The config file is immutable during a run, so the parsed+merged block is
    memoised per resolved ``config_path``: ``get_onnx_providers`` (and the
    benchmarking loops) call this once per session build — many times per ASR
    stage — and would otherwise re-open + ``yaml.safe_load`` the whole config
    on every call. A fresh ``dict`` copy is returned so callers can mutate it
    freely without poisoning the cache.
    """
    return dict(_runtime_cfg_cached(config_path))


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
