"""GPU / ONNX-Runtime helpers shared across pipeline stages.

Three small helpers:

* :func:`apply_torch_perf_defaults` — the same TF32 / Flash / Mem-efficient SDP
  toggles every stage opted into. Replaces the ~5-line ``torch.backends.cuda``
  block that used to live at the top of every module.
* :func:`get_onnx_providers` — single source of truth for the onnxruntime
  provider tuple list. With ``use_tensorrt=True`` it returns a TensorRT-first
  list using the engine cache root from the runtime config block.
* :func:`gpu_count` — small wrapper around ``torch.cuda.device_count`` that
  doesn't crash on CPU-only hosts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.utils.runtime_env import runtime_cfg


def apply_torch_perf_defaults(*, disable_math_sdp: bool = True) -> None:
    """Enable TF32 + Flash/Mem-efficient SDP; optionally disable math SDP."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    if disable_math_sdp:
        try:
            torch.backends.cuda.enable_math_sdp(False)
        except (RuntimeError, AttributeError):
            pass


def gpu_count() -> int:
    """Return ``torch.cuda.device_count()`` (or 0 when torch is unavailable)."""
    try:
        import torch
    except ImportError:
        return 0
    return int(torch.cuda.device_count())


def get_onnx_providers(
    cuda_id: int,
    *,
    use_tensorrt: bool = False,
    config_path: str | os.PathLike | None = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Return an onnxruntime provider tuple list for a single GPU.

    With ``use_tensorrt=False`` (default) returns just the CUDA provider.
    With ``use_tensorrt=True`` a ``TensorrtExecutionProvider`` entry comes
    first, sharing the engine cache root from the runtime config block (one
    cache directory per CUDA device id).
    """
    if not use_tensorrt:
        return [("CUDAExecutionProvider", {"device_id": cuda_id})]

    rt = runtime_cfg(config_path)
    cache_path = Path(str(rt["trt_cache_path"])) / f"trt_cache_{cuda_id}"
    cache_path.mkdir(parents=True, exist_ok=True)

    return [
        (
            "TensorrtExecutionProvider",
            {
                "device_id": cuda_id,
                "trt_max_workspace_size": int(rt["trt_workspace_bytes"]),
                "trt_fp16_enable": bool(rt["trt_fp16"]),
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache_path),
            },
        ),
        ("CUDAExecutionProvider", {"device_id": cuda_id}),
    ]
