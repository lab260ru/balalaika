"""GPU / ONNX-Runtime helpers shared across pipeline stages.

* :func:`apply_torch_perf_defaults` — the same TF32 / Flash / Mem-efficient SDP
  toggles every stage opted into. Replaces the ~5-line ``torch.backends.cuda``
  block that used to live at the top of every module.
* :func:`get_onnx_providers` — single source of truth for the onnxruntime
  provider tuple list. With ``use_tensorrt=True`` it returns a TensorRT-first
  list using the engine cache root from the runtime config block (the cache
  dir mkdir is memoised per process).
* :func:`make_session_options` / :func:`apply_ort_thread_caps` — build / patch
  an ORT ``SessionOptions`` with optional intra-op thread caps, gated on
  ``runtime.threads_per_worker`` (no-op by default).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.utils.runtime_env import runtime_cfg


@lru_cache(maxsize=None)
def _ensure_trt_cache_dir(cache_path: str) -> str:
    """``mkdir -p`` the TRT engine cache dir once per (path) per process.

    ``get_onnx_providers`` is called once per session build — many times per
    ASR stage / per benchmark loop — and used to re-``mkdir`` the same
    directory on every call. The directory is immutable during a run, so the
    syscall is memoised here.
    """
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    return cache_path


def apply_torch_perf_defaults(*, disable_math_sdp: bool = True) -> None:
    """Enable TF32 + Flash/Mem-efficient SDP; optionally disable math SDP."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

def configured_threads_per_worker(
    config_path: str | os.PathLike | None = None,
) -> int | None:
    """Return ``runtime.threads_per_worker`` as a positive int, or ``None``.

    ``None`` (the default, empty config value) means "leave library defaults
    alone" so single-worker latency is not regressed.
    """
    raw = runtime_cfg(config_path).get("threads_per_worker", "")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def apply_ort_thread_caps(
    sess_options: "Any",
    *,
    config_path: str | os.PathLike | None = None,
    for_gpu_ep: bool = True,
) -> "Any":
    """Apply intra/inter-op thread caps + no-spin to an ORT ``SessionOptions``.

    Gated on ``runtime.threads_per_worker``: when unset this is a no-op, so the
    session keeps ORT's default physical-core intra-op pool (no behavior or
    latency change). When set, it caps intra-op to that value, inter-op to 1,
    and — for GPU-EP sessions, where the CPU pool mostly idles between Run()
    calls — disables intra-op busy-spinning so the threads don't burn cycles
    contending with the co-resident training job. Returns the same object.
    """
    threads = configured_threads_per_worker(config_path)
    if threads is None:
        return sess_options
    sess_options.intra_op_num_threads = threads
    sess_options.inter_op_num_threads = 1
    if for_gpu_ep:
        try:
            sess_options.add_session_config_entry(
                "session.intra_op.allow_spinning", "0"
            )
        except Exception:
            pass
    return sess_options


def make_session_options(
    *,
    config_path: str | os.PathLike | None = None,
    for_gpu_ep: bool = True,
):
    """Build a fresh ORT ``SessionOptions`` with graph opt + thread caps.

    Thread caps are only applied when ``runtime.threads_per_worker`` is set
    (see :func:`apply_ort_thread_caps`); otherwise this is equivalent to a
    plain ``SessionOptions()`` with ``ORT_ENABLE_ALL`` graph optimization.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return apply_ort_thread_caps(
        opts, config_path=config_path, for_gpu_ep=for_gpu_ep
    )


def onnx_first_input_name(model_path: os.PathLike | str) -> str:
    """First real graph input name from ONNX metadata (no InferenceSession)."""
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    initializers = {init.name for init in model.graph.initializer}
    for graph_input in model.graph.input:
        if graph_input.name not in initializers:
            return graph_input.name
    raise ValueError(f"No graph inputs found in {model_path}")


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
    _ensure_trt_cache_dir(str(cache_path))

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
