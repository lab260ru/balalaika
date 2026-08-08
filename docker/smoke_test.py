#!/usr/bin/env python3
"""Verify the pinned container runtime on its single visible GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _check_ffmpeg() -> None:
    subprocess.run(
        ["ffmpeg", "-version"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    import torchaudio

    with tempfile.TemporaryDirectory(prefix="balalaika-smoke-") as tmp_dir:
        audio_path = Path(tmp_dir) / "tone.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=0.1",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(audio_path),
            ],
            check=True,
        )
        waveform, sample_rate = torchaudio.load_with_torchcodec(str(audio_path))
        if waveform.numel() == 0 or sample_rate != 16000:
            raise RuntimeError(
                "FFmpeg/torchaudio decode smoke test returned invalid audio"
            )
        encoded_path = Path(tmp_dir) / "roundtrip.wav"
        torchaudio.save_with_torchcodec(str(encoded_path), waveform, sample_rate)
        roundtrip, roundtrip_rate = torchaudio.load_with_torchcodec(str(encoded_path))
        if roundtrip.numel() == 0 or roundtrip_rate != sample_rate:
            raise RuntimeError(
                "TorchCodec encode/decode smoke test returned invalid audio"
            )


def _make_add_model(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    lhs = helper.make_tensor_value_info("lhs", TensorProto.FLOAT, [2, 2])
    rhs = helper.make_tensor_value_info("rhs", TensorProto.FLOAT, [2, 2])
    result = helper.make_tensor_value_info("result", TensorProto.FLOAT, [2, 2])
    graph = helper.make_graph(
        [helper.make_node("Add", ["lhs", "rhs"], ["result"])],
        "balalaika-smoke-add",
        [lhs, rhs],
        [result],
    )
    model = helper.make_model(
        graph,
        producer_name="balalaika-container-smoke",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = min(model.ir_version, 10)
    onnx.save(model, path)


def _run_ort(provider: str, *, trt_cache: Path | None = None) -> None:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if provider not in available:
        raise RuntimeError(f"{provider} is unavailable; providers={available}")

    with tempfile.TemporaryDirectory(prefix="balalaika-ort-") as tmp_dir:
        model_path = Path(tmp_dir) / "add.onnx"
        _make_add_model(model_path)
        if provider == "TensorrtExecutionProvider":
            assert trt_cache is not None
            trt_cache.mkdir(parents=True, exist_ok=True)
            providers = [
                (
                    provider,
                    {
                        "device_id": 0,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(trt_cache),
                    },
                ),
                ("CUDAExecutionProvider", {"device_id": 0}),
            ]
        else:
            providers = [(provider, {"device_id": 0})]

        session_options = ort.SessionOptions()
        session_options.enable_profiling = True
        session_options.profile_file_prefix = str(Path(tmp_dir) / "ort-profile")
        session = ort.InferenceSession(
            str(model_path), sess_options=session_options, providers=providers
        )
        if session.get_providers()[0] != provider:
            raise RuntimeError(
                f"Requested {provider}, session initialized {session.get_providers()}"
            )
        lhs = np.arange(4, dtype=np.float32).reshape(2, 2)
        rhs = np.ones((2, 2), dtype=np.float32)
        actual = session.run(None, {"lhs": lhs, "rhs": rhs})[0]
        np.testing.assert_allclose(actual, lhs + rhs)
        profile_path = Path(session.end_profiling())
        profile_events = json.loads(profile_path.read_text(encoding="utf-8"))
        executed_by = {
            event.get("args", {}).get("provider")
            for event in profile_events
            if isinstance(event, dict)
        }
        if provider not in executed_by:
            raise RuntimeError(
                f"No profiled node ran on {provider}; observed providers={executed_by}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tensorrt",
        action="store_true",
        help="Also build and run a tiny TensorRT engine on the visible GPU.",
    )
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(
            "This pilot smoke test requires CUDA_VISIBLE_DEVICES=0; "
            f"received {visible!r}"
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Container must see exactly one GPU; "
            f"torch reports {torch.cuda.device_count()}"
        )

    left = torch.arange(16, dtype=torch.float32, device="cuda:0").reshape(4, 4)
    product = left @ left.T
    torch.cuda.synchronize(0)
    if not torch.isfinite(product).all().item():
        raise RuntimeError("CUDA matmul produced non-finite values")

    _run_ort("CUDAExecutionProvider")
    if args.tensorrt:
        _run_ort(
            "TensorrtExecutionProvider",
            trt_cache=Path("/cache/trt/smoke"),
        )
    _check_ffmpeg()

    import onnxruntime as ort

    print("Balalaika container smoke test passed")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Torch: {torch.__version__}")
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"ORT providers: {ort.get_available_providers()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
