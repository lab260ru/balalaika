#!/usr/bin/env python3
"""Export the fine-tuned WavLM music detector to ONNX."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModel


class MusicDetectionModel(nn.Module):
    """Standalone copy of the architecture used by musicdetection."""

    def __init__(self, base_model: str) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(base_model)
        self.wavlm = AutoModel.from_pretrained(base_model, config=config)
        self.pool_attention = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.wavlm(
            input_values,
            attention_mask=attention_mask,
        ).last_hidden_state

        input_length = attention_mask.size(1)
        hidden_length = hidden_states.size(1)
        ratio = input_length / hidden_length
        indices = (
            torch.arange(hidden_length, device=attention_mask.device) * ratio
        ).long()
        pooled_mask = attention_mask[:, indices].bool()

        attention_weights = self.pool_attention(hidden_states)
        attention_weights = attention_weights.masked_fill(
            ~pooled_mask.unsqueeze(-1), -1e9
        )
        attention_weights = F.softmax(attention_weights, dim=1)
        pooled = torch.sum(hidden_states * attention_weights, dim=1)
        return torch.sigmoid(self.classifier(pooled))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        default="microsoft/wavlm-base-plus",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = MusicDetectionModel(args.base_model)
    state_dict = load_file(str(args.weights), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    sample_count = max(400, round(args.sample_seconds * 16_000))
    input_values = torch.zeros(2, sample_count, dtype=torch.float32)
    attention_mask = torch.ones(2, sample_count, dtype=torch.int32)
    attention_mask[1, sample_count * 3 // 4 :] = 0  # noqa: E203

    # CPU tracing needs the math scaled-dot-product attention implementation.
    torch.backends.cuda.enable_math_sdp(True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (input_values, attention_mask),
            str(args.output),
            input_names=["input_values", "attention_mask"],
            output_names=["music_probability"],
            dynamic_axes={
                "input_values": {0: "batch_size", 1: "audio_samples"},
                "attention_mask": {0: "batch_size", 1: "audio_samples"},
                "music_probability": {0: "batch_size"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    exported = onnx.load(str(args.output))
    exported.producer_name = "balalaika"
    onnx.helper.set_model_props(
        exported,
        {
            "architecture": "WavLMForMusicDetection",
            "base_model": args.base_model,
            "weights_sha256": sha256_file(args.weights),
            "sample_rate": "16000",
            "input_values": "float32[batch_size,audio_samples]",
            "attention_mask": "int32[batch_size,audio_samples]",
            "output": "float32[batch_size,1]",
        },
    )
    onnx.checker.check_model(exported)
    onnx.save(exported, str(args.output))

    size_mib = args.output.stat().st_size / 2**20
    print(f"Exported {args.output} ({size_mib:.2f} MiB)")


if __name__ == "__main__":
    main()
