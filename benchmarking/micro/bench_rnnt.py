"""Micro-benchmark for batched stateful RNN-T greedy decode (stage 7).

Times stock onnx-asr per-utterance decode vs the batched
``src.transcription.fast_rnnt`` decode on real audio, for the transducer
models (giga_rnnt, vosk), at batch sizes 1/4/8, on CPU (clean numbers) and
GPU 1 (contended on this node — label accordingly).  Reports items/s and
ms/file for the WHOLE recognize pass (encoder + decode), and — because the
stage critical path is the decode — the decode-only ms/file split so the
projected effect on the bottleneck is visible.

With ``--impl both`` it also proves equivalence: at each batch size,
stock-bsN vs fast-bsN must agree on text, timestamps and token streams
(the encoder is shared, so this isolates the decode).  Reported as
mismatch counts.

    # clean CPU numbers + equivalence proof (default 250 files):
    python -m benchmarking.micro.bench_rnnt --device cpu --label cpu
    # contended GPU spot check (~50 files):
    CUDA_VISIBLE_DEVICES=1 python -m benchmarking.micro.bench_rnnt \
        --device cuda --num-files 50 --label gpu1-contended
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="WARNING")

import torch  # noqa: E402
import torchaudio  # noqa: E402
import onnx_asr  # noqa: E402
from torch.nn.utils.rnn import pad_sequence  # noqa: E402

from src.transcription.fast_rnnt import patch_model, is_patched  # noqa: E402

DEFAULT_AUDIO = REPO_ROOT / "cache" / "bench_sample" / "audio"
# benchmark only the transducer models the fast path covers.
MODELS = {
    "giga_rnnt": "gigaam-v3-rnnt",
    "vosk": "alphacep/vosk-model-ru",
}


def _load_wav(path: str, sample_rate: int) -> torch.Tensor:
    waveform, sr = torchaudio.load_with_torchcodec(path)
    waveform = waveform.to(torch.float32)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform.squeeze(0).contiguous()


def _recognize(model, wavs):
    lengths = torch.tensor([w.numel() for w in wavs], dtype=torch.int64)
    padded = pad_sequence(wavs, batch_first=True).contiguous()
    wn = np.asarray(padded.numpy(), dtype=np.float32)
    ln = np.asarray(lengths.numpy(), dtype=np.int64)
    return list(model._recognize_batch(wn, ln))


def _run_pass(model, wavs, batch_size):
    out = []
    t0 = time.perf_counter()
    for i in range(0, len(wavs), batch_size):
        out.extend(_recognize(model, wavs[i : i + batch_size]))
    return out, time.perf_counter() - t0


def _decode_only_ms_per_file(asr, wavs, batch_size):
    """Time just the _decoding loop (the stage critical path), encoder
    pre-run and excluded, to project the bottleneck effect."""
    total = 0.0
    n = 0
    for i in range(0, len(wavs), batch_size):
        chunk = wavs[i : i + batch_size]
        lengths = torch.tensor([w.numel() for w in chunk], dtype=torch.int64)
        padded = pad_sequence(chunk, batch_first=True).contiguous()
        wn = np.asarray(padded.numpy(), dtype=np.float32)
        ln = np.asarray(lengths.numpy(), dtype=np.int64)
        enc, enc_lens = asr._encode(*asr._preprocessor(wn, ln))
        t0 = time.perf_counter()
        _ = list(asr._decoding(enc, enc_lens))
        total += time.perf_counter() - t0
        n += len(chunk)
    return 1000.0 * total / max(n, 1)


def _result_text(r):
    return r.text if hasattr(r, "text") else str(r)


def _result_ts(r):
    return getattr(r, "timestamps", None)


def _result_tokens(r):
    return getattr(r, "tokens", None)


def _compare(stock_out, fast_out):
    text = sum(1 for a, b in zip(stock_out, fast_out) if _result_text(a) != _result_text(b))
    ts = sum(1 for a, b in zip(stock_out, fast_out) if _result_ts(a) != _result_ts(b))
    tok = sum(1 for a, b in zip(stock_out, fast_out) if _result_tokens(a) != _result_tokens(b))
    return {"text": text, "timestamps": ts, "tokens": tok, "n": len(fast_out)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--impl", choices=["stock", "fast", "both"], default="both")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--audio-dir", default=str(DEFAULT_AUDIO))
    ap.add_argument("--num-files", type=int, default=250)
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    ap.add_argument("--with-timestamps", action="store_true", default=True)
    ap.add_argument("--no-timestamps", dest="with_timestamps", action="store_false")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    files = sorted(str(audio_dir / f) for f in os.listdir(audio_dir) if f.endswith(".wav"))
    files = files[: args.num_files]
    if not files:
        print(f"No wavs in {audio_dir}")
        return

    if args.device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        contended = True
    else:
        providers = ["CPUExecutionProvider"]
        contended = False

    print(f"{len(files)} files, device={args.device}, batch_sizes={args.batch_sizes}, "
          f"timestamps={args.with_timestamps}")

    records = []
    for short in args.models:
        onnx_name = MODELS[short]
        print(f"\n===== {short} ({onnx_name}) =====")
        sample_rate = 16_000
        wavs = [_load_wav(f, sample_rate) for f in files]

        impls = ["stock", "fast"] if args.impl == "both" else [args.impl]
        loaded = {}
        for impl in impls:
            m = onnx_asr.load_model(onnx_name, providers=providers)
            if args.with_timestamps:
                m = m.with_timestamps()
            if impl == "fast":
                patch_model(m, strict=True)
                assert is_patched(m), "fast path did not patch"
            loaded[impl] = m

        for bs in args.batch_sizes:
            timings = {}
            outs = {}
            decode_ms = {}
            for impl in impls:
                m = loaded[impl]
                _run_pass(m, wavs[: min(8, len(wavs))], bs)  # warmup
                best = None
                for _ in range(args.repeats):
                    out, dt = _run_pass(m, wavs, bs)
                    best = dt if best is None else min(best, dt)
                    outs[impl] = out
                timings[impl] = best
                decode_ms[impl] = _decode_only_ms_per_file(m.asr, wavs, bs)
                ips = len(wavs) / best
                mspf = 1000.0 * best / len(wavs)
                print(f"  [{impl} bs={bs}] {best:.2f}s  {ips:.1f} it/s  {mspf:.1f} ms/file  "
                      f"(decode-only {decode_ms[impl]:.2f} ms/file)")

            cmp = None
            if len(impls) == 2:
                cmp = _compare(outs["stock"], outs["fast"])
                print(f"  [bs={bs}] equivalence stock-vs-fast: "
                      f"text {cmp['text']}/{cmp['n']}  timestamps {cmp['timestamps']}/{cmp['n']}  "
                      f"tokens {cmp['tokens']}/{cmp['n']}")
                speedup = timings["stock"] / timings["fast"] if timings["fast"] else 0.0
                dspeed = decode_ms["stock"] / decode_ms["fast"] if decode_ms["fast"] else 0.0
                print(f"  [bs={bs}] speedup: full {speedup:.2f}x  decode-only {dspeed:.2f}x")

            records.append({
                "model": short,
                "batch_size": bs,
                "device": args.device,
                "contended": contended,
                "with_timestamps": args.with_timestamps,
                "n_files": len(wavs),
                "timings_s": {k: v for k, v in timings.items()},
                "decode_ms_per_file": decode_ms,
                "mismatches": cmp,
            })

    out_path = REPO_ROOT / "benchmarking" / "reports" / "micro" / "rnnt.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "label": args.label,
                **rec,
            }, ensure_ascii=False) + "\n")
    print(f"\nsaved {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
