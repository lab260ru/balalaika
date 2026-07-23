"""Micro-benchmark for BS.1770 integrated loudness (stage 1 fused path + stage 3).

    python -m benchmarking.micro.bench_loudness --label check
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyloudnorm as pyln

from src.preprocess.audio_postprocessing import _integrated_loudness_fast


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--files", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    paths = sorted((REPO_ROOT / "cache/bench_sample/audio").rglob("*.wav"))[: args.files]
    clips = []
    for p in paths:
        audio, rate = sf.read(p)
        clips.append((np.asarray(audio), int(rate)))
    total_sec = sum(len(a) / r for a, r in clips)
    print(f"{len(clips)} clips, {total_sec:.0f} audio-seconds")

    meters = {r: pyln.Meter(r, block_size=0.400) for r in {r for _, r in clips}}

    timings = {"stock": [], "fast": []}
    mismatches = 0
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        stock_vals = [meters[r].integrated_loudness(a.copy()) for a, r in clips]
        t1 = time.perf_counter()
        fast_vals = [_integrated_loudness_fast(meters[r], a.copy()) for a, r in clips]
        t2 = time.perf_counter()
        timings["stock"].append(t1 - t0)
        timings["fast"].append(t2 - t1)
        mismatches = sum(1 for s, f in zip(stock_vals, fast_vals) if s != f and not (np.isnan(s) and np.isnan(f)))
        print(f"stock {t1 - t0:.3f}s  fast {t2 - t1:.3f}s  LUFS mismatches {mismatches}/{len(clips)}")

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "files": len(clips),
        "audio_seconds": total_sec,
        "mismatches": mismatches,
        "stock_best_s": min(timings["stock"]),
        "fast_best_s": min(timings["fast"]),
        "speedup": min(timings["stock"]) / min(timings["fast"]),
        "stock_ms_per_file": min(timings["stock"]) / len(clips) * 1000,
        "fast_ms_per_file": min(timings["fast"]) / len(clips) * 1000,
    }
    out_path = REPO_ROOT / "benchmarking" / "reports" / "micro" / "loudness.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"speedup {record['speedup']:.2f}x  saved -> {out_path}")


if __name__ == "__main__":
    main()
