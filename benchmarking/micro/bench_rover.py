"""Micro-benchmark for ROVER transcript aggregation (stage 7 post-pass).

Generates ASR-like hypothesis sets (5 models per task, word-level
perturbations of a base text) and times crowd-kit's ROVER against the
fast implementation. With --impl both, every aggregated string is
compared for exact equality.

    python -m benchmarking.micro.bench_rover --label before --impl stock
    python -m benchmarking.micro.bench_rover --label after  --impl both
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
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

MODELS = ["gigaam-v3-e2e-ctc", "giga_ctc", "giga_rnnt", "vosk", "tone"]
CYRILLIC = "абвгдежзийклмнопрстуфхцчшщыэюя"


def _make_vocab(rng: np.random.Generator, size: int = 3000) -> list[str]:
    vocab = []
    for _ in range(size):
        length = int(rng.integers(2, 12))
        vocab.append("".join(rng.choice(list(CYRILLIC), size=length)))
    return vocab


def synth_tasks(n_tasks: int, seed: int = 42) -> pd.DataFrame:
    """ASR-like records: per task a base text and per-model corruptions."""
    rng = np.random.default_rng(seed)
    vocab = _make_vocab(rng)
    records = []
    for t in range(n_tasks):
        n_words = max(1, int(rng.lognormal(mean=3.2, sigma=0.5)))  # ~25 words
        base = [vocab[i] for i in rng.integers(0, len(vocab), size=n_words)]
        for model in MODELS:
            if rng.random() < 0.06:  # model produced nothing for this file
                continue
            words = []
            for w in base:
                r = rng.random()
                if r < 0.05:  # deletion
                    continue
                if r < 0.13:  # substitution
                    words.append(vocab[int(rng.integers(0, len(vocab)))])
                else:
                    words.append(w)
                if rng.random() < 0.03:  # insertion
                    words.append(vocab[int(rng.integers(0, len(vocab)))])
            text = " ".join(words)
            if rng.random() < 0.01:
                text = ""  # whitespace-only sidecars reach ROVER as '' after lower()
            records.append({"task": f"/d/{t:06d}.wav", "worker": model, "text": text})
    return pd.DataFrame.from_records(records, columns=["task", "worker", "text"])


def run_impl(impl: str, df: pd.DataFrame) -> tuple[pd.Series, float]:
    tokenizer = lambda s: s.lower().split()  # noqa: E731 — mirrors ROVERWrapper
    detokenizer = lambda tokens: " ".join(tokens)  # noqa: E731
    if impl == "stock":
        from crowdkit.aggregation import ROVER

        agg = ROVER(tokenizer, detokenizer)
    else:
        from src.transcription.fast_rover import FastROVER

        agg = FastROVER(tokenizer, detokenizer)
    start = time.perf_counter()
    result = agg.fit_predict(df.copy())
    elapsed = time.perf_counter() - start
    return result, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--impl", choices=["stock", "fast", "both"], default="both")
    ap.add_argument("--tasks", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = synth_tasks(args.tasks, seed=args.seed)
    n_hyp = len(df)
    print(f"{args.tasks} tasks, {n_hyp} hypothesis rows")

    impls = ["stock", "fast"] if args.impl == "both" else [args.impl]
    timings: dict[str, list[float]] = {}
    results: dict[str, pd.Series] = {}
    for impl in impls:
        for rep in range(args.repeats):
            result, elapsed = run_impl(impl, df)
            timings.setdefault(impl, []).append(elapsed)
            results[impl] = result
            print(f"[{impl} #{rep}] {elapsed:.3f}s  ({args.tasks / elapsed:.0f} tasks/s)")

    mismatches = None
    if len(impls) == 2:
        stock_r, fast_r = results["stock"].sort_index(), results["fast"].sort_index()
        same_index = stock_r.index.equals(fast_r.index)
        mismatches = 0 if same_index else -1
        if same_index:
            mismatches = int((stock_r.values != fast_r.values).sum())
            for task in stock_r.index[stock_r.values != fast_r.values][:5]:
                print(f"MISMATCH {task}\n  stock: {stock_r[task]!r}\n  fast:  {fast_r[task]!r}")
        print(f"output mismatches: {mismatches} / {args.tasks} (index match: {same_index})")

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "tasks": args.tasks,
        "hypothesis_rows": n_hyp,
        "seed": args.seed,
        "mismatches": mismatches,
        "timings": {
            impl: {
                "best_s": min(vals),
                "mean_s": statistics.mean(vals),
                "tasks_per_s": args.tasks / min(vals),
            }
            for impl, vals in timings.items()
        },
    }
    out_path = REPO_ROOT / "benchmarking" / "reports" / "micro" / "rover.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
