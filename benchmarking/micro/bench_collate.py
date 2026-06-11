"""Micro-benchmark for collate.py hot paths (ASR consistency, sidecar reads).

    python -m benchmarking.micro.bench_collate --label before
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

PHRASES = [
    "привет как дела",
    "сегодня хорошая погода",
    "балалайка играет громко",
    "",
]


def synth_df(n_rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {}
    base = rng.integers(0, len(PHRASES), size=n_rows)
    for name in ["giga_ctc", "giga_rnnt", "vosk", "tone", "gigaam-v3-e2e-ctc"]:
        # 70% of rows agree with the base phrase, others diverge / are empty
        agree = rng.random(n_rows) < 0.7
        other = rng.integers(0, len(PHRASES), size=n_rows)
        idx = np.where(agree, base, other)
        vals = [PHRASES[i] for i in idx]
        # sprinkle NaN like missing sidecars
        nan_mask = rng.random(n_rows) < 0.1
        cols[name] = pd.Series(vals).mask(nan_mask)
    cols["filepath"] = [f"/d/{i}.wav" for i in range(n_rows)]
    return pd.DataFrame(cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    from src.collate import add_asr_consistency_column

    model_names = ["giga_ctc", "giga_rnnt", "vosk", "tone", "gigaam-v3-e2e-ctc"]
    times = []
    for _ in range(args.repeats):
        df = synth_df(args.rows)
        t0 = time.perf_counter()
        out = add_asr_consistency_column(df, model_names)
        times.append(time.perf_counter() - t0)
    print(
        f"asr_consistency rows={args.rows} avg={statistics.mean(times):.3f}s "
        f"min={min(times):.3f}s  (non-nan: {out['asr_consistency_percent'].notna().sum()})"
    )

    out_path = REPO_ROOT / "benchmarking" / "reports" / "micro" / "collate.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "label": args.label,
                    "at": datetime.now(timezone.utc).isoformat(),
                    "rows": args.rows,
                    "asr_consistency_s": times,
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
