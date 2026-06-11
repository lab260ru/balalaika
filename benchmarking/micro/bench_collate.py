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


MODEL_NAMES = ["giga_ctc", "giga_rnnt", "vosk", "tone", "gigaam-v3-e2e-ctc"]

# Realistic-ish sidecar payloads: a transcript per model plus a multi-KB .tst
# JSON blob, so the RAM benchmark exercises the "3x text residency" path the
# §9.8 finding is about, not just the tiny consistency frame.
_TST_BLOB = json.dumps(
    [{"w": w, "s": round(i * 0.31, 2), "e": round(i * 0.31 + 0.3, 2)}
     for i, w in enumerate(("слово " * 40).split())]
)


def _build_ram_fixture(root: Path, rows: int, files_per_dir: int = 80) -> Path:
    """Write `rows` audio rows with on-disk sidecars; return the dataset root."""
    import numpy as np

    base = root / "data"
    rng = np.random.default_rng(0)
    csv_rows = []
    n_dirs = max(1, rows // files_per_dir)
    suffixes = {
        "rover": "_rover.txt",
        "punct": "_punct.txt",
        "accent": "_accent.txt",
        "giga_ctc": "_giga_ctc.txt",
        "giga_rnnt": "_giga_rnnt.txt",
        "vosk": "_vosk.txt",
        "tone": "_tone.txt",
        "gigaam-v3-e2e-ctc": "_gigaam-v3-e2e-ctc.txt",
        "giga_ctc_ts": "_giga_ctc.tst",
    }
    written = 0
    for d in range(n_dirs):
        if written >= rows:
            break
        dir_path = base / str(d // 100) / str(d)
        dir_path.mkdir(parents=True, exist_ok=True)
        for f in range(files_per_dir):
            if written >= rows:
                break
            stem = f"chunk_{written:08d}"
            ap = dir_path / f"{stem}.wav"
            ap.write_bytes(b"x")
            phrase = PHRASES[int(rng.integers(0, len(PHRASES)))] or "тихо"
            for key, suf in suffixes.items():
                content = _TST_BLOB if suf.endswith(".tst") else phrase
                (dir_path / f"{stem}{suf}").write_text(content, encoding="utf-8")
            csv_rows.append(
                {"filepath": str(ap), "speaker_id": written % 4,
                 "total_duration": round(float(rng.uniform(1, 15)), 4),
                 "crest_factor": round(float(rng.uniform(1, 5)), 4)}
            )
            written += 1
    pd.DataFrame(csv_rows).to_csv(base / "balalaika.csv", index=False)
    return base


def _peak_rss_mb() -> float:
    import resource
    # ru_maxrss is KB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _run_old_path(base: Path) -> None:
    """Verbatim old single-shot assembly (merge -> consistency -> to_parquet)."""
    import src.collate as collate

    df = pd.read_csv(base / "balalaika.csv")
    df.drop_duplicates(subset="filepath", inplace=True)
    file_types = collate.sidecar_specs(MODEL_NAMES)
    df = collate.drop_csv_text_columns(
        df, extra_columns=set(file_types) | collate.transcription_sidecar_columns(MODEL_NAMES)
    )
    audio_paths = df["filepath"].tolist()
    results = [collate.process_audio_file(p, base, file_types, {}) for p in audio_paths]
    extracted_df = pd.DataFrame(results)
    final_df = pd.merge(df, extracted_df, on="filepath", how="left")
    final_df = collate.add_asr_consistency_column(final_df, MODEL_NAMES)
    final_df.to_parquet(base / "balalaika_old.parquet", engine="pyarrow", index=False)


def _run_ram_bench(args) -> None:
    """Build a fixture and run old vs new collate assembly in subprocesses,
    each reporting peak RSS + wall. The point is the RATIO before/after."""
    import subprocess
    import tempfile

    which = getattr(args, "which", None)
    if which:  # invoked as a child for one path
        with tempfile.TemporaryDirectory():
            pass
        base = Path(args.fixture)
        t0 = time.perf_counter()
        if which == "old":
            _run_old_path(base)
        else:
            class A:
                config_path = str(base / "config.yaml")
                log_dir = str(base / "logs")
            import src.collate as collate
            collate.main(A())
        wall = time.perf_counter() - t0
        print(json.dumps({"which": which, "peak_rss_mb": _peak_rss_mb(), "wall_s": wall}))
        return

    import yaml

    tmp = Path(tempfile.mkdtemp(prefix="bench_collate_"))
    base = _build_ram_fixture(tmp, args.rows)
    cfg = {
        "download": {"podcasts_path": str(base), "num_workers": 8,
                     "collate_slab_rows": args.slab_rows,
                     "collate_parquet_compression": "zstd"},
        "transcription": {"model_names": MODEL_NAMES},
    }
    (base / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    results = {}
    for which in ("old", "new"):
        proc = subprocess.run(
            [sys.executable, "-m", "benchmarking.micro.bench_collate",
             "--label", args.label, "--mode", "ram", "--rows", str(args.rows),
             "--which", which, "--fixture", str(base)],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        line = [l for l in proc.stdout.splitlines() if l.startswith("{")]
        if not line:
            print(proc.stdout); print(proc.stderr)
            raise RuntimeError(f"child {which} produced no result")
        results[which] = json.loads(line[-1])

    # Verify read-back equality of the two parquet outputs.
    import pandas.testing as pdt
    old_df = pd.read_parquet(base / "balalaika_old.parquet").reset_index(drop=True)
    new_df = pd.read_parquet(base / "balalaika.parquet").reset_index(drop=True)
    equal = True
    try:
        assert list(old_df.columns) == list(new_df.columns)
        pdt.assert_frame_equal(new_df, old_df, check_dtype=True)
    except AssertionError as e:
        equal = False
        print("READ-BACK MISMATCH:", e)

    o, n = results["old"], results["new"]
    ratio = o["peak_rss_mb"] / n["peak_rss_mb"] if n["peak_rss_mb"] else float("nan")
    print(
        f"[collate RAM] rows={args.rows} slab={args.slab_rows}\n"
        f"  OLD peak_rss={o['peak_rss_mb']:.0f} MB  wall={o['wall_s']:.2f}s\n"
        f"  NEW peak_rss={n['peak_rss_mb']:.0f} MB  wall={n['wall_s']:.2f}s\n"
        f"  peak-RSS ratio old/new = {ratio:.2f}x   read-back equal: {equal}"
    )
    out_path = REPO_ROOT / "benchmarking" / "reports" / "micro" / "collate.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"label": args.label, "mode": "ram", "rows": args.rows,
                            "slab_rows": args.slab_rows, "old": o, "new": n,
                            "ratio": ratio, "readback_equal": equal,
                            "at": datetime.now(timezone.utc).isoformat()}) + "\n")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--mode", choices=["consistency", "ram"], default="consistency")
    ap.add_argument("--slab-rows", dest="slab_rows", type=int, default=50_000)
    ap.add_argument("--which", choices=["old", "new"], default=None,
                    help="internal: run one path as a child for RSS isolation")
    ap.add_argument("--fixture", default=None, help="internal: child fixture dir")
    args = ap.parse_args()

    if args.mode == "ram":
        _run_ram_bench(args)
        return

    from src.collate import add_asr_consistency_column

    model_names = MODEL_NAMES
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
