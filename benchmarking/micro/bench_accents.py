"""Micro-benchmark for the ruAccent stage (stage 9, accents).

Measures stock ``ruaccent.RUAccent`` vs the in-repo
``src.accents.fast_accent.FastRUAccent`` on CPU, reporting sentences/s and
ms/file, single worker and (with --workers 4) the multi-worker thread-cap
effect.  Also proves the fast path is character-identical on every fixture.

    python -m benchmarking.micro.bench_accents --make-fixtures   # once
    python -m benchmarking.micro.bench_accents --impl both --label check
    python -m benchmarking.micro.bench_accents --impl both --workers 4 --label cap

The fixture corpus mirrors the real workload (ASR-chunk transcripts: short,
~1 sentence/file, lowercase, unpunctuated) plus deliberately hard cases:
homographs in disambiguating contexts, e-restoration, OOV names/brands.  Each
run dumps outputs to ``cache/bench_fixtures/accents_outputs_<impl>_<label>.json``
for token-for-token comparison.  Results append to
``benchmarking/reports/micro/accents.jsonl``.

GPU note: the accent stage uses the CUDA EP only (TensorRT disabled — UINT8
casts).  CPU is the documented, GPU-free path and is what this bench times; a
brief contended GPU-1 number can be taken with --device cuda.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = REPO_ROOT / "cache" / "bench_fixtures" / "accents"
OUTPUT_DIR = REPO_ROOT / "cache" / "bench_fixtures"
REPORT = REPO_ROOT / "benchmarking" / "reports" / "micro" / "accents.jsonl"
WORKDIR = REPO_ROOT / "cache" / "ruaccent_workdir"
MODEL = "turbo3.1"

# Real-transcript-shaped sentence templates (lowercase, ASR-like) + homographs.
SUBJECTS = ["эксперт", "блогер", "политик", "учёный", "инженер", "музыкант",
            "историк", "экономист", "врач", "журналист", "писатель", "тренер"]
VERBS = ["рассказал", "объяснил", "показал", "обсудил", "прокомментировал",
         "поддержал", "описал", "вспомнил", "затронул", "разобрал"]
OBJECTS = ["новый проект", "последние события", "свою книгу", "эту проблему",
           "будущее технологий", "важный вопрос", "свежую идею", "этот фильм",
           "сложную тему", "интересный случай"]
# homograph words in disambiguating contexts
HOMOGRAPH_SENTS = [
    "я повесил замок на дверь потому что старый замок сломался",
    "замок стоит на горе а это стоит очень дорого",
    "через дорогу шла дорога и стояли окна",
    "мука дорогая а мука творчества бесценна",
    "окна открыты закрой окна белки прыгают кормим белки",
    "стоит дом стоит ли это того дорога домой была долгой",
]
YO_SENTS = [
    "ёлки палки ещё один ёжик пробежал мимо",
    "пёс и пес всё и все ещё и еще объём растёт",
    "берёза чёрный тёплый актёр и щётка",
]
OOV_SENTS = [
    "сегодня обсудим блогершу и стримера на тиктоке",
    "криптовалюта веб разработчик фрилансе питоне джанго докере кубернетес",
    "хабиб зеленский илон маск навального дудя оксимирона",
]


def make_fixtures(n_files: int = 250, seed: int = 17):
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    specials = HOMOGRAPH_SENTS + YO_SENTS + OOV_SENTS
    files = []
    for i in range(n_files):
        # ~1 sentence per file, like real ASR-chunk transcripts; sprinkle the
        # hard cases through ~40% of files.
        if rng.random() < 0.4:
            text = rng.choice(specials)
        else:
            text = f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)} в нашем выпуске"
        # a minority are multi-sentence (longer monologues)
        if rng.random() < 0.15:
            extra = f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)}"
            text = text + ". " + extra
        files.append(text)
    for i, t in enumerate(files):
        (FIXTURE_DIR / f"text_{i:04d}.txt").write_text(t, encoding="utf-8")
    print(f"Wrote {len(files)} fixtures to {FIXTURE_DIR}")
    return files


def load_fixtures():
    if not FIXTURE_DIR.exists():
        make_fixtures()
    return [
        p.read_text(encoding="utf-8")
        for p in sorted(FIXTURE_DIR.glob("text_*.txt"))
    ]


def providers(device: str):
    if device == "cuda":
        return [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_stock(device, intra_op_threads):
    from ruaccent import RUAccent

    from src.accents.fast_accent import capped_onnx_threads

    acc = RUAccent()
    t0 = time.perf_counter()
    with capped_onnx_threads(intra_op_threads):
        acc.load(
            omograph_model_size=MODEL,
            use_dictionary=True,
            tiny_mode=False,
            providers=providers(device),
            workdir=str(WORKDIR),
        )
    return acc, time.perf_counter() - t0


def load_fast(device, intra_op_threads):
    from src.accents.fast_accent import FastRUAccent, capped_onnx_threads

    acc = FastRUAccent()
    t0 = time.perf_counter()
    with capped_onnx_threads(intra_op_threads):
        acc.load(
            omograph_model_size=MODEL,
            use_dictionary=True,
            tiny_mode=False,
            providers=providers(device),
            workdir=str(WORKDIR),
        )
    return acc, time.perf_counter() - t0


def run_stock(acc, texts):
    return [acc.process_all(t) for t in texts]


def run_fast(acc, texts, batch_size):
    out = []
    for i in range(0, len(texts), batch_size):
        out.extend(acc.process_batch(texts[i : i + batch_size]))
    return out


def _impl_threads(impl, intra_op_threads, stock_intra_op_threads):
    # Production reality: the fast path caps intra-op threads; stock ruAccent
    # has no SessionOptions hook so each session defaults to all cores (0).
    return stock_intra_op_threads if impl == "stock" else intra_op_threads


def _worker_run(args):
    impl, texts, device, intra_op_threads, stock_iot, batch_size = args
    iot = _impl_threads(impl, intra_op_threads, stock_iot)
    if impl == "stock":
        acc, init_s = load_stock(device, iot)
        t0 = time.perf_counter()
        run_stock(acc, texts)
        return init_s, time.perf_counter() - t0
    acc, init_s = load_fast(device, iot)
    t0 = time.perf_counter()
    run_fast(acc, texts, batch_size)
    return init_s, time.perf_counter() - t0


def bench_one(
    impl, texts, device, intra_op_threads, stock_iot, batch_size, workers
):
    from ruaccent.text_preprocessor import TextPreprocessor

    n_sent = sum(len(TextPreprocessor.split_by_sentences(t)) for t in texts)
    iot = _impl_threads(impl, intra_op_threads, stock_iot)
    if workers == 1:
        if impl == "stock":
            acc, init_s = load_stock(device, iot)
            acc.process_all("разминка модели")  # warm
            t0 = time.perf_counter()
            outs = run_stock(acc, texts)
            wall = time.perf_counter() - t0
        else:
            acc, init_s = load_fast(device, iot)
            acc.process_batch(["разминка модели"])  # warm
            t0 = time.perf_counter()
            outs = run_fast(acc, texts, batch_size)
            wall = time.perf_counter() - t0
        return {
            "wall_s": wall,
            "init_s": init_s,
            "ms_per_file": 1000 * wall / len(texts),
            "files_per_s": len(texts) / wall,
            "sent_per_s": n_sent / wall,
            "outputs": outs,
        }
    # multi-worker: split files across `workers` processes, time the slowest.
    import multiprocessing as mp

    shards = [texts[i::workers] for i in range(workers)]
    args = [
        (impl, s, device, intra_op_threads, stock_iot, batch_size) for s in shards
    ]
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(workers) as pool:
        results = pool.map(_worker_run, args)
    wall = time.perf_counter() - t0
    init_s = max(r[0] for r in results)
    return {
        "wall_s": wall,
        "init_s": init_s,
        "ms_per_file": 1000 * wall / len(texts),
        "files_per_s": len(texts) / wall,
        "sent_per_s": n_sent / wall,
        "outputs": None,  # outputs not collected in the multi-worker timing path
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["stock", "fast", "both"], default="both")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--intra-op-threads", type=int, default=4,
                    help="intra-op thread cap for the FAST path (production knob)")
    ap.add_argument("--stock-intra-op-threads", type=int, default=0,
                    help="intra-op threads for STOCK (0 = ruAccent default = all cores)")
    ap.add_argument("--n-files", type=int, default=250)
    ap.add_argument("--label", default="adhoc")
    ap.add_argument("--make-fixtures", action="store_true")
    args = ap.parse_args()

    if args.make_fixtures:
        make_fixtures(args.n_files)
        return

    texts = load_fixtures()[: args.n_files]
    impls = ["stock", "fast"] if args.impl == "both" else [args.impl]
    results = {}
    for impl in impls:
        res = bench_one(
            impl, texts, args.device, args.intra_op_threads,
            args.stock_intra_op_threads, args.batch_size, args.workers,
        )
        results[impl] = res
        print(
            f"[{impl:5s}] workers={args.workers} device={args.device} "
            f"wall={res['wall_s']:.2f}s init={res['init_s']:.1f}s "
            f"{res['ms_per_file']:.1f} ms/file {res['files_per_s']:.0f} files/s "
            f"{res['sent_per_s']:.0f} sent/s"
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if res["outputs"] is not None:
            (OUTPUT_DIR / f"accents_outputs_{impl}_{args.label}.json").write_text(
                json.dumps(res["outputs"], ensure_ascii=False)
            )

    if "stock" in results and "fast" in results:
        sp = results["stock"]["wall_s"] / results["fast"]["wall_s"]
        print(f"speedup (stock/fast): {sp:.2f}x")
        so = results["stock"].get("outputs")
        fo = results["fast"].get("outputs")
        if so is not None and fo is not None:
            diffs = sum(1 for a, b in zip(so, fo) if a != b)
            print(f"equivalence: {diffs}/{len(so)} character diffs")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        for impl, res in results.items():
            f.write(
                json.dumps(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "label": args.label,
                        "impl": impl,
                        "device": args.device,
                        "workers": args.workers,
                        "batch_size": args.batch_size,
                        "intra_op_threads": args.intra_op_threads,
                        "n_files": len(texts),
                        "wall_s": res["wall_s"],
                        "init_s": res["init_s"],
                        "ms_per_file": res["ms_per_file"],
                        "files_per_s": res["files_per_s"],
                        "sent_per_s": res["sent_per_s"],
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
