"""Micro-benchmark for the TryIParu G2P stage (stage 10, phonemizer).

Run BEFORE and AFTER an optimization with identical fixtures:

    python -m benchmarking.micro.bench_g2p --make-fixtures        # once
    python -m benchmarking.micro.bench_g2p --impl stock --label before
    ... apply patch ...
    python -m benchmarking.micro.bench_g2p --impl fast --label after

Fixtures are deterministic Russian texts built from tryiparu's own dictionary
plus synthetic OOV words (mutations that are guaranteed absent from the
dictionary): bare space-joined words, like ROVER ASR output, but with no
punctuation/digit tokens — those branches are covered functionally by
``tests/test_phonemizer_fast_g2p.py``, not timed here.  Outputs of every run
are dumped to ``cache/bench_fixtures/g2p_outputs_<impl>_<label>.json`` so
stock vs fast can be compared token-for-token.  Note ``init_s`` asymmetry:
the fast impl reads/creates ``cache/g2p_dict.pkl``, so its first-ever run on
a node pays the CSV parse and later runs don't; stock always pays the pandas
parse.

Results are appended to ``benchmarking/reports/micro/g2p.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = REPO_ROOT / "cache" / "bench_fixtures" / "g2p"
OUTPUT_DIR = REPO_ROOT / "cache" / "bench_fixtures"
REPORT = REPO_ROOT / "benchmarking" / "reports" / "micro" / "g2p.jsonl"

OOV_SUFFIXES = [
    "ейшество", "ируемость", "озавр", "анутый", "ификация",
    "ёвина", "ышко", "евидность", "англ", "плекс",
]


def load_dict_words():
    import csv

    import tryiparu

    path = Path(tryiparu.__file__).parent / "data" / "cleaned_dataset.csv"
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return [row[0] for row in reader if row]


def make_oov_words(dict_words, vocab, rng, count):
    """Deterministic pseudo-words guaranteed absent from the dictionary."""
    oov = []
    seen = set()
    while len(oov) < count:
        base = rng.choice(dict_words)
        word = base + rng.choice(OOV_SUFFIXES)
        if len(word) > 24:
            word = word[:24]
        if word in vocab or word in seen or not word.isalpha():
            continue
        seen.add(word)
        oov.append(word)
    return oov


def make_fixtures():
    rng = random.Random(1337)
    dict_words = load_dict_words()
    vocab = set(dict_words)
    # A shared OOV pool: the same unknown word recurring across files is the
    # realistic case (speaker names, brand words in one podcast).
    oov_pool = make_oov_words(dict_words, vocab, rng, 220)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    specs = (
        [("typical", 250, 0.02)] * 24
        + [("high_oov", 200, 0.15)] * 4
        + [("short", 20, 0.05)] * 2
    )
    manifest = []
    for idx, (kind, n_words, oov_rate) in enumerate(specs):
        words = []
        for _ in range(n_words):
            if rng.random() < oov_rate:
                words.append(rng.choice(oov_pool))
            else:
                words.append(rng.choice(dict_words))
        text = " ".join(words)
        name = f"text_{idx:03d}_{kind}.txt"
        (FIXTURE_DIR / name).write_text(text, encoding="utf-8")
        manifest.append({"name": name, "kind": kind, "n_words": n_words, "oov_rate": oov_rate})
    (FIXTURE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"Wrote {len(specs)} fixtures to {FIXTURE_DIR}")


def build_model(impl: str, device: str):
    t0 = time.perf_counter()
    if impl == "stock":
        from tryiparu import G2PModel

        model = G2PModel(load_dataset=True, device=device)
    else:
        from src.phonemizer.fast_g2p import FastG2P

        model = FastG2P(device=device)
    return model, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-fixtures", action="store_true")
    parser.add_argument("--impl", choices=["stock", "fast"], default="stock")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--label", default="run")
    args = parser.parse_args()

    if args.make_fixtures:
        make_fixtures()
        return

    texts = sorted(FIXTURE_DIR.glob("text_*.txt"))
    if not texts:
        sys.exit("No fixtures; run with --make-fixtures first.")

    model, init_s = build_model(args.impl, args.device)

    # Warm up CUDA context / first-call overheads on a throwaway word, then
    # drop it from the cache so per-text timings still see it as unseen.
    warm = "бенчмаркозавр"
    t0 = time.perf_counter()
    model(warm)
    warmup_s = time.perf_counter() - t0
    model.data_dict.pop(warm, None)

    per_text = []
    outputs = {}
    total0 = time.perf_counter()
    for path in texts:
        text = path.read_text(encoding="utf-8")
        t0 = time.perf_counter()
        phonemes = model(text)
        per_text.append(time.perf_counter() - t0)
        outputs[path.name] = " ".join(phonemes)
    total_s = time.perf_counter() - total0

    out_path = OUTPUT_DIR / f"g2p_outputs_{args.impl}_{args.label}.json"
    out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=0))

    # Per-word OOV decode latency on fresh words (cache can't help).
    rng = random.Random(7331)
    dict_words = load_dict_words()
    fresh_oov = make_oov_words(dict_words, set(dict_words), rng, 50)
    fresh_oov = [w for w in fresh_oov if w not in model.data_dict][:30]
    t0 = time.perf_counter()
    if hasattr(model, "decode_batch"):
        model.decode_batch(fresh_oov)
    else:
        for w in fresh_oov:
            model.greedy_decode(src=w, max_length=model.max_length)
    oov_decode_s = time.perf_counter() - t0

    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "impl": args.impl,
        "device": args.device,
        "init_s": round(init_s, 3),
        "warmup_s": round(warmup_s, 3),
        "total_s": round(total_s, 3),
        "per_text_mean_s": round(statistics.mean(per_text), 4),
        "per_text_max_s": round(max(per_text), 4),
        "n_texts": len(texts),
        "oov_words_timed": len(fresh_oov),
        "oov_decode_s": round(oov_decode_s, 3),
        "oov_ms_per_word": round(1000 * oov_decode_s / max(1, len(fresh_oov)), 1),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
    print(f"outputs -> {out_path}")


if __name__ == "__main__":
    main()
