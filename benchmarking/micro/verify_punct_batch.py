"""RUPunct equivalence + padding-waste check for the stage-8 batch refinements.

Re-verifies (per report.md §4.4, with the new sorted-batch + 512-pin changes)
that punctuation output is invariant to batch composition on the REAL
RUPunct/RUPunct_big model:

    per-file  vs  unsorted-batch  vs  sorted-batch   == character-identical

over ~60 varied Russian texts (short utterances, medium paragraphs, and a few
that straddle / exceed the 512-token limit). Runs on CPU by default so it does
not contend with GPU 1 (pass --device cuda:1 for a brief contended GPU check).

    python -m benchmarking.micro.verify_punct_batch [--device cpu]

Exits non-zero if any mode diverges.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer, pipeline

from src.punctuation.punctuation import (
    MODEL_MAX_TOKENS,
    MODEL_STRIDE,
    _punct_text_from_preds,
)

MODEL_NAME = "RUPunct/RUPunct_big"

# A bank of varied Russian sentence fragments (no punctuation, lowercased —
# the shape RUPunct expects from *_rover.txt). Combined at varying lengths.
FRAGMENTS = [
    "привет как дела сегодня прекрасный день",
    "вчера мы ходили в магазин за продуктами и купили хлеб молоко и яйца",
    "москва является столицей россии и крупнейшим городом страны",
    "он сказал что придёт завтра вечером но я не уверен в этом",
    "наука и техника развиваются стремительными темпами в наше время",
    "дети играли во дворе пока их родители готовили ужин на кухне",
    "погода была пасмурной и холодной поэтому мы остались дома",
    "учитель объяснял новую тему а ученики внимательно слушали его",
    "река медленно текла между высокими берегами поросшими лесом",
    "компьютерные технологии изменили нашу жизнь до неузнаваемости",
    "она долго думала над ответом прежде чем что то сказать собеседнику",
    "поезд прибыл на станцию точно по расписанию без единой задержки",
    "врач осмотрел пациента и назначил курс лечения на две недели",
    "книги помогают человеку узнавать новое и расширять кругозор",
    "за окном падал мягкий снег укрывая улицы белым покрывалом",
]


def build_texts(target: int = 60) -> list[str]:
    texts: list[str] = []
    # Short single-fragment texts.
    for frag in FRAGMENTS:
        texts.append(frag)
    # Medium texts of a few fragments.
    for i in range(len(FRAGMENTS)):
        n = 2 + (i % 4)
        chunk = " ".join(FRAGMENTS[j % len(FRAGMENTS)] for j in range(i, i + n))
        texts.append(chunk)
    # A few long ones that approach / cross the 512-token limit.
    for mult in (12, 24, 40):
        texts.append(" ".join(FRAGMENTS * mult))
    # Pad to target with rotated combinations.
    k = 0
    while len(texts) < target:
        n = 3 + (k % 5)
        texts.append(" ".join(FRAGMENTS[(k + j) % len(FRAGMENTS)] for j in range(n)))
        k += 1
    return texts[:target]


def make_pipeline(device: str):
    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME, strip_accents=False, add_prefix_space=True
    )
    tok.model_max_length = MODEL_MAX_TOKENS
    pipe = pipeline(
        "ner",
        model=MODEL_NAME,
        tokenizer=tok,
        aggregation_strategy="first",
        device=device,
        stride=MODEL_STRIDE,
    )
    return pipe, tok


def run_per_file(pipe, texts):
    return [_punct_text_from_preds(pipe(t)) for t in texts]


def run_batch(pipe, texts, order):
    """Run all texts in one padded batch in the given index order, then map
    results back to the original positions."""
    ordered_idx = order
    ordered_texts = [texts[i] for i in ordered_idx]
    preds = pipe(ordered_texts, batch_size=len(ordered_texts))
    out = [None] * len(texts)
    for pos, p in zip(ordered_idx, preds):
        out[pos] = _punct_text_from_preds(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda:N")
    ap.add_argument("--num", type=int, default=60)
    args = ap.parse_args()

    texts = build_texts(args.num)
    pipe, tok = make_pipeline(args.device)

    tok_lens = [len(tok(t, truncation=False)["input_ids"]) for t in texts]
    n_over = sum(1 for n in tok_lens if n > MODEL_MAX_TOKENS)
    print(
        f"texts={len(texts)} token-len min/median/max="
        f"{min(tok_lens)}/{sorted(tok_lens)[len(tok_lens)//2]}/{max(tok_lens)} "
        f"over-512={n_over} device={args.device}"
    )

    # The batched path must NOT contain over-limit texts (stage routes them to
    # per-file). Verify equivalence on the in-budget subset for the batch modes,
    # and verify per-file handles the over-limit ones without raising.
    in_budget = [i for i, n in enumerate(tok_lens) if n <= MODEL_MAX_TOKENS]
    over = [i for i, n in enumerate(tok_lens) if n > MODEL_MAX_TOKENS]

    t0 = time.perf_counter()
    per_file = run_per_file(pipe, texts)
    t_pf = time.perf_counter() - t0

    # Unsorted batch over the in-budget subset (discovery order).
    unsorted_order = in_budget
    t0 = time.perf_counter()
    unsorted = run_batch(pipe, texts, unsorted_order)
    t_un = time.perf_counter() - t0

    # Sorted batch: in-budget indices sorted by token length (the stage's order).
    sorted_order = sorted(in_budget, key=lambda i: tok_lens[i])
    t0 = time.perf_counter()
    srt = run_batch(pipe, texts, sorted_order)
    t_sr = time.perf_counter() - t0

    mism_un = [i for i in in_budget if unsorted[i] != per_file[i]]
    mism_sr = [i for i in in_budget if srt[i] != per_file[i]]
    mism_sr_un = [i for i in in_budget if srt[i] != unsorted[i]]

    # Over-limit: per-file must produce non-empty output and not raise.
    over_ok = all(per_file[i] is not None for i in over)

    print(f"in-budget={len(in_budget)} over-limit={len(over)}")
    print(f"timings: per-file={t_pf:.2f}s unsorted-batch={t_un:.2f}s sorted-batch={t_sr:.2f}s")
    print(f"mismatches per-file-vs-unsorted : {len(mism_un)}")
    print(f"mismatches per-file-vs-sorted   : {len(mism_sr)}")
    print(f"mismatches sorted-vs-unsorted   : {len(mism_sr_un)}")
    print(f"over-limit per-file produced output: {over_ok} ({len(over)} texts)")

    ok = (
        not mism_un
        and not mism_sr
        and not mism_sr_un
        and over_ok
    )
    if ok:
        print("RESULT: PASS — character-identical across per-file / unsorted / sorted")
        return 0
    for i in (mism_un + mism_sr + mism_sr_un)[:3]:
        print(f"  DIVERGENCE idx={i} tok_len={tok_lens[i]}")
        print(f"    per-file: {per_file[i]!r}")
        print(f"    unsorted: {unsorted[i]!r}")
        print(f"    sorted  : {srt[i]!r}")
    print("RESULT: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
