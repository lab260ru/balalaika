"""Micro-benchmark: Sortformer CPU post-processing + get_chunk_metrics.

Compares the old pure-Python loops against the vectorized kernels on realistic
synthetic shapes. CPU-only; the ONNX model is not needed. Run:

    python -m benchmarking.micro.bench_sortformer_postproc
    python -m benchmarking.micro.bench_sortformer_postproc --label check
"""

import argparse
import random
import time

import numpy as np

from src.preprocess.sortformer_onnx import (
    DiarizationConfig,
    Sortformer,
    EMB_DIM,
    NUM_SPEAKERS,
    FRAME_DURATION,
    MAX_INDEX,
)
from src.preprocess.preprocess import SegmentIndex, get_chunk_metrics


# ---- verbatim old implementations -----------------------------------------
def old_binarize(cfg, preds, audio_duration_sec):
    raw_segments = []
    num_frames = preds.shape[0]
    for spk in range(NUM_SPEAKERS):
        raw_intervals = []
        in_seg = False
        start_t = 0.0
        for t in range(num_frames):
            p = preds[t, spk]
            if p >= cfg.onset and not in_seg:
                in_seg = True
                start_t = t * FRAME_DURATION
            elif p < cfg.offset and in_seg:
                in_seg = False
                raw_intervals.append([start_t, t * FRAME_DURATION, spk])
        if in_seg:
            raw_intervals.append([start_t, num_frames * FRAME_DURATION, spk])
        if not raw_intervals:
            continue
        merged = [raw_intervals[0]]
        for i in range(1, len(raw_intervals)):
            if raw_intervals[i][0] - merged[-1][1] <= cfg.min_duration_off:
                merged[-1][1] = raw_intervals[i][1]
            else:
                merged.append(raw_intervals[i])
        filt = [s for s in merged if (s[1] - s[0]) >= cfg.min_duration_on]
        padded = []
        for seg in filt:
            ss = max(0.0, seg[0] - cfg.pad_onset)
            es = min(audio_duration_sec, seg[1] + cfg.pad_offset)
            if not padded:
                padded.append([ss, es, spk])
            elif ss <= padded[-1][1]:
                padded[-1][1] = max(padded[-1][1], es)
            else:
                padded.append([ss, es, spk])
        raw_segments.extend(padded)
    raw_segments.sort(key=lambda x: (x[0], x[2]))
    return [f"{s[0]} {s[1]} speaker_{s[2]}" for s in raw_segments]


def old_get_chunk_metrics(c_start, c_end, raw_segments):
    chunk_dur = c_end - c_start
    if chunk_dur <= 0:
        return 0.0, 0.0, 0
    intervals = []
    spk = set()
    for rs, re_, s in raw_segments:
        os_ = max(c_start, rs)
        oe = min(c_end, re_)
        if os_ < oe:
            intervals.append([os_, oe])
            spk.add(s)
    intervals.sort(key=lambda x: x[0])
    if not intervals:
        return 100.0, round(chunk_dur, 2), 0
    merged = []
    for it in intervals:
        if not merged:
            merged.append(it)
        elif it[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], it[1])
        else:
            merged.append(it)
    speech = sum(e - s for s, e in merged)
    silence = max(0.0, chunk_dur - speech)
    gaps = [merged[0][0] - c_start]
    for i in range(len(merged) - 1):
        gaps.append(merged[i + 1][0] - merged[i][1])
    gaps.append(c_end - merged[-1][1])
    return round((silence / chunk_dur) * 100, 2), round(max(gaps), 2), len(spk)


def _timed(fn, repeat):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_binarize(repeat):
    rng = np.random.default_rng(0)
    # 900 s window ~= 11250 frames at 0.08 s/frame.
    preds = rng.random((11250, NUM_SPEAKERS)).astype(np.float32)
    dur = preds.shape[0] * FRAME_DURATION
    cfg = DiarizationConfig()
    diar = Sortformer.__new__(Sortformer)
    diar.config = cfg
    assert old_binarize(cfg, preds, dur) == diar._binarize(preds, dur)
    old = _timed(lambda: old_binarize(cfg, preds, dur), repeat)
    new = _timed(lambda: diar._binarize(preds, dur), repeat)
    return old, new


def bench_get_chunk_metrics(repeat):
    rng = random.Random(0)
    # 2 h episode: ~2500 raw segments, ~600 final chunks.
    segs = []
    t = 0.0
    for _ in range(2500):
        s = t + rng.uniform(0, 2)
        e = s + rng.uniform(0.5, 3)
        segs.append((s, e, rng.randint(0, 3)))
        t = s + rng.uniform(0.5, 2.5)
    segs.sort(key=lambda x: x[0])
    chunks = [(rng.uniform(0, t), 0.0) for _ in range(600)]
    chunks = [(cs, cs + rng.uniform(5, 15)) for cs, _ in chunks]

    def run_old():
        for cs, ce in chunks:
            old_get_chunk_metrics(cs, ce, segs)

    def run_new():
        idx = SegmentIndex(segs)
        for cs, ce in chunks:
            get_chunk_metrics(cs, ce, segs, seg_index=idx)

    # equality spot-check
    idx = SegmentIndex(segs)
    for cs, ce in chunks[:50]:
        assert old_get_chunk_metrics(cs, ce, segs) == get_chunk_metrics(
            cs, ce, segs, seg_index=idx
        )
    return _timed(run_old, repeat), _timed(run_new, repeat)


# ---- spkcache top-k + gather (one compression trigger) --------------------
def old_boost(scores, n_boost, scale):
    for s in range(NUM_SPEAKERS):
        col = scores[:, s].copy()
        top = np.argsort(col)[::-1][:n_boost]
        vm = scores[top, s] != -np.inf
        scores[top[vm], s] -= scale * np.log(0.5)
    return scores


def old_topk(scores, n_no_sil, spkcache_len):
    n_frames = scores.shape[0]
    flat = scores.flatten("F")
    sidx = np.argsort(flat)[::-1]
    topk_flat = []
    for idx in sidx[:spkcache_len]:
        topk_flat.append(MAX_INDEX if flat[idx] == -np.inf else idx)
    topk_flat.sort()
    is_dis = [False] * spkcache_len
    fi = [0] * spkcache_len
    for i, f in enumerate(topk_flat):
        if f == MAX_INDEX:
            is_dis[i] = True
        else:
            fidx = f % n_frames
            if fidx >= n_no_sil:
                is_dis[i] = True
            else:
                fi[i] = fidx
    return fi, is_dis


def old_gather(spkcache, spkcache_preds, mean_sil_emb, indices, is_disabled, spkcache_len):
    new_embs = np.zeros((1, spkcache_len, EMB_DIM), dtype=np.float32)
    new_preds = np.zeros((1, spkcache_len, NUM_SPEAKERS), dtype=np.float32)
    cache_preds = spkcache_preds[0]
    cache_embs = spkcache[0]
    for i, (idx, disabled) in enumerate(zip(indices, is_disabled)):
        if disabled:
            new_embs[0, i, :] = mean_sil_emb[0]
        elif idx < cache_embs.shape[0]:
            new_embs[0, i, :] = cache_embs[idx]
            new_preds[0, i, :] = cache_preds[idx]
    return new_embs, np.expand_dims(new_preds[0], axis=0)


def bench_spkcache(repeat):
    rng = np.random.default_rng(0)
    n_frames = 312  # spkcache overflow size before compression
    spkcache_len = 188
    diar = Sortformer.__new__(Sortformer)
    diar.spkcache_len = spkcache_len
    diar.spkcache = rng.standard_normal((1, n_frames, EMB_DIM)).astype(np.float32)
    diar.spkcache_preds = rng.standard_normal((1, n_frames, NUM_SPEAKERS)).astype(np.float32)
    diar.mean_sil_emb = rng.standard_normal((1, EMB_DIM)).astype(np.float32)

    def make_scores():
        sc = rng.standard_normal((n_frames + 3, NUM_SPEAKERS)).astype(np.float32)
        sc[rng.random(sc.shape) < 0.3] = -np.inf
        return sc

    def run_old():
        sc = make_scores()
        sc = old_boost(sc, 30, 2.0)
        fi, dis = old_topk(sc, n_frames, spkcache_len)
        old_gather(diar.spkcache, diar.spkcache_preds, diar.mean_sil_emb, fi, dis, spkcache_len)

    def run_new():
        sc = make_scores()
        sc = diar._boost_topk_scores(sc, 30, 2.0)
        fi, dis = diar._get_topk_indices(sc, n_frames)
        diar._gather_spkcache(fi, dis)

    return _timed(run_old, repeat), _timed(run_new, repeat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="bench")
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    print(f"[{args.label}] Sortformer post-processing micro-bench (CPU)\n")
    rows = [
        ("_binarize (900 s window, 11250 frames)", *bench_binarize(args.repeat)),
        ("get_chunk_metrics (2 h ep: 2500 segs x 600 chunks)", *bench_get_chunk_metrics(args.repeat)),
        ("spkcache boost+topk(+gather) (312->188)", *bench_spkcache(args.repeat)),
    ]
    print(f"{'kernel':<52} {'old (ms)':>10} {'new (ms)':>10} {'speedup':>8}")
    print("-" * 84)
    for name, old, new in rows:
        spd = old / new if new > 0 else float("inf")
        print(f"{name:<52} {old*1e3:>10.3f} {new*1e3:>10.3f} {spd:>7.1f}x")


if __name__ == "__main__":
    main()
