"""Exact-equality tests for the vectorized Sortformer CPU post-processing.

Each test pins a rewritten kernel in ``src/preprocess/sortformer_onnx.py``
against a VERBATIM copy of the original pure-Python implementation (captured
below) over randomized synthetic inputs plus edge cases. The Sortformer ONNX
model is absent on this node, so these kernels are exercised directly on
synthetic ``preds`` / ``embs`` / ``scores`` arrays — no model is loaded.
"""

import numpy as np
import pytest

from src.preprocess import sortformer_onnx as S
from src.preprocess.sortformer_onnx import (
    DiarizationConfig,
    Sortformer,
    EMB_DIM,
    NUM_SPEAKERS,
    FRAME_DURATION,
    MAX_INDEX,
    SIL_THRESHOLD,
)


# --------------------------------------------------------------------------
# Verbatim copies of the ORIGINAL implementations (pre-optimization).
# --------------------------------------------------------------------------
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
        merged_intervals = [raw_intervals[0]]
        for i in range(1, len(raw_intervals)):
            gap = raw_intervals[i][0] - merged_intervals[-1][1]
            if gap <= cfg.min_duration_off:
                merged_intervals[-1][1] = raw_intervals[i][1]
            else:
                merged_intervals.append(raw_intervals[i])
        filtered_intervals = []
        for seg in merged_intervals:
            if (seg[1] - seg[0]) >= cfg.min_duration_on:
                filtered_intervals.append(seg)
        padded_intervals = []
        for seg in filtered_intervals:
            start_s = max(0.0, seg[0] - cfg.pad_onset)
            end_s = min(audio_duration_sec, seg[1] + cfg.pad_offset)
            if not padded_intervals:
                padded_intervals.append([start_s, end_s, spk])
            else:
                if start_s <= padded_intervals[-1][1]:
                    padded_intervals[-1][1] = max(padded_intervals[-1][1], end_s)
                else:
                    padded_intervals.append([start_s, end_s, spk])
        raw_segments.extend(padded_intervals)
    raw_segments.sort(key=lambda x: (x[0], x[2]))
    return [f"{seg[0]} {seg[1]} speaker_{seg[2]}" for seg in raw_segments]


def old_update_silence_profile(mean_sil_emb, n_sil_frames, embs, preds):
    sums = np.sum(preds, axis=1)
    sil_mask = sums < SIL_THRESHOLD
    if np.any(sil_mask):
        sil_embs = embs[sil_mask]
        for emb in sil_embs:
            mean_sil_emb[0] = (mean_sil_emb[0] * n_sil_frames + emb) / (n_sil_frames + 1)
            n_sil_frames += 1
    return mean_sil_emb, n_sil_frames


def old_boost_topk_scores(scores, n_boost, scale_factor):
    for s in range(NUM_SPEAKERS):
        col = scores[:, s].copy()
        top_idx = np.argsort(col)[::-1][:n_boost]
        valid_mask = scores[top_idx, s] != -np.inf
        scores[top_idx[valid_mask], s] -= scale_factor * np.log(0.5)
    return scores


def old_get_topk_indices(scores, n_frames_no_sil, spkcache_len):
    n_frames = scores.shape[0]
    flat_scores = scores.flatten("F")
    sorted_flat_idx = np.argsort(flat_scores)[::-1]
    topk_flat = []
    for idx in sorted_flat_idx[:spkcache_len]:
        if flat_scores[idx] == -np.inf:
            topk_flat.append(MAX_INDEX)
        else:
            topk_flat.append(idx)
    topk_flat.sort()
    is_disabled = [False] * spkcache_len
    frame_indices = [0] * spkcache_len
    for i, flat_idx in enumerate(topk_flat):
        if flat_idx == MAX_INDEX:
            is_disabled[i] = True
        else:
            frame_idx = flat_idx % n_frames
            if frame_idx >= n_frames_no_sil:
                is_disabled[i] = True
            else:
                frame_indices[i] = frame_idx
    return frame_indices, is_disabled


def old_gather_spkcache(spkcache, spkcache_preds, mean_sil_emb, indices, is_disabled, spkcache_len):
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


# --------------------------------------------------------------------------
# Helper: a Sortformer instance without touching the ONNX model.
# --------------------------------------------------------------------------
def make_diarizer(cfg=None, spkcache_len=188):
    diar = Sortformer.__new__(Sortformer)
    diar.config = cfg or DiarizationConfig()
    diar.spkcache_len = spkcache_len
    return diar


# --------------------------------------------------------------------------
# _binarize
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(60))
def test_binarize_default_config(seed):
    rng = np.random.default_rng(seed)
    num_frames = int(rng.integers(0, 400))
    preds = rng.random((num_frames, NUM_SPEAKERS)).astype(np.float32)
    dur = num_frames * FRAME_DURATION
    cfg = DiarizationConfig()
    diar = make_diarizer(cfg)
    assert old_binarize(cfg, preds, dur) == diar._binarize(preds, dur)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(min_duration_off=0.3),
        dict(min_duration_on=0.5),
        dict(pad_onset=0.2, pad_offset=0.1),
        dict(min_duration_on=0.4, min_duration_off=0.25, pad_onset=0.1, pad_offset=0.2),
    ],
)
@pytest.mark.parametrize("seed", range(20))
def test_binarize_threshold_with_postfilters(kwargs, seed):
    # onset == offset (default 0.5) but merge/filter/pad knobs vary.
    rng = np.random.default_rng(seed + 1000)
    num_frames = int(rng.integers(0, 300))
    preds = rng.random((num_frames, NUM_SPEAKERS)).astype(np.float32)
    dur = num_frames * FRAME_DURATION
    cfg = DiarizationConfig(**kwargs)
    diar = make_diarizer(cfg)
    assert old_binarize(cfg, preds, dur) == diar._binarize(preds, dur)


@pytest.mark.parametrize("kwargs", [dict(onset=0.6, offset=0.4), dict(onset=0.7, offset=0.2)])
@pytest.mark.parametrize("seed", range(20))
def test_binarize_true_hysteresis_falls_back(kwargs, seed):
    # onset != offset keeps the verbatim sequential loop; pin it stays identical.
    rng = np.random.default_rng(seed + 2000)
    num_frames = int(rng.integers(0, 300))
    preds = rng.random((num_frames, NUM_SPEAKERS)).astype(np.float32)
    dur = num_frames * FRAME_DURATION
    cfg = DiarizationConfig(**kwargs)
    diar = make_diarizer(cfg)
    assert old_binarize(cfg, preds, dur) == diar._binarize(preds, dur)


@pytest.mark.parametrize("num_frames", [0, 1, 2, 3])
def test_binarize_tiny_inputs(num_frames):
    preds = np.zeros((num_frames, NUM_SPEAKERS), dtype=np.float32)
    if num_frames:
        preds[0, 0] = 1.0  # single active frame
    dur = num_frames * FRAME_DURATION
    cfg = DiarizationConfig()
    diar = make_diarizer(cfg)
    assert old_binarize(cfg, preds, dur) == diar._binarize(preds, dur)


@pytest.mark.parametrize("val", [0.0, 0.5, 1.0])
def test_binarize_constant_and_tie(val):
    # all-silence (0), all-on (1), and exactly-onset ties (0.5 >= 0.5 -> active).
    preds = np.full((50, NUM_SPEAKERS), val, dtype=np.float32)
    cfg = DiarizationConfig()
    diar = make_diarizer(cfg)
    assert old_binarize(cfg, preds, 4.0) == diar._binarize(preds, 4.0)


# --------------------------------------------------------------------------
# silence-profile running mean
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(40))
def test_silence_profile_running_mean(seed):
    rng = np.random.default_rng(seed)
    k = int(rng.integers(1, 130))
    embs = rng.standard_normal((k, EMB_DIM)).astype(np.float32)
    preds = rng.random((k, NUM_SPEAKERS)).astype(np.float32) * 0.1  # mostly silence
    n0 = int(rng.integers(0, 100))
    mean0 = rng.standard_normal((1, EMB_DIM)).astype(np.float32)

    old_mean, old_n = old_update_silence_profile(mean0.copy(), n0, embs, preds.copy())

    diar = make_diarizer()
    diar.mean_sil_emb = mean0.copy()
    diar.n_sil_frames = n0
    diar._update_silence_profile(embs, preds.copy())

    assert np.array_equal(old_mean, diar.mean_sil_emb), "silence mean not bit-identical"
    assert old_n == diar.n_sil_frames


def test_silence_profile_no_silence_frames():
    embs = np.random.randn(10, EMB_DIM).astype(np.float32)
    preds = np.ones((10, NUM_SPEAKERS), dtype=np.float32)  # all speech -> no update
    diar = make_diarizer()
    diar.mean_sil_emb = np.zeros((1, EMB_DIM), dtype=np.float32)
    diar.n_sil_frames = 0
    diar._update_silence_profile(embs, preds)
    assert diar.n_sil_frames == 0
    assert np.array_equal(diar.mean_sil_emb, np.zeros((1, EMB_DIM), dtype=np.float32))


def test_silence_running_mean_py_reference_matches_runner():
    # The numba runner (if active) must equal the pure-Python reference.
    rng = np.random.default_rng(7)
    for _ in range(20):
        k = int(rng.integers(1, 130))
        embs = rng.standard_normal((k, EMB_DIM)).astype(np.float32)
        n0 = int(rng.integers(0, 100))
        mean0 = rng.standard_normal(EMB_DIM).astype(np.float32)
        ref_mean, ref_n = S._silence_running_mean_py(mean0.copy(), n0, embs)
        run_mean, run_n = S._silence_running_mean(mean0.copy(), n0, embs)
        assert np.array_equal(ref_mean, run_mean)
        assert ref_n == run_n


# --------------------------------------------------------------------------
# _boost_topk_scores
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(60))
def test_boost_topk_scores(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 250))
    n_boost = int(rng.integers(0, 60))
    scores = rng.standard_normal((n, NUM_SPEAKERS)).astype(np.float32)
    scores[rng.random((n, NUM_SPEAKERS)) < 0.3] = -np.inf
    diar = make_diarizer()
    a = old_boost_topk_scores(scores.copy(), n_boost, 2.0)
    b = diar._boost_topk_scores(scores.copy(), n_boost, 2.0)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# _get_topk_indices
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(80))
def test_get_topk_indices(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 260))
    spkcache_len = 188
    scores = rng.standard_normal((n, NUM_SPEAKERS)).astype(np.float32)
    scores[rng.random((n, NUM_SPEAKERS)) < 0.35] = -np.inf
    n_no_sil = max(0, n - 3)
    diar = make_diarizer(spkcache_len=spkcache_len)
    a_fi, a_dis = old_get_topk_indices(scores, n_no_sil, spkcache_len)
    b_fi, b_dis = diar._get_topk_indices(scores, n_no_sil)
    assert [int(x) for x in b_fi] == a_fi
    assert [bool(x) for x in b_dis] == a_dis


def test_get_topk_indices_short_flat():
    # Fewer flat scores than spkcache_len: trailing slots keep defaults.
    spkcache_len = 188
    scores = np.random.randn(10, NUM_SPEAKERS).astype(np.float32)
    diar = make_diarizer(spkcache_len=spkcache_len)
    a_fi, a_dis = old_get_topk_indices(scores, 7, spkcache_len)
    b_fi, b_dis = diar._get_topk_indices(scores, 7)
    assert [int(x) for x in b_fi] == a_fi
    assert [bool(x) for x in b_dis] == a_dis


# --------------------------------------------------------------------------
# _gather_spkcache
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(60))
def test_gather_spkcache(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 200))
    spkcache_len = 188
    spkcache = rng.standard_normal((1, n, EMB_DIM)).astype(np.float32)
    spkpreds = rng.standard_normal((1, n, NUM_SPEAKERS)).astype(np.float32)
    msil = rng.standard_normal((1, EMB_DIM)).astype(np.float32)
    indices = [int(rng.integers(0, n + 5)) for _ in range(spkcache_len)]
    is_disabled = [bool(rng.random() < 0.3) for _ in range(spkcache_len)]

    a_e, a_p = old_gather_spkcache(spkcache, spkpreds, msil, indices, is_disabled, spkcache_len)

    diar = make_diarizer(spkcache_len=spkcache_len)
    diar.spkcache = spkcache
    diar.spkcache_preds = spkpreds
    diar.mean_sil_emb = msil
    b_e, b_p = diar._gather_spkcache(indices, is_disabled)

    assert np.array_equal(a_e, b_e)
    assert np.array_equal(a_p, b_p)
