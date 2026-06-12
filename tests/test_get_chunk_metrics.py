"""Exact-equality tests for the searchsorted get_chunk_metrics.

Pins the new ``SegmentIndex`` / ``seg_index`` fast path against a VERBATIM copy
of the original O(segments) per-chunk scan over randomized sorted segment sets,
including edge cases (empty, single segment, zero/negative chunk, full overlap,
ties on boundaries).
"""

import random

import numpy as np
import pytest

from src.preprocess.preprocess import SegmentIndex, get_chunk_metrics


def old_get_chunk_metrics(c_start, c_end, raw_segments):
    chunk_dur = c_end - c_start
    if chunk_dur <= 0:
        return 0.0, 0.0, 0
    intervals = []
    speakers_in_chunk = set()
    for rs, re_, spk in raw_segments:
        overlap_s = max(c_start, rs)
        overlap_e = min(c_end, re_)
        if overlap_s < overlap_e:
            intervals.append([overlap_s, overlap_e])
            speakers_in_chunk.add(spk)
    intervals.sort(key=lambda x: x[0])
    if not intervals:
        return 100.0, round(chunk_dur, 2), 0
    merged_speech = []
    for interval in intervals:
        if not merged_speech:
            merged_speech.append(interval)
        else:
            prev = merged_speech[-1]
            if interval[0] <= prev[1]:
                prev[1] = max(prev[1], interval[1])
            else:
                merged_speech.append(interval)
    speech_dur = sum(e - s for s, e in merged_speech)
    silence_dur = max(0.0, chunk_dur - speech_dur)
    silence_pct = (silence_dur / chunk_dur) * 100
    gaps = [merged_speech[0][0] - c_start]
    for i in range(len(merged_speech) - 1):
        gaps.append(merged_speech[i + 1][0] - merged_speech[i][1])
    gaps.append(c_end - merged_speech[-1][1])
    max_gap = max(gaps)
    return round(silence_pct, 2), round(max_gap, 2), len(speakers_in_chunk)


def _random_sorted_segments(rng, n):
    segs = []
    t = 0.0
    for _ in range(n):
        s = t + rng.uniform(0, 2)
        e = s + rng.uniform(0.01, 3)
        segs.append((s, e, rng.randint(0, 3)))
        t = s + rng.uniform(-0.5, 1.5)
    return sorted(segs, key=lambda x: x[0])


@pytest.mark.parametrize("seed", range(400))
def test_searchsorted_matches_full_scan(seed):
    rng = random.Random(seed)
    segs = _random_sorted_segments(rng, rng.randint(0, 200))
    idx = SegmentIndex(segs)
    span = (segs[-1][1] if segs else 5.0)
    # Several queries against the same index (mirrors many chunks per episode).
    for _ in range(5):
        c_start = rng.uniform(-1.0, span + 1.0)
        c_end = c_start + rng.uniform(0.0, 6.0)
        expected = old_get_chunk_metrics(c_start, c_end, segs)
        got = get_chunk_metrics(c_start, c_end, segs, seg_index=idx)
        assert got == expected, (c_start, c_end)


def test_empty_segments():
    idx = SegmentIndex([])
    assert get_chunk_metrics(0.0, 5.0, [], seg_index=idx) == old_get_chunk_metrics(0.0, 5.0, [])
    assert idx.candidate_range(0.0, 5.0) == range(0, 0)


def test_zero_and_negative_chunk():
    segs = [(0.0, 1.0, 0), (2.0, 3.0, 1)]
    idx = SegmentIndex(segs)
    assert get_chunk_metrics(2.0, 2.0, segs, seg_index=idx) == (0.0, 0.0, 0)
    assert get_chunk_metrics(3.0, 1.0, segs, seg_index=idx) == (0.0, 0.0, 0)


def test_full_overlap_single_speaker():
    segs = [(0.0, 10.0, 2)]
    idx = SegmentIndex(segs)
    assert get_chunk_metrics(0.0, 10.0, segs, seg_index=idx) == old_get_chunk_metrics(
        0.0, 10.0, segs
    )


def test_boundary_ties():
    # chunk edges exactly on segment edges (overlap is strict <, so touching
    # endpoints contribute nothing).
    segs = [(0.0, 2.0, 0), (2.0, 4.0, 1), (4.0, 6.0, 0)]
    idx = SegmentIndex(segs)
    for c_start, c_end in [(2.0, 4.0), (0.0, 2.0), (1.0, 5.0), (2.0, 2.0)]:
        assert get_chunk_metrics(c_start, c_end, segs, seg_index=idx) == old_get_chunk_metrics(
            c_start, c_end, segs
        )


def test_default_path_unchanged():
    # Without seg_index, behavior must equal the verbatim original.
    rng = random.Random(99)
    segs = _random_sorted_segments(rng, 50)
    assert get_chunk_metrics(1.0, 8.0, segs) == old_get_chunk_metrics(1.0, 8.0, segs)
