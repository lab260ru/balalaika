"""Fast drop-in replacement for crowd-kit's ROVER aggregation.

crowd-kit's ``ROVER._align`` runs an O(n*m) dynamic program per (task,
hypothesis) pair in pure Python, allocating an options list, tuples and
attr objects for every cell — ~5-15 ms per task at typical 25-40 word
ASR outputs. This module keeps the exact algorithm (same costs, same
tie-breaking, same word-transition-network voting) but runs the DP in a
numba kernel over integer word ids; edge sets are plain ``{word_id:
count}`` dicts and only the O(n+m) traceback/voting stays in Python.

Equivalence with ``crowdkit.aggregation.ROVER`` is pinned by
``tests/test_fast_rover.py`` (randomized + edge-case corpora) and by
``benchmarking/micro/bench_rover.py --impl both``: aggregated strings
are required to match character-for-character.

Tie-breaking details replicated from crowd-kit (these matter):
- DP options are evaluated in order [diagonal, deletion, insertion] and
  ties keep the earliest option (Python ``min`` semantics).
- A deletion against a reference set that already contains the empty
  token costs 0 (the WTN already has a skip edge there).
- Voting picks ``max((count, len(word), word))`` per position; empty
  words are dropped from the final token list.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Hashable, List

import numpy as np
import pandas as pd
from numba import njit

_EMPTY = 0  # word id reserved for the empty token ""

# Action codes written by the DP kernel.
_CORRECT, _SUB, _DEL, _INS = 0, 1, 2, 3


@njit(cache=True)
def _dp_actions(n: int, m: int, match: np.ndarray, has_empty: np.ndarray) -> np.ndarray:
    """Edit-distance DP returning the per-cell chosen action.

    match[i, j] — hyp word i is present in reference edge set j.
    has_empty[j] — reference edge set j contains the empty token.
    """
    dist = np.empty((n + 1, m + 1), dtype=np.int64)
    act = np.empty((n + 1, m + 1), dtype=np.uint8)
    dist[0, 0] = 0
    act[0, 0] = 255
    for j in range(1, m + 1):
        dist[0, j] = j
        act[0, j] = _DEL
    for i in range(1, n + 1):
        dist[i, 0] = i
        act[i, 0] = _INS
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if match[i - 1, j - 1]:
                best = dist[i - 1, j - 1]
                a = _CORRECT
            else:
                best = dist[i - 1, j - 1] + 1
                a = _SUB
            cost_del = dist[i, j - 1] + (0 if has_empty[j - 1] else 1)
            if cost_del < best:
                best = cost_del
                a = _DEL
            cost_ins = dist[i - 1, j] + 1
            if cost_ins < best:
                best = cost_ins
                a = _INS
            dist[i, j] = best
            act[i, j] = a
    return act


def _align(
    ref_edges_sets: List[Dict[int, int]],
    hyp_ids: List[int],
    sources_count: int,
) -> List[Dict[int, int]]:
    """One crowd-kit ``ROVER._align`` pass over integer word ids."""
    n, m = len(hyp_ids), len(ref_edges_sets)
    match = np.empty((n, m), dtype=np.bool_)
    has_empty = np.empty(m, dtype=np.bool_)
    for j, ref in enumerate(ref_edges_sets):
        has_empty[j] = _EMPTY in ref
        for i, word in enumerate(hyp_ids):
            match[i, j] = word in ref
    act = _dp_actions(n, m, match, has_empty)

    alignment: List[Dict[int, int]] = []
    i, j = n, m
    while i != 0 or j != 0:
        a = act[i, j]
        if a == _DEL:
            joined = dict(ref_edges_sets[j - 1])
            joined[_EMPTY] = joined.get(_EMPTY, 0) + 1
            j -= 1
        elif a == _INS:
            word = hyp_ids[i - 1]
            joined = {_EMPTY: sources_count}
            joined[word] = joined.get(word, 0) + 1
            i -= 1
        else:  # _CORRECT / _SUB
            word = hyp_ids[i - 1]
            joined = dict(ref_edges_sets[j - 1])
            joined[word] = joined.get(word, 0) + 1
            i -= 1
            j -= 1
        alignment.append(joined)
    alignment.reverse()
    return alignment


def _aggregate_task(hypotheses: List[List[str]]) -> List[str]:
    """Build the word transition network and vote, like crowd-kit does."""
    words: Dict[str, int] = {"": _EMPTY}
    id_to_word: List[str] = [""]

    def to_ids(tokens: List[str]) -> List[int]:
        ids = []
        for tok in tokens:
            wid = words.get(tok)
            if wid is None:
                wid = len(id_to_word)
                words[tok] = wid
                id_to_word.append(tok)
            ids.append(wid)
        return ids

    first = to_ids(hypotheses[0])
    edges: List[Dict[int, int]] = [{wid: 1} for wid in first]
    for sources_count, hyp in enumerate(hypotheses[1:], start=1):
        edges = _align(edges, to_ids(hyp), sources_count)

    result = []
    for edge_set in edges:
        _, _, value = max(
            (count, len(id_to_word[wid]), id_to_word[wid])
            for wid, count in edge_set.items()
        )
        result.append(value)
    return result


class FastROVER:
    """API-compatible subset of ``crowdkit.aggregation.ROVER``."""

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        detokenizer: Callable[[List[str]], str],
        silent: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        self.silent = silent

    def fit(self, data: pd.DataFrame) -> "FastROVER":
        result: Dict[Hashable, str] = {}
        for task, df in data.groupby("task"):
            hypotheses = [self.tokenizer(text) for text in df["text"]]
            rover_result = _aggregate_task(hypotheses)
            result[task] = self.detokenizer(
                [value for value in rover_result if value != ""]
            )
        texts = pd.Series(result, name="text")
        texts.index.name = "task"
        self.texts_ = texts
        return self

    def fit_predict(self, data: pd.DataFrame) -> "pd.Series[Any]":
        self.fit(data)
        return self.texts_
