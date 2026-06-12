"""Tests for bug fixes in src.utils.parallel and src.utils.utils.

These cover:
* run_per_gpu_pool return arity / contract (always a 2-tuple).
* run_per_gpu_pool error attribution (each failure carries its own item,
  not the last-submitted loop variable).
* process_token fallback for unrecognized labels (returns the token
  unchanged instead of None, so " ".join(...) downstream cannot crash).

All run on CPU only — they never allocate GPU memory. run_per_gpu_pool is
exercised with explicit ``gpu_ids`` (so it never calls
``torch.cuda.device_count()``) and a trivial CPU initializer/work_fn. The
default multiprocessing start method on this host is ``fork``, so the
module-level worker functions below are inherited by the pool workers without
pickling.

Run: .dev_venv/bin/python -m pytest tests/test_parallel_and_tokens.py -q
"""
from __future__ import annotations

import pytest

from src.utils.parallel import (
    _chunk_list,
    run_per_gpu_pool,
    run_per_gpu_pool_chunked,
)
from src.utils.utils import process_token


# ---------------------------------------------------------------------------
# Module-level helpers for run_per_gpu_pool (must be importable by pool workers)
# ---------------------------------------------------------------------------

def _noop_init() -> None:
    """Trivial initializer — touches no GPU."""
    return None


def _work_fail_on_marked(item):
    """Raise iff the item string contains 'FAIL', echoing the item in the message."""
    if "FAIL" in str(item):
        raise ValueError(f"boom:{item}")
    return item


def _chunk_work_fail_on_marked(chunk):
    """Chunk work_fn: return one {item, reason} per item containing 'FAIL'."""
    failures = []
    for item in chunk:
        if "FAIL" in str(item):
            failures.append({"item": item, "reason": f"boom:{item}"})
    return failures


def _chunk_work_crash(chunk):
    """Chunk work_fn that crashes the whole slab."""
    raise RuntimeError("slab exploded")


# ---------------------------------------------------------------------------
# run_per_gpu_pool — return contract
# ---------------------------------------------------------------------------

class TestRunPerGpuPoolContract:
    def test_empty_items_returns_two_tuple(self):
        """Bug #1/#3: empty-items path must return a 2-tuple, not a 3-tuple."""
        result = run_per_gpu_pool(
            [],
            work_fn=_work_fail_on_marked,
            initializer=_noop_init,
            init_args_factory=lambda gpu_id: (),
            num_workers_per_gpu=1,
            gpu_ids=[0],
            desc="empty",
        )
        assert result == (0, [])
        # Unpacking as a 2-tuple (as every caller does) must succeed.
        error_count, error_details = result
        assert error_count == 0
        assert error_details == []

    def test_all_success_returns_no_errors(self):
        error_count, error_details = run_per_gpu_pool(
            ["a", "b", "c"],
            work_fn=_work_fail_on_marked,
            initializer=_noop_init,
            init_args_factory=lambda gpu_id: (),
            num_workers_per_gpu=2,
            gpu_ids=[0],
            desc="ok",
        )
        assert error_count == 0
        assert error_details == []

    def test_no_gpu_ids_raises(self):
        with pytest.raises(RuntimeError):
            run_per_gpu_pool(
                ["a"],
                work_fn=_work_fail_on_marked,
                initializer=_noop_init,
                init_args_factory=lambda gpu_id: (),
                gpu_ids=[],
            )


# ---------------------------------------------------------------------------
# run_per_gpu_pool — error attribution (bug #2)
# ---------------------------------------------------------------------------

class TestRunPerGpuPoolErrorAttribution:
    def test_errors_carry_their_own_item(self):
        """Bug #2: each error_details entry must reference the failing item,
        not the last-submitted loop variable.

        Items are ordered so that the failing items are NOT last; the old
        code would have attributed every failure to the last submitted item.
        """
        items = ["ok-1", "FAIL-2", "ok-3", "FAIL-4", "ok-5"]
        error_count, error_details = run_per_gpu_pool(
            items,
            work_fn=_work_fail_on_marked,
            # One worker, one GPU slot => deterministic single shard; the
            # last *submitted* item is "ok-5", which never fails.
            initializer=_noop_init,
            init_args_factory=lambda gpu_id: (),
            num_workers_per_gpu=1,
            gpu_ids=[0],
            desc="attrib",
        )
        assert error_count == 2
        failed_items = {d["item"] for d in error_details}
        assert failed_items == {"FAIL-2", "FAIL-4"}
        # Sanity: no failure was misattributed to the last-submitted item.
        assert "ok-5" not in failed_items
        # Each reason references its own item, proving the mapping is correct.
        for d in error_details:
            assert d["item"] in d["reason"]


# ---------------------------------------------------------------------------
# run_per_gpu_pool_chunked — slab submit (accents/phonemizer item 5)
# ---------------------------------------------------------------------------

class TestChunkList:
    def test_chunk_sizes_and_coverage(self):
        items = list(range(10))
        chunks = _chunk_list(items, 3)
        assert chunks == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        # every item present exactly once, order preserved
        assert [x for c in chunks for x in c] == items

    def test_chunk_size_floor_is_one(self):
        assert _chunk_list([1, 2, 3], 0) == [[1], [2], [3]]


class TestRunPerGpuPoolChunked:
    def test_empty_items_returns_two_tuple(self):
        result = run_per_gpu_pool_chunked(
            [],
            work_fn=_chunk_work_fail_on_marked,
            initializer=_noop_init,
            init_args_factory=lambda gpu_id: (),
            chunk_size=4,
            gpu_ids=[0],
            desc="empty",
        )
        assert result == (0, [])

    def test_all_success(self):
        error_count, error_details = run_per_gpu_pool_chunked(
            ["a", "b", "c", "d", "e"],
            work_fn=_chunk_work_fail_on_marked,
            initializer=_noop_init,
            init_args_factory=lambda gpu_id: (),
            chunk_size=2,
            num_workers_per_gpu=2,
            gpu_ids=[0],
            desc="ok",
        )
        assert error_count == 0
        assert error_details == []

    def test_per_item_failures_inside_a_slab(self):
        """A single bad file is reported; its slab-mates are NOT counted."""
        items = ["ok-1", "FAIL-2", "ok-3", "FAIL-4", "ok-5", "ok-6"]
        error_count, error_details = run_per_gpu_pool_chunked(
            items,
            work_fn=_chunk_work_fail_on_marked,
            initializer=_noop_init,
            init_args_factory=lambda gpu_id: (),
            chunk_size=3,
            num_workers_per_gpu=1,
            gpu_ids=[0],
            desc="attrib",
        )
        assert error_count == 2
        failed = {d["item"] for d in error_details}
        assert failed == {"FAIL-2", "FAIL-4"}
        for d in error_details:
            assert d["item"] in d["reason"]

    def test_slab_crash_attributes_all_its_items(self):
        items = ["a", "b", "c", "d"]
        error_count, error_details = run_per_gpu_pool_chunked(
            items,
            work_fn=_chunk_work_crash,
            initializer=_noop_init,
            init_args_factory=lambda gpu_id: (),
            chunk_size=4,  # single slab => all 4 items attributed the crash
            num_workers_per_gpu=1,
            gpu_ids=[0],
            desc="crash",
        )
        assert error_count == 4
        assert {d["item"] for d in error_details} == {"a", "b", "c", "d"}

    def test_no_gpu_ids_raises(self):
        with pytest.raises(RuntimeError):
            run_per_gpu_pool_chunked(
                ["a"],
                work_fn=_chunk_work_fail_on_marked,
                initializer=_noop_init,
                init_args_factory=lambda gpu_id: (),
                chunk_size=4,
                gpu_ids=[],
            )


# ---------------------------------------------------------------------------
# process_token — fallback for unknown labels (bug #5)
# ---------------------------------------------------------------------------

class TestProcessTokenFallback:
    def test_unknown_label_returns_token_unchanged(self):
        assert process_token("слово", "TOTALLY_UNKNOWN_LABEL") == "слово"

    def test_unknown_label_never_returns_none(self):
        # The punctuation stage does " ".join(process_token(...) for ...),
        # so a None would raise TypeError. Guard against regression.
        assert process_token("x", "NOPE") is not None

    def test_known_labels_still_work(self):
        assert process_token("word", "LOWER_O") == "word"
        assert process_token("word", "LOWER_PERIOD") == "word."
        assert process_token("word", "UPPER_O") == "Word"
        assert process_token("word", "UPPER_TOTAL_O") == "WORD"

    def test_join_does_not_crash_with_unknown_label(self):
        preds = [
            ("hello", "LOWER_O"),
            ("world", "UPPER_PERIOD"),
            ("oops", "MYSTERY_LABEL"),
        ]
        out = " ".join(process_token(t, l) for t, l in preds)
        assert out == "hello World. oops"
