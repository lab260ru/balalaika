"""Equivalence tests for the stage-8 / ROVER pending scans (chunk-JSON based).

Stage 8 resume (``get_valid_paths``, ``check_consensus``) and the ROVER
pending/record scans read each chunk's ``<stem>.json`` (``asr.<model>`` /
``rover`` keys) through one ``ChunkJsonCache`` (one ``os.scandir`` per directory
+ one JSON parse per existing file). These tests pin the *decisions* against a
straightforward per-file reference oracle across a tree mixing present / empty /
absent fields and missing JSONs, for both ``retry_empty`` settings.

The reference implementations below read the same JSONs the production code does;
they are the oracle the cached code must match.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from loguru import logger

import src.transcription.rover as rover_mod
import src.transcription.transcription as tr
from src.utils.chunk_json import (
    ChunkJsonCache,
    chunk_json_path,
    get_field,
    read_chunk_json,
)
from src.utils.sidecars import DirNameCache, pending

MODELS = ["giga_rnnt", "giga_ctc", "tone", "vosk", "parakeet_v2"]
SUFFIXES = MODELS


def _suffix(model_name: str) -> str:
    return "vosk" if "vosk" in model_name else model_name


# --- reference (per-file) implementations -----------------------------------

def _ref_field_complete(data: dict, dotted: str, *, retry_empty: bool) -> bool:
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    if cur is None:
        return False
    if retry_empty and isinstance(cur, str):
        return cur != ""
    return True


def _ref_check_consensus(audio_path: Path, model_names: List[str],
                         consensus_num: int) -> bool:
    data = read_chunk_json(chunk_json_path(audio_path))
    texts = []
    for mn in model_names:
        t = get_field(data, f"asr.{_suffix(mn)}")
        if t:
            norm = tr.normalize_consensus_text(t)
            if norm:
                texts.append(norm)
    if len(texts) < consensus_num:
        return False
    return max(Counter(texts).values()) >= consensus_num


def _ref_get_valid_paths(audio: List[Path], output_suffix: str,
                         processed: List[str], consensus_num: int,
                         retry_empty: bool) -> List[str]:
    valid = []
    for p in audio:
        data = read_chunk_json(chunk_json_path(p))
        if _ref_field_complete(data, f"asr.{output_suffix}", retry_empty=retry_empty):
            continue
        valid.append(p)
    if consensus_num > 0 and len(processed) >= consensus_num:
        valid = [p for p in valid
                 if not _ref_check_consensus(p, processed, consensus_num)]
    return [str(p) for p in valid]


def _ref_rover_pending(audio, retry_empty: bool, excluded) -> List[str]:
    out = []
    for raw in audio:
        ap = Path(raw)
        if any(pat in ap.stem for pat in excluded):
            continue
        data = read_chunk_json(chunk_json_path(ap))
        if _ref_field_complete(data, "rover", retry_empty=retry_empty):
            continue
        out.append(str(ap))
    return out


def _ref_rover_records(audio, model_names, excluded) -> pd.DataFrame:
    records = []
    for raw in audio:
        ap = Path(raw)
        if any(pat in ap.stem for pat in excluded):
            continue
        data = read_chunk_json(chunk_json_path(ap))
        for mn in model_names:
            t = get_field(data, f"asr.{_suffix(mn)}")
            if not t:
                continue
            records.append({'task': str(ap), 'worker': mn, 'text': str(t).lower()})
    return pd.DataFrame.from_records(records, columns=['task', 'worker', 'text'])


# --- fixture tree -----------------------------------------------------------

def build_tree(root: Path, n_dirs: int = 240, files_per_dir: int = 20) -> List[Path]:
    """A few hundred dirs x ~20 audio files; each chunk's JSON mixes asr/rover
    fields that are present / empty-string / absent, and some chunks have no
    JSON at all (every field absent)."""
    audio: List[Path] = []
    n = 0
    for d_i in range(n_dirs):
        d = root / f"playlist{d_i // 20}" / f"episode{d_i}"
        d.mkdir(parents=True, exist_ok=True)
        for f_i in range(files_per_dir):
            stem = f"chunk_{f_i:03d}"
            ap = d / f"{stem}.wav"
            ap.touch()
            audio.append(ap)

            data: dict = {"asr": {}}
            for s_i, suffix in enumerate(SUFFIXES):
                kind = (n + s_i) % 3
                if kind == 1:
                    data["asr"][suffix] = f"text {n} {suffix}"  # present
                elif kind == 2:
                    data["asr"][suffix] = ""  # empty (retry-empty hazard)
                # kind == 0: absent
            rkind = n % 3
            if rkind == 1:
                data["rover"] = "rover out"
            elif rkind == 2:
                data["rover"] = ""

            if data["asr"] or "rover" in data:
                (d / f"{stem}.json").write_text(json.dumps(data), encoding="utf-8")
            # else: no JSON at all -> all fields read as absent
            n += 1
    return audio


# --- equivalence tests ------------------------------------------------------

@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("scan_tree")
    audio = build_tree(root)
    return root, audio


@pytest.mark.parametrize("retry_empty", [False, True])
def test_get_valid_paths_equivalent(tree, retry_empty, monkeypatch):
    root, audio = tree
    monkeypatch.setattr(tr, "discover_audio_paths",
                        lambda src, config_path=None: [str(p) for p in audio])

    for suffix in SUFFIXES:
        new = tr.get_valid_paths(str(root), suffix, [], 0, retry_empty, None)
        ref = _ref_get_valid_paths(audio, suffix, [], 0, retry_empty)
        assert new == ref, f"valid-set mismatch for {suffix} retry_empty={retry_empty}"


@pytest.mark.parametrize("retry_empty", [False, True])
def test_get_valid_paths_with_consensus_equivalent(tree, retry_empty, monkeypatch):
    root, audio = tree
    monkeypatch.setattr(tr, "discover_audio_paths",
                        lambda src, config_path=None: [str(p) for p in audio])

    processed = MODELS[:3]
    consensus_num = 2
    suffix = MODELS[3]
    new = tr.get_valid_paths(str(root), suffix, processed, consensus_num, retry_empty, None)
    ref = _ref_get_valid_paths(audio, suffix, processed, consensus_num, retry_empty)
    assert new == ref


def test_check_consensus_equivalent(tree):
    root, audio = tree
    cache = ChunkJsonCache()
    for p in audio:
        for cn in (1, 2, 3):
            new = tr.check_consensus(p, MODELS, cn, cache)
            ref = _ref_check_consensus(p, MODELS, cn)
            assert new == ref, f"consensus mismatch {p} cn={cn}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Привет, мир!", "привет мир"),
        ("«Как дела?» — Хорошо…", "как дела хорошо"),
        ("слово,слово; слово", "слово слово слово"),
        ("  МНОГО\tПРОБЕЛОВ\n", "много пробелов"),
        ("по-русски (тест)", "по русски тест"),
    ],
)
def test_normalize_consensus_text(text, expected):
    assert tr.normalize_consensus_text(text) == expected


def test_check_consensus_ignores_case_and_punctuation(tmp_path):
    audio = tmp_path / "chunk.wav"
    audio.touch()
    variants = {
        "giga_rnnt": "Привет, мир!",
        "giga_ctc": "привет мир",
        "tone": "«ПРИВЕТ — МИР...»",
    }
    (tmp_path / "chunk.json").write_text(
        json.dumps({"asr": variants}), encoding="utf-8"
    )
    assert tr.check_consensus(audio, list(variants), 3, ChunkJsonCache())


def test_shared_cache_matches_per_call_cache(tree, monkeypatch):
    """The grouped-decode path reuses one cache across model sweeps; the result
    must match independent per-suffix scans."""
    root, audio = tree
    monkeypatch.setattr(tr, "discover_audio_paths",
                        lambda src, config_path=None: [str(p) for p in audio])
    shared = ChunkJsonCache()
    for suffix in SUFFIXES:
        with_shared = tr.get_valid_paths(str(root), suffix, [], 2, True, None, cache=shared)
        fresh = tr.get_valid_paths(str(root), suffix, [], 2, True, None, cache=None)
        assert with_shared == fresh


@pytest.mark.parametrize("retry_empty", [False, True])
def test_rover_pending_equivalent(tree, retry_empty):
    root, audio = tree
    wrapper = rover_mod.ROVERWrapper(
        podcasts_path=str(root), model_names=MODELS,
        retry_empty_outputs=retry_empty,
    )
    new = wrapper._pending_audio_paths([str(p) for p in audio])
    ref = _ref_rover_pending(audio, retry_empty, wrapper.excluded_patterns)
    assert new == ref


def test_rover_records_equivalent(tree):
    root, audio = tree
    wrapper = rover_mod.ROVERWrapper(podcasts_path=str(root), model_names=MODELS)
    new = wrapper._records_for_audio_paths([str(p) for p in audio])
    ref = _ref_rover_records(audio, MODELS, wrapper.excluded_patterns)
    pd.testing.assert_frame_equal(
        new.reset_index(drop=True), ref.reset_index(drop=True)
    )


# --- DirNameCache unit semantics (retained helper) --------------------------

def test_dirnamecache_size_and_symlinks(tmp_path):
    d = tmp_path / "dir"
    d.mkdir()
    (d / "present.txt").write_text("xyz", encoding="utf-8")  # size 3
    (d / "empty.txt").touch()                                # size 0
    (d / "good_link.txt").symlink_to(d / "present.txt")      # -> size 3
    (d / "empty_link.txt").symlink_to(d / "empty.txt")       # -> size 0
    (d / "dangling.txt").symlink_to(d / "nope.txt")          # missing

    cache = DirNameCache()
    assert cache.exists(d / "present.txt")
    assert cache.exists(d / "empty.txt")
    assert cache.exists(d / "good_link.txt")
    assert cache.exists(d / "empty_link.txt")
    assert not cache.exists(d / "dangling.txt")
    assert not cache.exists(d / "absent.txt")

    assert cache.size(d / "present.txt") == 3
    assert cache.size(d / "empty.txt") == 0
    assert cache.size(d / "good_link.txt") == 3   # resolves through link
    assert cache.size(d / "empty_link.txt") == 0
    assert cache.size(d / "dangling.txt") is None
    assert cache.size(d / "absent.txt") is None

    assert cache.sidecar_complete(d / "present.txt")
    assert cache.sidecar_complete(d / "present.txt", retry_empty=True)
    assert cache.sidecar_complete(d / "empty.txt")                      # retry off -> complete
    assert not cache.sidecar_complete(d / "empty.txt", retry_empty=True)
    assert not cache.sidecar_complete(d / "dangling.txt", retry_empty=True)
    assert cache.sidecar_complete(d / "empty_link.txt")                 # retry off
    assert not cache.sidecar_complete(d / "empty_link.txt", retry_empty=True)


def test_dirnamecache_name_too_long_treated_as_complete(tmp_path):
    """A name exceeding NAME_MAX can never appear in a listing, so it is treated
    as already complete (skip once, forever) and warned about once."""
    d = tmp_path / "dir"
    d.mkdir()
    too_long = d / (("x" * 300) + ".txt")

    sink: List[str] = []
    handle = logger.add(sink.append, level="WARNING", format="{message}")
    try:
        cache = DirNameCache()
        assert cache.exists(too_long)
        assert cache.sidecar_complete(too_long)
        assert cache.sidecar_complete(too_long, retry_empty=True)
        cache.exists(too_long)
        cache.sidecar_complete(too_long, retry_empty=True)
    finally:
        logger.remove(handle)

    warnings = [m for m in sink if "NAME_MAX" in m]
    assert len(warnings) == 1, sink

    (d / "src_a.txt").write_text("a", encoding="utf-8")
    (d / "src_b.txt").write_text("b", encoding="utf-8")
    (d / "out_b.txt").write_text("done", encoding="utf-8")  # b already done
    long_stem = "y" * 300

    def derive(p: Path) -> Path:
        if p.name == "src_a.txt":
            return d / (long_stem + ".txt")   # over-NAME_MAX output
        return d / p.name.replace("src_", "out_")

    todo = pending([d / "src_a.txt", d / "src_b.txt"], derive)
    assert todo == []  # a's output is "complete" (too long), b's output exists


def test_dirnamecache_one_scandir_per_dir(tmp_path, monkeypatch):
    """Existence is one scandir per directory, regardless of how many probes;
    sizes are targeted memoised os.stat, not extra scandirs."""
    d = tmp_path / "dir"
    d.mkdir()
    for i in range(10):
        (d / f"a{i}.txt").write_text("x", encoding="utf-8")

    scandir_calls = {"n": 0}
    stat_calls = {"n": 0}
    real_scandir, real_stat = os.scandir, os.stat

    def counting_scandir(path):
        scandir_calls["n"] += 1
        return real_scandir(path)

    def counting_stat(path, *a, **k):
        stat_calls["n"] += 1
        return real_stat(path, *a, **k)

    monkeypatch.setattr(os, "scandir", counting_scandir)
    monkeypatch.setattr(os, "stat", counting_stat)

    cache = DirNameCache()
    for i in range(10):                # 10 existence probes, all same dir
        cache.exists(d / f"a{i}.txt")
    assert scandir_calls["n"] == 1     # one scandir for the whole directory
    assert stat_calls["n"] == 0        # existence pass does no os.stat here

    cache.size(d / "a0.txt")           # one targeted os.stat
    cache.size(d / "a0.txt")           # memoised, no new stat
    assert stat_calls["n"] == 1
    assert scandir_calls["n"] == 1     # size did NOT trigger another scandir
