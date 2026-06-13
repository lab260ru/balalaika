"""Equivalence + syscall-count tests for the stage-7 / ROVER pending scans.

The transcription resume scans (``get_valid_paths``, ``check_consensus``) and
the ROVER pending/record scans used to do one or two ``stat``/``exists`` calls
per audio file (times #models). They now route every existence/size probe
through a single ``DirNameCache`` (one ``os.scandir`` per directory). These
tests pin that the *decisions* are byte-identical to the old per-file logic
across a tree mixing present / absent / zero-byte / dangling-symlink sidecars,
for both ``retry_empty`` settings.

The reference implementations below are verbatim copies of the pre-cache
logic; they are the oracle the cached code must match.
"""
from __future__ import annotations

import errno
import os
from collections import Counter
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pytest

from loguru import logger

import src.transcription.rover as rover_mod
import src.transcription.transcription as tr
from src.utils.sidecars import DirNameCache, pending

MODELS = ["giga_rnnt", "giga_ctc", "tone", "vosk", "parakeet_v2"]
SUFFIXES = ["giga_rnnt", "giga_ctc", "tone", "vosk", "parakeet_v2"]


# --- reference (old, per-file) implementations ------------------------------

def _ref_path_exists(path: Path, *, missing_on_too_long: bool) -> bool:
    try:
        return path.exists()
    except OSError as exc:
        if exc.errno == errno.ENAMETOOLONG:
            return not missing_on_too_long
        raise


def _ref_text_sidecar_complete(path: Path, *, retry_empty: bool = False) -> bool:
    try:
        if not path.exists():
            return False
        if retry_empty and path.suffix == ".txt":
            return path.stat().st_size != 0
        return True
    except OSError as exc:
        if exc.errno == errno.ENAMETOOLONG:
            return True
        raise


def _ref_check_consensus(audio_path: Path, model_names: List[str], consensus_num: int,
                         read) -> bool:
    texts = []
    for mn in model_names:
        suffix = 'vosk' if 'vosk' in mn else mn
        tp = audio_path.with_name(f"{audio_path.stem}_{suffix}.txt")
        if _ref_text_sidecar_complete(tp):
            try:
                t = read(tp)
                if t:
                    texts.append(t.lower().strip())
            except Exception:
                pass
    if len(texts) < consensus_num:
        return False
    return max(Counter(texts).values()) >= consensus_num


def _ref_get_valid_paths(all_paths: List[Path], output_suffix: str,
                         processed: List[str], consensus_num: int,
                         retry_empty_outputs: bool, read) -> List[str]:
    if not all_paths:
        return []
    valid = []
    for p in all_paths:
        sidecar = p.with_name(f"{p.stem}_{output_suffix}.txt")
        if _ref_text_sidecar_complete(sidecar, retry_empty=retry_empty_outputs):
            continue
        valid.append(p)
    if consensus_num > 0 and len(processed) >= consensus_num:
        valid = [p for p in valid
                 if not _ref_check_consensus(p, processed, consensus_num, read)]
    return [str(p) for p in valid]


def _ref_rover_pending(audio_paths, retry_empty: bool, excluded) -> List[str]:
    pending = []
    for raw in audio_paths:
        ap = Path(raw)
        if any(pat in ap.stem for pat in excluded):
            continue
        rover_out = ap.with_name(f"{ap.stem}_rover.txt")
        if _ref_text_sidecar_complete(rover_out, retry_empty=retry_empty):
            continue
        pending.append(str(ap))
    return pending


def _ref_rover_records(audio_paths, model_names, excluded, read) -> pd.DataFrame:
    records = []
    for raw in audio_paths:
        ap = Path(raw)
        if any(pat in ap.stem for pat in excluded):
            continue
        for mn in model_names:
            suffix = 'vosk' if 'vosk' in mn else mn
            tp = ap.with_name(f"{ap.stem}_{suffix}.txt")
            if not _ref_path_exists(tp, missing_on_too_long=True):
                continue
            try:
                text = read(tp)
                if not text:
                    continue
                records.append({'task': str(ap), 'worker': mn, 'text': text.lower()})
            except Exception:
                pass
    return pd.DataFrame.from_records(records, columns=['task', 'worker', 'text'])


# --- fixture tree -----------------------------------------------------------

def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def build_tree(root: Path, n_dirs: int = 240, files_per_dir: int = 20) -> List[Path]:
    """A few hundred dirs x ~20 audio files, sidecars mixed present / absent /
    zero-byte / dangling-symlink across the 5 model suffixes plus _rover."""
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
            for s_i, suffix in enumerate(SUFFIXES):
                sc = d / f"{stem}_{suffix}.txt"
                kind = (n + s_i) % 4
                if kind == 0:
                    pass  # absent
                elif kind == 1:
                    _write(sc, f"text {n} {suffix}")  # present, non-empty
                elif kind == 2:
                    sc.touch()  # present, zero-byte
                elif kind == 3:
                    # dangling symlink -> non-existent target
                    sc.symlink_to(d / f"{stem}_{suffix}.missing")
            # _rover.txt: vary present / absent / empty / dangling too
            rkind = n % 4
            rsc = d / f"{stem}_rover.txt"
            if rkind == 1:
                _write(rsc, "rover out")
            elif rkind == 2:
                rsc.touch()
            elif rkind == 3:
                rsc.symlink_to(d / f"{stem}_rover.missing")
            n += 1
    return audio


def _reader(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


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
    monkeypatch.setattr(tr, "read_file_content", _reader)

    for suffix in SUFFIXES:
        new = tr.get_valid_paths(str(root), suffix, [], 0, retry_empty, None)
        ref = _ref_get_valid_paths(audio, suffix, [], 0, retry_empty, _reader)
        assert new == ref, f"valid-set mismatch for {suffix} retry_empty={retry_empty}"


@pytest.mark.parametrize("retry_empty", [False, True])
def test_get_valid_paths_with_consensus_equivalent(tree, retry_empty, monkeypatch):
    root, audio = tree
    monkeypatch.setattr(tr, "discover_audio_paths",
                        lambda src, config_path=None: [str(p) for p in audio])
    monkeypatch.setattr(tr, "read_file_content", _reader)

    # Mimic the sequential flow: model idx=3 has 3 earlier processed models.
    processed = MODELS[:3]
    consensus_num = 2
    suffix = MODELS[3]
    new = tr.get_valid_paths(str(root), suffix, processed, consensus_num, retry_empty, None)
    ref = _ref_get_valid_paths(audio, suffix, processed, consensus_num, retry_empty, _reader)
    assert new == ref


def test_check_consensus_equivalent(tree, monkeypatch):
    root, audio = tree
    monkeypatch.setattr(tr, "read_file_content", _reader)
    cache = DirNameCache()
    for p in audio:
        for cn in (1, 2, 3):
            new = tr.check_consensus(p, MODELS, cn, cache)
            ref = _ref_check_consensus(p, MODELS, cn, _reader)
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


def test_check_consensus_ignores_case_and_punctuation(tmp_path, monkeypatch):
    audio = tmp_path / "chunk.wav"
    audio.touch()
    variants = {
        "giga_rnnt": "Привет, мир!",
        "giga_ctc": "привет мир",
        "tone": "«ПРИВЕТ — МИР...»",
    }
    for model, text in variants.items():
        (tmp_path / f"chunk_{model}.txt").write_text(text, encoding="utf-8")

    monkeypatch.setattr(tr, "read_file_content", _reader)
    assert tr.check_consensus(audio, list(variants), 3, DirNameCache())


def test_shared_cache_matches_per_call_cache(tree, monkeypatch):
    """The grouped-decode path reuses one cache across suffix sweeps; the
    result must match independent per-suffix scans."""
    root, audio = tree
    monkeypatch.setattr(tr, "discover_audio_paths",
                        lambda src, config_path=None: [str(p) for p in audio])
    shared = DirNameCache()
    for suffix in SUFFIXES:
        with_shared = tr.get_valid_paths(str(root), suffix, [], 2, True, None, cache=shared)
        fresh = tr.get_valid_paths(str(root), suffix, [], 2, True, None, cache=None)
        assert with_shared == fresh


@pytest.mark.parametrize("retry_empty", [False, True])
def test_rover_pending_equivalent(tree, retry_empty, monkeypatch):
    root, audio = tree
    wrapper = rover_mod.ROVERWrapper(
        podcasts_path=str(root), model_names=MODELS,
        retry_empty_outputs=retry_empty,
    )
    new = wrapper._pending_audio_paths([str(p) for p in audio])
    ref = _ref_rover_pending(audio, retry_empty, wrapper.excluded_patterns)
    assert new == ref


def test_rover_records_equivalent(tree, monkeypatch):
    root, audio = tree
    monkeypatch.setattr(rover_mod, "read_file_content", _reader)
    wrapper = rover_mod.ROVERWrapper(podcasts_path=str(root), model_names=MODELS)
    new = wrapper._records_for_audio_paths([str(p) for p in audio])
    ref = _ref_rover_records(audio, MODELS, wrapper.excluded_patterns, _reader)
    # Same rows, same order (both iterate audio then models in order).
    pd.testing.assert_frame_equal(
        new.reset_index(drop=True), ref.reset_index(drop=True)
    )


# --- DirNameCache unit semantics --------------------------------------------

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

    # sidecar_complete mirrors text_sidecar_complete
    assert cache.sidecar_complete(d / "present.txt")
    assert cache.sidecar_complete(d / "present.txt", retry_empty=True)
    assert cache.sidecar_complete(d / "empty.txt")                      # retry off -> complete
    assert not cache.sidecar_complete(d / "empty.txt", retry_empty=True)
    assert not cache.sidecar_complete(d / "dangling.txt", retry_empty=True)
    assert cache.sidecar_complete(d / "empty_link.txt")                 # retry off
    assert not cache.sidecar_complete(d / "empty_link.txt", retry_empty=True)


def test_dirnamecache_name_too_long_treated_as_complete(tmp_path):
    """A sidecar whose name exceeds NAME_MAX can never appear in a directory
    listing, so the old per-file helpers caught the ENAMETOOLONG OSError and
    treated such outputs as already complete (skip once, forever). Restore
    that: ``exists`` / ``sidecar_complete`` report True and ``pending``
    excludes the file, so stages 7-10 don't re-run the model + crash on the
    even-longer ``.tmp`` write every run. Normal names are unaffected.
    """
    d = tmp_path / "dir"
    d.mkdir()
    # ~300 bytes, comfortably over the 255-byte NAME_MAX on typical filesystems.
    too_long = d / (("x" * 300) + ".txt")

    sink: List[str] = []
    handle = logger.add(sink.append, level="WARNING", format="{message}")
    try:
        cache = DirNameCache()
        assert cache.exists(too_long)
        assert cache.sidecar_complete(too_long)
        assert cache.sidecar_complete(too_long, retry_empty=True)
        # Re-probing the same offending name does not re-log the warning.
        cache.exists(too_long)
        cache.sidecar_complete(too_long, retry_empty=True)
    finally:
        logger.remove(handle)

    warnings = [m for m in sink if "NAME_MAX" in m]
    assert len(warnings) == 1, sink

    # The pending scan must exclude the over-long output (no reprocess loop)
    # while still keeping a genuinely missing normal sidecar.
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

    # DirNameCache does `import os; os.scandir(...)` / `os.stat(...)`.
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
