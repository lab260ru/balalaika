"""Sidecar-text helpers (``*_rover.txt``, ``*_punct.txt``, …).

Pipeline stages 7-10 (transcription, punctuation, accents, phonemizer) all
work by reading one ``.txt`` file and writing another next to the audio. The
resume / skip-already-done logic for those stages is the same: enumerate
candidates, derive the expected output path, keep only candidates whose
output is missing.

This module replaces the four near-identical ``get_valid_*_paths`` helpers
that lived in each stage with a couple of small composable functions.
"""
from __future__ import annotations

import errno
from pathlib import Path
from typing import Callable, Iterable, List

from loguru import logger

from src.utils.csv_manager import discover_audio_paths
from src.utils.utils import get_audio_paths, get_txt_paths


def path_exists(path: Path, *, missing_on_too_long: bool, label: str = "Sidecar") -> bool:
    try:
        return path.exists()
    except OSError as exc:
        if exc.errno == errno.ENAMETOOLONG:
            logger.error(f"{label} path is too long, skipping: {path}")
            return not missing_on_too_long
        raise


class DirNameCache:
    """Existence checks backed by one scandir per directory.

    Equivalent to ``os.path.exists`` per path but O(#directories) syscalls
    instead of O(#paths) — pending-work scans over millions of sidecars do
    two existence probes per audio file, which this collapses to dictionary
    lookups. Dangling symlinks are verified with a real ``exists`` so they
    still read as missing. (Names longer than NAME_MAX can't appear in a
    directory listing, so the ENAMETOOLONG special case disappears here:
    such outputs simply count as missing and fail loudly at write time
    instead of being silently skipped.)
    """

    def __init__(self) -> None:
        self._names: dict[str, set[str]] = {}

    def _dir_names(self, d: str) -> set[str]:
        import os

        cached = self._names.get(d)
        if cached is not None:
            return cached
        present: set[str] = set()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_symlink():
                            if os.path.exists(entry.path):
                                present.add(entry.name)
                        else:
                            present.add(entry.name)
                    except OSError:
                        continue
        except OSError:
            pass
        self._names[d] = present
        return present

    def exists(self, path: Path | str) -> bool:
        import os

        d, name = os.path.split(str(path))
        return name in self._dir_names(d)


def text_sidecar_complete(path: Path, *, retry_empty: bool = False, label: str = "Sidecar") -> bool:
    """Return whether a text sidecar should be treated as already complete.

    ``retry_empty=True`` makes existing zero-byte ``.txt`` files count as
    missing, which lets transcription rerun interrupted writes.
    """
    try:
        if not path.exists():
            return False
        if retry_empty and path.suffix == ".txt":
            return path.stat().st_size != 0
        return True
    except OSError as exc:
        if exc.errno == errno.ENAMETOOLONG:
            logger.error(f"{label} path is too long, skipping: {path}")
            return True
        raise


def with_suffix_at_stem(path: Path, suffix: str) -> Path:
    """``foo.flac`` + ``"_punct.txt"`` → ``foo_punct.txt`` next to the input."""
    return path.with_name(f"{path.stem}{suffix}")


def replace_in_stem(path: Path, old: str, new: str, *, ext: str = ".txt") -> Path:
    """``foo_punct.txt`` + ``("_punct", "_accent")`` → ``foo_accent.txt``."""
    return path.with_name(path.stem.replace(old, new) + ext)


def pending(
    inputs: Iterable[Path | str],
    derive_output: Callable[[Path], Path],
    *,
    require_input_exists: bool = True,
) -> List[Path]:
    """Return inputs whose ``derive_output(input)`` does not yet exist on disk.

    With ``require_input_exists=True`` (default) inputs that no longer exist
    on disk are excluded too — useful for stages chained off a previous
    stage's sidecar (so a deleted upstream file doesn't reappear as work).
    """
    cache = DirNameCache()
    out: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if require_input_exists and not cache.exists(p):
            continue
        if not cache.exists(derive_output(p)):
            out.append(p)
    return out


def pending_audio_to_sidecar(
    podcasts_path: str | Path,
    *,
    in_suffix: str,
    out_suffix: str,
    config_path: str | Path | None = None,
) -> List[Path]:
    """Audio-rooted scan: keep ``stem+in_suffix`` paths that lack ``stem+out_suffix``.

    Returns the **input sidecar** paths (not the audio paths), matching the
    convention of the original stage helpers.
    """
    audio = (
        discover_audio_paths(podcasts_path, config_path=config_path)
        if config_path
        else get_audio_paths(str(podcasts_path))
    )
    cache = DirNameCache()
    pendings: List[Path] = []
    for a in audio:
        a = Path(a)
        in_path = with_suffix_at_stem(a, in_suffix)
        if not cache.exists(in_path):
            continue
        if cache.exists(with_suffix_at_stem(a, out_suffix)):
            continue
        pendings.append(in_path)
    return pendings


def pending_sidecar_chain(
    podcasts_path: str | Path,
    *,
    in_suffix: str,
    out_derive: Callable[[Path], Path],
    config_path: str | Path | None = None,
) -> List[Path]:
    """Return sidecar inputs whose derived output is missing.

    Without ``config_path`` this scans ``*<in_suffix>`` directly. With
    ``config_path`` it derives sidecars from the configured audio source, so
    stages can use ``runtime.audio_paths_source`` consistently.
    """
    if config_path:
        inputs = [
            with_suffix_at_stem(Path(path), in_suffix)
            for path in discover_audio_paths(podcasts_path, config_path=config_path)
        ]
    else:
        inputs = get_txt_paths(str(podcasts_path), in_suffix)
    return pending(inputs, out_derive)
