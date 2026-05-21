"""Sidecar-text helpers (``*_rover.txt``, ``*_punct.txt``, …).

Pipeline stages 6-9 (transcription, punctuation, accents, phonemizer) all
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

from src.utils.utils import get_audio_paths, get_txt_paths


def path_exists(path: Path, *, missing_on_too_long: bool, label: str = "Sidecar") -> bool:
    try:
        return path.exists()
    except OSError as exc:
        if exc.errno == errno.ENAMETOOLONG:
            logger.error(f"{label} path is too long, skipping: {path}")
            return not missing_on_too_long
        raise


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
    out: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if require_input_exists and not path_exists(p, missing_on_too_long=True):
            continue
        if not path_exists(derive_output(p), missing_on_too_long=False):
            out.append(p)
    return out


def pending_audio_to_sidecar(
    podcasts_path: str | Path,
    *,
    in_suffix: str,
    out_suffix: str,
) -> List[Path]:
    """Audio-rooted scan: keep ``stem+in_suffix`` paths that lack ``stem+out_suffix``.

    Returns the **input sidecar** paths (not the audio paths), matching the
    convention of the original stage helpers.
    """
    audio = get_audio_paths(str(podcasts_path))
    pendings: List[Path] = []
    for a in audio:
        a = Path(a)
        in_path = with_suffix_at_stem(a, in_suffix)
        if not path_exists(in_path, missing_on_too_long=True):
            continue
        if path_exists(with_suffix_at_stem(a, out_suffix), missing_on_too_long=False):
            continue
        pendings.append(in_path)
    return pendings


def pending_sidecar_chain(
    podcasts_path: str | Path,
    *,
    in_suffix: str,
    out_derive: Callable[[Path], Path],
) -> List[Path]:
    """Sidecar-rooted scan: enumerate ``*<in_suffix>`` and keep ones missing output.

    ``out_derive(input_sidecar)`` returns the expected output path. This is
    the path-style equivalent of :func:`pending_audio_to_sidecar` but driven
    from the input ``.txt`` files directly (handy when the audio file may
    already have been deleted by an earlier filter stage).
    """
    inputs = get_txt_paths(str(podcasts_path), in_suffix)
    return pending(inputs, out_derive)
