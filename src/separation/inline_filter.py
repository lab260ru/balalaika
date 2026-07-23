"""Shared inline-deletion logic for the separation scoring stages (4/5/6/7).

Each scoring stage (``music_detect``, ``distillmos_process``, ``antispoofing``,
``tts_suitability``) writes a quality score to ``balalaika.parquet`` and, by
default, leaves deletion to a dedicated ``.5`` filter stage that re-reads the
state and prunes files. That clean split costs a second full pass over the
dataset tree.

When a stage's ``inline_filter`` flag is on **and** a numeric threshold is
configured, the scoring worker applies the same delete predicate the matching
``.5`` filter uses and removes the file in the *same* pass — no extra IO walk.
The threshold is read from the stage's ``*_filter`` subsection so there is one
source of truth shared by the inline path and the standalone filter stage.

The per-stage delete predicates (identical to the standalone filters):

===========  ===============================  ============================
Stage        Score column(s)                  Delete when
===========  ===============================  ============================
music        ``music_prob``                   ``prob > threshold``
distillmos   ``DistillMOS``                   ``mos  < threshold``
antispoof    ``score_spoof``/``score_bonafide``  ``spoof - bonafide > threshold``
tts          ``p_not_tts``/``p_tts``          ``p_not_tts - p_tts > threshold``
===========  ===============================  ============================

This module is intentionally free of torch / onnxruntime / IO-heavy imports so
the predicate and threshold-resolution logic can be covered by fast CPU tests
(``tests/test_inline_filter.py``) even though the GPU scoring loops cannot.
"""
from __future__ import annotations

import os
from typing import Dict, Mapping, Optional, Tuple

from loguru import logger

# Stage identifiers (match the ``inline_filter`` config flag owner).
MUSIC = "music"
DISTILLMOS = "distillmos"
ANTISPOOF = "antispoof"
TTS = "tts"

# Extra partial-CSV columns the scoring stages carry only when inline deletion
# is active, so a stopped run resumes and ``audit_from_filter_partials`` can
# reproduce the dropped hours exactly as the standalone filters do.
INLINE_PARTIAL_FIELDS: Tuple[str, ...] = ("total_duration", "duration_s", "deleted")


def resolve_inline(
    stage_cfg: Mapping, filter_cfg: Mapping
) -> Optional[float]:
    """Return the inline-delete threshold, or ``None`` when inline is disabled.

    Inline deletion is active only when the scoring stage's ``inline_filter``
    flag is truthy **and** the matching ``*_filter`` subsection carries a
    numeric ``threshold``. A null/missing threshold disables inline deletion
    (with a warning) rather than deleting by an undefined cutoff.
    """
    if not bool(stage_cfg.get("inline_filter", False)):
        return None
    threshold = filter_cfg.get("threshold")
    if threshold is None:
        logger.warning(
            "inline_filter is on but no numeric threshold is configured in the "
            "matching *_filter subsection; inline deletion is DISABLED. Set the "
            "filter threshold or run the standalone .5 filter stage instead."
        )
        return None
    try:
        return float(threshold)
    except (TypeError, ValueError):
        logger.warning(
            f"inline_filter threshold {threshold!r} is not a number; inline "
            "deletion is DISABLED."
        )
        return None


def should_delete(stage: str, scores: Mapping[str, float], threshold: float) -> bool:
    """Apply ``stage``'s delete predicate to ``scores`` at ``threshold``.

    ``scores`` holds the keys the stage produced (e.g. ``{"music_prob": 0.7}``
    or ``{"score_spoof": .., "score_bonafide": ..}``). The boundary is exclusive
    on every stage, so a value exactly equal to the threshold is **kept** —
    matching the ``< threshold`` / ``> threshold`` comparisons in the standalone
    filters.
    """
    if stage == MUSIC:
        return float(scores["music_prob"]) > threshold
    if stage == DISTILLMOS:
        return float(scores["DistillMOS"]) < threshold
    if stage == ANTISPOOF:
        return float(scores["score_spoof"]) - float(scores["score_bonafide"]) > threshold
    if stage == TTS:
        return float(scores["p_not_tts"]) - float(scores["p_tts"]) > threshold
    raise ValueError(f"Unknown inline-filter stage: {stage!r}")


def delete_and_measure(
    path: str, duration_hint: float = 0.0
) -> Tuple[bool, float]:
    """Delete ``path`` and return ``(deleted, duration_s)``.

    ``duration_hint`` is the duration already known for this file (carried in a
    shard annotation or the state). The audio duration is probed with
    :func:`src.utils.audit.safe_audio_duration` **only** when no usable hint is
    available, so just the deleted files — never the kept majority — pay a seek,
    and the probe happens before ``os.remove`` so a candidate with no stored
    duration still records one. A missing file counts as deleted (idempotent
    resume); other ``OSError``s are surfaced to the caller as ``(False, dur)``.
    """
    duration_s = float(duration_hint) if duration_hint and duration_hint > 0 else 0.0
    if duration_s <= 0:
        from src.utils.audit import safe_audio_duration

        try:
            duration_s = float(safe_audio_duration(path))
        except Exception:
            duration_s = 0.0

    try:
        os.remove(path)
        return True, duration_s
    except FileNotFoundError:
        return True, duration_s
    except OSError as exc:
        logger.warning(f"Could not delete {path}: {exc}")
        return False, duration_s


def inline_row(
    resolved_path: str, scores: Dict[str, float], deleted: bool, duration_s: float
) -> Dict[str, object]:
    """Build the partial-CSV row shared by the inline-deletion scoring stages.

    Combines the stage's score columns with the ``total_duration`` /
    ``duration_s`` / ``deleted`` audit columns (:data:`INLINE_PARTIAL_FIELDS`).
    """
    row: Dict[str, object] = {"filepath": resolved_path}
    row.update(scores)
    row["total_duration"] = round(duration_s, 4)
    row["duration_s"] = round(duration_s, 4)
    row["deleted"] = deleted
    return row


def write_score_row(
    writer,
    *,
    stage: str,
    resolved_path: str,
    audio_path: str,
    scores: Dict[str, float],
    inline_threshold: Optional[float],
    audio_lengths: Optional[Mapping[str, float]],
    errors_counter,
) -> None:
    """Write one scoring-stage partial row, deleting in-pass when inline is on.

    The unified worker write for stages 5/6/7 (stage 4 inlines the same logic
    directly): when ``inline_threshold`` is ``None`` it writes the plain
    score-only row exactly as before; otherwise it applies the stage predicate,
    deletes matches, and writes the extended audit row (:func:`inline_row`).
    ``audio_lengths`` supplies the duration carried in the shard annotation so
    kept rows record their hours without a probe; only deleted files lacking a
    known duration pay a seek.
    """
    if inline_threshold is None:
        row: Dict[str, object] = {"filepath": resolved_path}
        row.update(scores)
        writer.write(row)
        return

    duration_s = (
        float(audio_lengths.get(str(audio_path), 0.0)) if audio_lengths else 0.0
    )
    deleted = False
    if should_delete(stage, scores, inline_threshold):
        deleted, duration_s = delete_and_measure(audio_path, duration_s)
        if not deleted:
            errors_counter.value += 1
    writer.write(inline_row(resolved_path, scores, deleted, duration_s))
