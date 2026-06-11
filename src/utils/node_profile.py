"""Consume the per-node batch-size profile written by ``benchmarking/warmup.py``.

Stages call :func:`resolve_batch_size` with whatever the YAML config holds.
Plain integers pass through untouched, so existing configs keep working. The
string ``auto`` (or ``node`` / ``profile``) resolves against
``cache/node_profile.json`` — generate it once per machine with::

    python -m benchmarking.warmup --config_path configs/config.yaml

Override the profile location with the ``BALALAIKA_NODE_PROFILE`` env var.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_PATH = REPO_ROOT / "cache" / "node_profile.json"
_AUTO_VALUES = {"auto", "node", "profile"}

_profile_cache: dict[str, Optional[dict]] = {}


def _load_profile(profile_path: Optional[os.PathLike | str]) -> Optional[dict]:
    path = Path(
        profile_path
        or os.environ.get("BALALAIKA_NODE_PROFILE", "")
        or DEFAULT_PROFILE_PATH
    )
    key = str(path)
    if key in _profile_cache:
        return _profile_cache[key]
    profile: Optional[dict] = None
    try:
        with path.open("r", encoding="utf-8") as f:
            profile = json.load(f)
    except FileNotFoundError:
        logger.warning(
            f"node profile not found at {path} — run "
            "`python -m benchmarking.warmup` to generate it."
        )
    except Exception as exc:
        logger.warning(f"could not read node profile {path}: {exc}")
    _profile_cache[key] = profile
    return profile


def resolve_batch_size(
    model_key: str,
    configured,
    default: int,
    *,
    profile_path: Optional[os.PathLike | str] = None,
) -> int:
    """Resolve a configured batch size, honoring the ``auto`` sentinel.

    Args:
        model_key: profile key, e.g. ``distillmos``, ``antispoofing``,
            ``denoising``, ``music_detect``, ``transcription.giga_ctc``.
        configured: raw config value (int, numeric string, ``auto`` or None).
        default: fallback when nothing else resolves.
    """
    if configured is None:
        return int(default)

    text = str(configured).strip().lower()
    if text not in _AUTO_VALUES:
        try:
            return int(configured)
        except (TypeError, ValueError):
            logger.warning(
                f"invalid batch size {configured!r} for {model_key}; using {default}"
            )
            return int(default)

    profile = _load_profile(profile_path)
    if profile:
        models = profile.get("models", {}) or {}
        entry = models.get(model_key) or {}
        best = entry.get("best_batch_size")
        if best:
            logger.info(f"{model_key}: batch_size=auto -> {best} (node profile)")
            return int(best)
        # e.g. transcription.<model> falls back to the flat recommendation
        rec = (profile.get("recommended_batch_sizes", {}) or {}).get(
            model_key.split(".", 1)[0]
        )
        if rec:
            logger.info(f"{model_key}: batch_size=auto -> {rec} (node recommendation)")
            return int(rec)

    logger.warning(
        f"{model_key}: batch_size=auto but no usable node profile entry; using {default}"
    )
    return int(default)
