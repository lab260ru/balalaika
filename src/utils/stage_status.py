"""Shared utility for writing per-stage status files consumed by base.sh --strict."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def last_line(text: object) -> str:
    """Compact a (possibly multi-line) error string to its last non-empty line.

    Error reasons are ``str(exc)``; some libraries (torchcodec) return a
    ~90-line traceback as the exception message. Stored verbatim these reasons
    accumulate in each stage's ``error_details`` list — and, for transcription,
    in a cross-worker ``mp.Manager().list()`` held in RAM for the whole run — so
    thousands of them balloon memory into a real OOM when a systematic failure
    (e.g. a broken decoder) makes every file error. The full trace is still
    written to the stage log; the status JSON only needs a one-line summary.
    Returns "" for empty input.
    """
    s = str(text)
    lines = [ln for ln in s.splitlines() if ln.strip()]
    return lines[-1] if lines else s.strip()


def write_stage_status(
    stage: int,
    stage_name: str,
    log_dir: str,
    processed: int,
    skipped: int,
    errors: int,
    error_details: list[dict] | None = None,
) -> None:
    """Write stage_N_status.json to log_dir.

    base.sh reads this file after each stage. When --strict is set and
    ``errors > 0``, the pipeline aborts.

    ``error_details`` is capped at 50 entries to keep the file small.
    """
    if error_details is None:
        error_details = []

    status = {
        "stage": stage,
        "stage_name": stage_name,
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "error_details": error_details[:50],
    }

    out_dir = Path(log_dir)
    path = out_dir / f"stage_{stage}_status.json"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        logger.warning("Failed to write stage %d status file: %s", stage, path)
