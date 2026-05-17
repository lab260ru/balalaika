"""Shared utility for writing per-stage status files consumed by base.sh --strict."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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
