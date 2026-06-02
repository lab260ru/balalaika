"""Centralized loguru configuration: stderr + per-stage timestamped log file.

Every pipeline stage calls :func:`setup_logging` once at process start. The
function wipes loguru's default sinks, attaches a colored stderr sink and a
plain UTF-8 file sink under a configurable directory. The file sink rotates at
200 MB and keeps the 10 most recent files per stage so logs never grow without
bound on long runs.

Resolution order for the log directory:
  1. Explicit ``log_dir`` argument.
  2. ``BALALAIKA_LOG_DIR`` environment variable.
  3. ``./logs`` relative to the current working directory.

The function returns the absolute path of the file sink so callers can surface
it to the user (e.g. ``logger.info(f"Logs: {log_path}")``).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

_FORMAT_CONSOLE = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
_FORMAT_FILE = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{process.id}:{thread.id} | {name}:{function}:{line} - {message}"
)


def _resolve_log_dir(log_dir: Optional[str]) -> Path:
    if log_dir is None:
        log_dir = os.environ.get("BALALAIKA_LOG_DIR", "./logs")
    return Path(log_dir).expanduser().resolve()


def setup_logging(
    stage: str,
    log_dir: Optional[str] = None,
    level: Optional[str] = None,
    capture_warnings: bool = True,
) -> Path:
    """Configure loguru sinks for a pipeline stage.

    Args:
        stage: Short identifier used as the log filename prefix
            (e.g. ``"preprocess"``, ``"music_detect"``).
        log_dir: Override for the log directory. ``None`` falls back to the
            ``BALALAIKA_LOG_DIR`` env var, then to ``./logs``.
        level: Minimum level for both sinks.
        capture_warnings: When True, also routes Python's ``warnings`` module
            and standard ``logging`` records into loguru.

    Returns:
        The absolute path to the file sink for this run.
    """
    resolved_level = str(level or os.environ.get("BALALAIKA_LOG_LEVEL", "INFO")).upper()
    resolved_dir = _resolve_log_dir(log_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = resolved_dir / f"{stage}_{timestamp}.log"

    logger.remove()
    logger.add(
        sys.stderr,
        format=_FORMAT_CONSOLE,
        level=resolved_level,
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )
    logger.add(
        log_path,
        format=_FORMAT_FILE,
        level=resolved_level,
        rotation="200 MB",
        retention=10,
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    if capture_warnings:
        try:
            import logging
            import warnings

            class _InterceptHandler(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
                    try:
                        level_name = logger.level(record.levelname).name
                    except Exception:
                        level_name = record.levelno
                    logger.opt(depth=6, exception=record.exc_info).log(
                        level_name, record.getMessage()
                    )

            logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
            warnings.simplefilter("default")
        except Exception:
            pass

    logger.info(f"[{stage}] log file: {log_path}")
    return log_path
