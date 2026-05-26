"""Render the per-stage filter report (hours dropped at each stage).

Reads ``<podcasts_path>/filter_summary.csv`` (populated by every filter stage
through :func:`src.utils.audit.record_stage_summary`) and writes a Markdown
report to ``<podcasts_path>/filter_report.md``.

Usage::

    python -m src.report --config_path configs/config.yaml
    # or
    python -m src.report --podcasts_path /mnt/data/ruslan
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from src.utils.audit import audit_path
from src.utils.logging_setup import setup_logging
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config

# Stages emit rows in this canonical order; unknown stages are appended after.
STAGE_ORDER = [
    "preprocess",
    "crest_factor",
    "music_detect",
    "distillmos",
    "distillmos_filter",
    "antispoofing",
    "transcription",
    "punctuation",
    "accents",
    "phonemizer",
    "export",
]


def _read_summary(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _format_hours(hours: float) -> str:
    h = int(hours)
    m = int(round((hours - h) * 60))
    return f"{hours:.2f}h ({h}h {m}m)"


def _stage_sort_key(stage: str) -> int:
    try:
        return STAGE_ORDER.index(stage)
    except ValueError:
        return len(STAGE_ORDER) + hash(stage) % 1000


def build_report(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return (
            "# Balalaika filter report\n\n"
            "_No `filter_summary.csv` rows found yet._\n\n"
            "Run the pipeline first; each filter stage will append its summary.\n"
        )

    rows.sort(key=lambda r: r["timestamp"])

    latest_per_stage: Dict[str, Dict[str, str]] = {}
    for r in rows:
        latest_per_stage[r["stage"]] = r

    ordered_stages = sorted(latest_per_stage.keys(), key=_stage_sort_key)

    lines = [
        "# Balalaika filter report",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
        "",
        "Per-stage filtering, summarized in hours of audio kept vs. removed. "
        "Each row is the latest run of that stage in `filter_summary.csv`.",
        "",
        "| Stage | Files in → out | Hours in | Hours out | Hours removed | % removed | Last run | Params |",
        "|-------|----------------|---------:|----------:|--------------:|----------:|----------|--------|",
    ]

    pipeline_first_in: Optional[float] = None
    pipeline_first_files: Optional[int] = None
    pipeline_last_out: float = 0.0
    pipeline_last_files: int = 0

    for stage in ordered_stages:
        r = latest_per_stage[stage]
        files_in = int(r["files_in"])
        files_out = int(r["files_out"])
        hin = float(r["hours_in"])
        hout = float(r["hours_out"])
        hrem = float(r["hours_removed"])
        pct = (hrem / hin * 100.0) if hin > 0 else 0.0

        try:
            params = json.loads(r.get("params") or "{}")
            params_str = ", ".join(f"`{k}={v}`" for k, v in params.items()) if params else "—"
        except Exception:
            params_str = r.get("params") or "—"

        lines.append(
            f"| `{stage}` | {files_in} → {files_out} | "
            f"{hin:.2f} | {hout:.2f} | **{hrem:.2f}** | {pct:.1f}% | "
            f"{r['timestamp']} | {params_str} |"
        )

        if pipeline_first_in is None:
            pipeline_first_in = hin
            pipeline_first_files = files_in
        pipeline_last_out = hout
        pipeline_last_files = files_out

    overall_removed = max(0.0, (pipeline_first_in or 0.0) - pipeline_last_out)
    overall_pct = (
        overall_removed / pipeline_first_in * 100.0 if pipeline_first_in else 0.0
    )

    lines.append("")
    lines.append("**Pipeline net effect (latest stage runs):**")
    lines.append(
        f"- Files: {pipeline_first_files} → {pipeline_last_files}"
    )
    lines.append(
        f"- Hours: {_format_hours(pipeline_first_in or 0.0)} → "
        f"{_format_hours(pipeline_last_out)} "
        f"(**{_format_hours(overall_removed)}** removed, {overall_pct:.1f}%)"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Full history (all runs)")
    lines.append("")
    lines.append(
        "| Timestamp | Stage | Files in → out | Hours in | Hours out | Hours removed | Params |"
    )
    lines.append(
        "|-----------|-------|----------------|---------:|----------:|--------------:|--------|"
    )
    for r in rows:
        try:
            params = json.loads(r.get("params") or "{}")
            params_str = ", ".join(f"`{k}={v}`" for k, v in params.items()) if params else "—"
        except Exception:
            params_str = r.get("params") or "—"
        lines.append(
            f"| {r['timestamp']} | `{r['stage']}` | {r['files_in']} → {r['files_out']} | "
            f"{float(r['hours_in']):.2f} | {float(r['hours_out']):.2f} | "
            f"{float(r['hours_removed']):.2f} | {params_str} |"
        )
    lines.append("")
    return "\n".join(lines)


def _resolve_podcasts_path(args) -> Path:
    if args.podcasts_path:
        return Path(args.podcasts_path)
    if args.config_path:
        for section in ("download", "preprocess", "separation", "export"):
            cfg = load_config(args.config_path, section)
            p = cfg.get("podcasts_path") if cfg else None
            if p:
                logger.info(f"Using podcasts_path from config section '{section}': {p}")
                return Path(p)
    raise SystemExit("Provide --podcasts_path or --config_path with a section that has podcasts_path.")


def main(args):
    setup_logging("report", log_dir=args.log_dir)

    podcasts_path = _resolve_podcasts_path(args)
    summary_path = audit_path(podcasts_path)
    rows = _read_summary(summary_path)

    if not rows:
        logger.warning(f"No rows found in {summary_path}; report will be a placeholder.")

    report_md = build_report(rows)

    output = Path(args.output) if args.output else (podcasts_path / "filter_report.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report_md, encoding="utf-8")
    logger.success(f"Filter report written to {output}")

    if not args.quiet:
        print(report_md)

    write_stage_status(
        stage=13,
        stage_name="report",
        log_dir=args.log_dir or "./logs",
        processed=1,
        skipped=0,
        errors=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render the Balalaika per-stage filter report (hours dropped at each stage)."
    )
    parser.add_argument("--config_path", type=str, default=None, help="YAML config to read podcasts_path from")
    parser.add_argument("--podcasts_path", type=str, default=None, help="Dataset root (overrides config)")
    parser.add_argument("--output", type=str, default=None, help="Output markdown file (default: <podcasts_path>/filter_report.md)")
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    parser.add_argument("--quiet", action="store_true", help="Don't print the report to stdout")
    main(parser.parse_args())
