#!/usr/bin/env python3
"""Detailed EDA and visualization for Balalaika metadata CSV.

The script reads balalaika.csv, discovers numeric metrics automatically, adds
speech-rate features from text sidecars when available, and writes plots plus
machine-readable summary tables into an analysis directory.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - convenience message for users
    raise SystemExit(
        "matplotlib is required for plots. Install it with: "
        "python -m pip install matplotlib"
    ) from exc


DEFAULT_CSV = Path("/mnt/ssd_1tb_2/youtube_data/balalaika.csv")
TEXT_SUFFIX_PRIORITY = ("_accent.txt", "_punct.txt", "_rover.txt")
ID_LIKE_COLUMNS = {"speaker_id", "playlist_id", "podcast_id"}
WORD_RE = re.compile(r"[\w]+", flags=re.UNICODE)
CORE_METRICS = (
    "DistillMOS",
    "music_prob",
    "crest_factor",
    "silence_percent",
    "max_silence_duration",
    "total_duration",
    "speech_words_per_min",
    "asr_consistency_percent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create detailed visualizations and EDA report for balalaika.csv."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to balalaika.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write plots and reports. Defaults to <csv_dir>/balalaika_analysis.",
    )
    parser.add_argument(
        "--text-source",
        choices=("auto", "accent", "punct", "rover", "none"),
        default="auto",
        help="Text sidecar used for speech-rate calculation.",
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=10,
        help="Number of quantile buckets for speech-rate and metric analysis.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Threads used to read text sidecars.",
    )
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=50000,
        help="Maximum rows sampled for dense scatter plots.",
    )
    return parser.parse_args()


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "plots": output_dir / "plots",
        "core_plots": output_dir / "plots" / "core_quality",
        "metric_plots": output_dir / "plots" / "metrics",
        "tables": output_dir / "tables",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def read_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_parquet(csv_path)
    if "filepath" not in df.columns:
        raise ValueError("Expected a 'filepath' column in balalaika.csv")
    df = df.drop_duplicates(subset=["filepath"]).reset_index(drop=True)
    return df


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == "filepath":
            continue
        if out[col].dtype == object:
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().sum() > 0:
                out[col] = converted
    return out


def add_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "total_duration" in out.columns:
        out["duration_sec"] = pd.to_numeric(out["total_duration"], errors="coerce")
    elif {"start", "end"}.issubset(out.columns):
        out["duration_sec"] = pd.to_numeric(out["end"], errors="coerce") - pd.to_numeric(
            out["start"], errors="coerce"
        )
    else:
        out["duration_sec"] = np.nan

    if "silence_percent" in out.columns:
        silence_fraction = pd.to_numeric(out["silence_percent"], errors="coerce").clip(0, 100) / 100
        out["speech_duration_sec"] = out["duration_sec"] * (1 - silence_fraction)
    else:
        out["speech_duration_sec"] = out["duration_sec"]

    out["duration_min"] = out["duration_sec"] / 60
    out["speech_duration_min"] = out["speech_duration_sec"] / 60
    return out


def sidecar_suffixes(text_source: str) -> tuple[str, ...]:
    if text_source == "none":
        return ()
    if text_source == "auto":
        return TEXT_SUFFIX_PRIORITY
    return (f"_{text_source}.txt",)


def find_sidecar(audio_path: str, suffixes: Iterable[str]) -> Path | None:
    path = Path(audio_path)
    for suffix in suffixes:
        candidate = path.with_name(f"{path.stem}{suffix}")
        if candidate.exists():
            return candidate
    return None


def text_stats_for_path(audio_path: str, suffixes: tuple[str, ...]) -> dict[str, object]:
    sidecar = find_sidecar(audio_path, suffixes)
    if sidecar is None:
        return {
            "filepath": audio_path,
            "text_path": "",
            "text_source": "",
            "text_chars": np.nan,
            "word_count": np.nan,
        }

    try:
        text = sidecar.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        text = ""

    words = WORD_RE.findall(text)
    return {
        "filepath": audio_path,
        "text_path": str(sidecar),
        "text_source": sidecar.name[len(Path(audio_path).stem) :],
        "text_chars": len(text),
        "word_count": len(words),
    }


def add_text_and_speech_rate_features(
    df: pd.DataFrame, text_source: str, workers: int
) -> pd.DataFrame:
    suffixes = sidecar_suffixes(text_source)
    if not suffixes:
        return df

    rows: list[dict[str, object]] = []
    paths = df["filepath"].astype(str).tolist()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(text_stats_for_path, path, suffixes) for path in paths]
        for future in as_completed(futures):
            rows.append(future.result())

    text_df = pd.DataFrame(rows)
    out = df.merge(text_df, on="filepath", how="left")
    out["speech_words_per_min"] = out["word_count"] / out["speech_duration_min"].replace(0, np.nan)
    out["words_per_min_total"] = out["word_count"] / out["duration_min"].replace(0, np.nan)
    out["chars_per_sec_total"] = out["text_chars"] / out["duration_sec"].replace(0, np.nan)
    return out


def numeric_metric_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    excluded = ID_LIKE_COLUMNS | {"start", "end"}
    return [col for col in numeric_cols if col not in excluded]


def core_metric_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in CORE_METRICS if col in df.columns]


def finite_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        values = series.astype("float64")
    else:
        values = pd.to_numeric(series, errors="coerce")
    return values[np.isfinite(values)]


def save_numeric_summary(df: pd.DataFrame, metric_cols: list[str], tables_dir: Path) -> pd.DataFrame:
    summary = df[metric_cols].describe(
        percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).T
    summary["missing"] = df[metric_cols].isna().sum()
    summary["missing_percent"] = (summary["missing"] / len(df) * 100).round(3)
    summary.to_csv(tables_dir / "numeric_summary.csv")
    return summary


def save_core_metric_summary(df: pd.DataFrame, core_cols: list[str], tables_dir: Path) -> pd.DataFrame:
    rows = []
    for col in core_cols:
        values = finite_series(df[col])
        if values.empty:
            continue
        rows.append(
            {
                "metric": col,
                "count": int(values.count()),
                "missing": int(df[col].isna().sum()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "p01": float(values.quantile(0.01)),
                "p05": float(values.quantile(0.05)),
                "p10": float(values.quantile(0.10)),
                "p25": float(values.quantile(0.25)),
                "p50": float(values.quantile(0.50)),
                "p75": float(values.quantile(0.75)),
                "p90": float(values.quantile(0.90)),
                "p95": float(values.quantile(0.95)),
                "p99": float(values.quantile(0.99)),
                "max": float(values.max()),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(tables_dir / "core_metric_summary.csv", index=False)
    return result


def save_metric_quantiles(
    df: pd.DataFrame, metric_cols: list[str], quantiles: int, tables_dir: Path
) -> pd.DataFrame:
    rows = []
    q_values = np.linspace(0, 1, quantiles + 1)
    for col in metric_cols:
        values = finite_series(df[col])
        if values.empty:
            continue
        qs = values.quantile(q_values)
        for q, value in qs.items():
            rows.append({"metric": col, "quantile": float(q), "value": float(value)})
    result = pd.DataFrame(rows)
    result.to_csv(tables_dir / "metric_quantiles.csv", index=False)
    return result


def missing_like_mask(series: pd.Series) -> pd.Series:
    mask = series.isna()
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        mask = mask | series.astype("string").str.strip().eq("").fillna(False)
    return mask


def save_missing_summary(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    masks = {col: missing_like_mask(df[col]) for col in df.columns}
    missing_counts = pd.Series({col: int(mask.sum()) for col, mask in masks.items()})
    missing = pd.DataFrame(
        {
            "column": df.columns,
            "missing": [missing_counts[col] for col in df.columns],
            "missing_percent": [round(float(masks[col].mean() * 100), 3) for col in df.columns],
            "dtype": [str(dtype) for dtype in df.dtypes],
            "unique": [df[col].nunique(dropna=True) for col in df.columns],
        }
    ).sort_values(["missing_percent", "column"], ascending=[False, True])
    missing.to_csv(tables_dir / "missing_values.csv", index=False)
    return missing


def save_categorical_summary(df: pd.DataFrame, tables_dir: Path) -> None:
    categorical_cols = [
        col
        for col in df.columns
        if df[col].dtype == object or pd.api.types.is_bool_dtype(df[col])
    ]
    rows = []
    for col in categorical_cols:
        counts = df[col].astype("string").fillna("<NA>").value_counts(dropna=False).head(50)
        for value, count in counts.items():
            rows.append(
                {
                    "column": col,
                    "value": value,
                    "count": int(count),
                    "percent": round(float(count) / len(df) * 100, 3),
                }
            )
    pd.DataFrame(rows).to_csv(tables_dir / "categorical_top_values.csv", index=False)


def plot_histogram_and_box(df: pd.DataFrame, col: str, output_path: Path) -> None:
    values = finite_series(df[col])
    if values.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(values, bins=min(80, max(10, int(math.sqrt(len(values))))), color="#3b82f6", alpha=0.85)
    axes[0].set_title(f"{col}: distribution")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("count")
    axes[0].grid(alpha=0.25)

    axes[1].boxplot(values, vert=False, showfliers=True)
    axes[1].set_title(f"{col}: boxplot")
    axes[1].set_xlabel(col)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_core_metric_distribution(df: pd.DataFrame, col: str, output_path: Path) -> None:
    values = finite_series(df[col])
    if values.empty:
        return

    p01, p50, p99 = values.quantile([0.01, 0.50, 0.99])
    trimmed = values[(values >= p01) & (values <= p99)]
    if trimmed.empty:
        trimmed = values

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].hist(trimmed, bins=80, color="#2563eb", alpha=0.85)
    axes[0].axvline(p50, color="#dc2626", linestyle="--", linewidth=1.5, label=f"median={p50:.3g}")
    axes[0].set_title(f"{col}: distribution, p01-p99")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("count")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    sorted_values = np.sort(values.to_numpy())
    y = np.linspace(0, 1, len(sorted_values), endpoint=True)
    axes[1].plot(sorted_values, y, color="#16a34a", linewidth=1.8)
    axes[1].axvline(p50, color="#dc2626", linestyle="--", linewidth=1.5)
    axes[1].set_title(f"{col}: cumulative distribution")
    axes[1].set_xlabel(col)
    axes[1].set_ylabel("share <= value")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_core_quality_overview(df: pd.DataFrame, core_cols: list[str], output_path: Path) -> None:
    cols = [col for col in core_cols if col != "speech_words_per_min"]
    if not cols:
        return

    ncols = 2
    nrows = math.ceil(len(cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        values = finite_series(df[col])
        if values.empty:
            ax.axis("off")
            continue
        p01, p50, p99 = values.quantile([0.01, 0.50, 0.99])
        trimmed = values[(values >= p01) & (values <= p99)]
        if trimmed.empty:
            trimmed = values
        ax.hist(trimmed, bins=80, color="#2563eb", alpha=0.85)
        ax.axvline(p50, color="#dc2626", linestyle="--", linewidth=1.5)
        ax.set_title(f"{col} distribution (p01-p99), median={p50:.3g}")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.grid(alpha=0.25)

    for ax in axes[len(cols) :]:
        ax.axis("off")

    fig.suptitle("Core quality metric distributions", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame, metric_cols: list[str], output_path: Path) -> None:
    corr_cols = [col for col in metric_cols if finite_series(df[col]).nunique() > 1]
    if len(corr_cols) < 2:
        return

    corr = df[corr_cols].corr(numeric_only=True)
    fig_size = max(7, min(18, 0.55 * len(corr_cols)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax.set_yticklabels(corr_cols)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Numeric metric correlation")

    if len(corr_cols) <= 20:
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    corr.to_csv(output_path.with_suffix(".csv"))


def plot_missingness(missing: pd.DataFrame, output_path: Path) -> None:
    data = missing.sort_values("missing_percent", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(data))))
    ax.barh(data["column"], data["missing_percent"], color="#f97316")
    ax.set_xlabel("missing, %")
    ax.set_title("Missing values by column")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_speech_rate(df: pd.DataFrame, tables_dir: Path, plots_dir: Path, quantiles: int) -> None:
    col = "speech_words_per_min"
    if col not in df.columns or finite_series(df[col]).empty:
        return

    values = finite_series(df[col])
    q_values = np.linspace(0, 1, quantiles + 1)
    quantile_table = values.quantile(q_values).reset_index()
    quantile_table.columns = ["quantile", col]
    quantile_table.to_csv(tables_dir / "speech_rate_quantiles.csv", index=False)

    plot_histogram_and_box(df, col, plots_dir / "speech_rate_distribution.png")

    tmp = df.copy()
    tmp["speech_rate_quantile"] = pd.qcut(
        pd.to_numeric(tmp[col], errors="coerce"),
        q=quantiles,
        duplicates="drop",
    )
    grouped = (
        tmp.groupby("speech_rate_quantile", observed=True)
        .agg(
            rows=(col, "size"),
            speech_words_per_min_mean=(col, "mean"),
            speech_words_per_min_median=(col, "median"),
            duration_sec_median=("duration_sec", "median"),
            silence_percent_median=(
                "silence_percent",
                "median" if "silence_percent" in tmp.columns else "size",
            ),
            DistillMOS_median=("DistillMOS", "median" if "DistillMOS" in tmp.columns else "size"),
            music_prob_median=("music_prob", "median" if "music_prob" in tmp.columns else "size"),
        )
        .reset_index()
    )
    grouped["speech_rate_quantile"] = grouped["speech_rate_quantile"].astype(str)
    grouped.to_csv(tables_dir / "speech_rate_quantile_buckets.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(grouped["speech_rate_quantile"], grouped["rows"], color="#22c55e")
    ax.set_title("Rows per speech-rate quantile")
    ax.set_xlabel("speech_words_per_min quantile")
    ax.set_ylabel("rows")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "speech_rate_quantile_counts.png", dpi=160)
    plt.close(fig)


def plot_metric_vs_speech_rate(
    df: pd.DataFrame, metric_cols: list[str], plots_dir: Path, max_scatter_points: int
) -> None:
    if "speech_words_per_min" not in df.columns or finite_series(df["speech_words_per_min"]).empty:
        return

    cols = [
        col
        for col in metric_cols
        if col != "speech_words_per_min" and finite_series(df[col]).nunique() > 1
    ]
    if not cols:
        return

    sample = df[["speech_words_per_min", *cols]].dropna()
    if sample.empty:
        return
    if len(sample) > max_scatter_points:
        sample = sample.sample(max_scatter_points, random_state=42)

    for col in cols:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(sample["speech_words_per_min"], sample[col], s=5, alpha=0.25, color="#6366f1")
        ax.set_xlabel("speech_words_per_min")
        ax.set_ylabel(col)
        ax.set_title(f"{col} vs speech rate")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{safe_filename(col)}_vs_speech_rate.png", dpi=150)
        plt.close(fig)


def plot_duration_quality_overview(df: pd.DataFrame, plots_dir: Path) -> None:
    candidates = [col for col in ("DistillMOS", "music_prob", "silence_percent", "crest_factor") if col in df]
    if not candidates or "duration_sec" not in df:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    for ax, col in zip(axes, candidates):
        data = df[["duration_sec", col]].dropna()
        if len(data) > 50000:
            data = data.sample(50000, random_state=42)
        ax.scatter(data["duration_sec"], data[col], s=4, alpha=0.25)
        ax.set_xlabel("duration_sec")
        ax.set_ylabel(col)
        ax.set_title(f"{col} vs duration")
        ax.grid(alpha=0.25)
    for ax in axes[len(candidates) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(plots_dir / "duration_quality_overview.png", dpi=160)
    plt.close(fig)


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def write_report(
    df: pd.DataFrame,
    output_dir: Path,
    metric_cols: list[str],
    core_cols: list[str],
    numeric_summary: pd.DataFrame,
    missing: pd.DataFrame,
) -> None:
    text_rows = int(df["word_count"].notna().sum()) if "word_count" in df.columns else 0
    total_hours = float(df["duration_sec"].sum(skipna=True) / 3600) if "duration_sec" in df else float("nan")
    report = [
        "# Balalaika Dataset Analysis",
        "",
        f"- Rows: {len(df):,}",
        f"- Columns: {len(df.columns):,}",
        f"- Total audio hours: {total_hours:,.2f}",
        f"- Numeric metrics visualized: {len(metric_cols):,}",
        f"- Rows with discovered text sidecars: {text_rows:,}",
        "",
        "## Main Output Files",
        "",
        "- `plots/core_quality/core_quality_distributions.png` - main quality metric distributions",
        "- `plots/core_quality/<metric>_distribution.png` - detailed histogram and CDF per core metric",
        "- `tables/core_metric_summary.csv` - focused summary for the important metrics",
        "- `tables/numeric_summary.csv` - descriptive statistics for every numeric metric",
        "- `tables/metric_quantiles.csv` - metric values at quantile boundaries",
        "- `tables/speech_rate_quantiles.csv` - speech-rate quantiles when text exists",
        "- `plots/metrics/*.png` - distribution and boxplot for every metric",
        "- `plots/correlation_heatmap.png` - correlation between numeric metrics",
        "- `plots/speech_rate_distribution.png` - speech-rate distribution",
        "",
        "## Highest Missingness",
        "",
    ]

    for _, row in missing.head(10).iterrows():
        report.append(
            f"- `{row['column']}`: {int(row['missing']):,} missing ({row['missing_percent']:.3f}%)"
        )

    report.extend(["", "## Core Quality Metrics", ""])
    for col in core_cols:
        if col not in numeric_summary.index:
            continue
        row = numeric_summary.loc[col]
        mean = row.get("mean", np.nan)
        p10 = row.get("10%", np.nan)
        p50 = row.get("50%", np.nan)
        p90 = row.get("90%", np.nan)
        report.append(f"- `{col}`: mean={mean:.4g}, p10={p10:.4g}, p50={p50:.4g}, p90={p90:.4g}")

    secondary_cols = [col for col in metric_cols if col not in core_cols]
    if secondary_cols:
        report.extend(["", "## Secondary Numeric Metrics", ""])
        for col in secondary_cols[:20]:
            if col not in numeric_summary.index:
                continue
            row = numeric_summary.loc[col]
            mean = row.get("mean", np.nan)
            p50 = row.get("50%", np.nan)
            p90 = row.get("90%", np.nan)
            report.append(f"- `{col}`: mean={mean:.4g}, p50={p50:.4g}, p90={p90:.4g}")

    if "speech_words_per_min" in df.columns and finite_series(df["speech_words_per_min"]).notna().any():
        speech = finite_series(df["speech_words_per_min"])
        report.extend(
            [
                "",
                "## Speech Rate",
                "",
                f"- Median words/minute: {speech.median():.2f}",
                f"- P10 words/minute: {speech.quantile(0.10):.2f}",
                f"- P90 words/minute: {speech.quantile(0.90):.2f}",
            ]
        )

    report.extend(
        [
            "",
            "## Missing Values By Column",
            "",
            "NaN values and empty strings are counted as missing.",
            "",
            "| Column | Missing | Missing, % | Dtype |",
            "|--------|--------:|-----------:|-------|",
        ]
    )
    for _, row in missing.iterrows():
        report.append(
            f"| `{row['column']}` | {int(row['missing']):,} | {row['missing_percent']:.3f}% | `{row['dtype']}` |"
        )

    (output_dir / "analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def save_overview_json(df: pd.DataFrame, output_dir: Path) -> None:
    overview = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "total_hours": float(df["duration_sec"].sum(skipna=True) / 3600),
        "columns_list": df.columns.tolist(),
        "rows_with_text": int(df["word_count"].notna().sum()) if "word_count" in df else 0,
    }
    (output_dir / "overview.json").write_text(
        json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.csv.parent / "balalaika_analysis"
    dirs = ensure_output_dirs(output_dir)

    df = read_csv(args.csv)
    df = coerce_numeric_columns(df)
    df = add_duration_features(df)
    df = add_text_and_speech_rate_features(df, args.text_source, args.workers)

    enriched_path = dirs["tables"] / "balalaika_enriched_metrics.csv"
    df.to_csv(enriched_path, index=False)

    metric_cols = numeric_metric_columns(df)
    core_cols = core_metric_columns(df)
    missing = save_missing_summary(df, dirs["tables"])
    numeric_summary = save_numeric_summary(df, metric_cols, dirs["tables"])
    save_core_metric_summary(df, core_cols, dirs["tables"])
    save_metric_quantiles(df, metric_cols, args.quantiles, dirs["tables"])
    save_categorical_summary(df, dirs["tables"])
    save_overview_json(df, dirs["root"])

    plot_core_quality_overview(df, core_cols, dirs["core_plots"] / "core_quality_distributions.png")
    for col in core_cols:
        plot_core_metric_distribution(df, col, dirs["core_plots"] / f"{safe_filename(col)}_distribution.png")
    for col in metric_cols:
        plot_histogram_and_box(df, col, dirs["metric_plots"] / f"{safe_filename(col)}.png")
    plot_correlation(df, metric_cols, dirs["plots"] / "correlation_heatmap.png")
    plot_missingness(missing, dirs["plots"] / "missing_values.png")
    plot_speech_rate(df, dirs["tables"], dirs["plots"], args.quantiles)
    plot_metric_vs_speech_rate(df, metric_cols, dirs["plots"], args.max_scatter_points)
    plot_duration_quality_overview(df, dirs["plots"])

    write_report(df, dirs["root"], metric_cols, core_cols, numeric_summary, missing)
    print(f"Analysis written to: {output_dir}")
    print(f"Open report: {output_dir / 'analysis_report.md'}")
    print(f"Enriched CSV: {enriched_path}")


if __name__ == "__main__":
    main()
