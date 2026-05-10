"""Shared CSV-state manager for the balalaika pipeline.

All filtering / scoring stages collaborate around a single ``balalaika.csv``
that lives at ``<podcasts_path>/balalaika.csv``. Each stage adds one or more
columns (``crest_factor``, ``loudness_normalized``, ``music_prob``,
``DistillMOS``, …) and may delete rows whose physical files were removed.

This module centralises the bookkeeping so every stage gets the same
guarantees:

* **Bootstrapping.** If ``balalaika.csv`` is missing when a stage starts, it
  is recreated from the audio tree (filepath-only column) so the stage has a
  place to merge results into.
* **Atomic writes.** ``balalaika.csv`` is rewritten via tmp-file + rename, so
  ``SIGINT`` / ``SIGKILL`` mid-write cannot corrupt it.
* **Incremental worker partials.** Each worker streams rows to its own
  ``<prefix>_part_<rank>.csv`` row by row (``flush()`` after every row), so a
  forced stop preserves whatever rows were already produced. On the next run
  the leftover partials are merged into the main CSV before the stage decides
  what's still pending — re-runs *resume* instead of starting over.
* **Skip-already-processed.** Stages query ``unprocessed_paths`` against a
  particular column to find files still missing a value, regardless of the
  worker count or how the previous run was killed.
* **Drop-deleted-files awareness.** Filter stages (crest, music) can pass
  ``drop_missing_files=True`` so rows whose audio was removed are pruned from
  the main CSV during the merge.
"""
from __future__ import annotations

import csv
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd
from loguru import logger

CSV_NAME = "balalaika.csv"

# Canonical column ordering for the main CSV. Anything not listed is appended
# after the recognised columns in original insertion order.
BASE_COLUMNS: Tuple[str, ...] = (
    "filepath",
    "speaker_id",
    "start",
    "end",
    "total_duration",
    "playlist_id",
    "podcast_id",
    "silence_percent",
    "max_silence_duration",
    "is_single_speaker",
    "crest_factor",
    "loudness_normalized",
    "music_prob",
    "DistillMOS",
)


def csv_path(podcasts_path: os.PathLike | str) -> Path:
    """Return the canonical path to ``balalaika.csv`` for a dataset root."""
    return Path(podcasts_path) / CSV_NAME


def resolve_path(p: os.PathLike | str) -> str:
    """Normalise a filesystem path to an absolute string."""
    return str(Path(p).resolve())


# ---------------------------------------------------------------------------
# Atomic CSV read/write helpers
# ---------------------------------------------------------------------------

def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    """Best-effort read; tolerates a stale ``.tmp`` left by an earlier crash."""
    if path.exists():
        try:
            return pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except Exception as exc:
            logger.warning(f"Failed to read {path}: {exc}; trying tmp fallback.")

    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        try:
            df = pd.read_csv(tmp, low_memory=False)
            logger.info(f"Recovered {len(df)} rows from leftover {tmp.name}.")
            return df
        except Exception as exc:
            logger.warning(f"Could not recover {tmp}: {exc}")

    return None


def atomic_write_csv(df: pd.DataFrame, path: os.PathLike | str) -> None:
    """Write ``df`` to ``path`` atomically (tmp file + rename + fsync)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    try:
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
    except OSError:
        pass
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main-CSV operations
# ---------------------------------------------------------------------------

def _normalize_filepath_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "filepath" in df.columns:
        df = df.copy()
        df["filepath"] = df["filepath"].astype(str).map(resolve_path)
    return df


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    base = [c for c in BASE_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in base]
    return df[base + extras]


def load_main_csv(podcasts_path: os.PathLike | str) -> pd.DataFrame:
    """Return the current ``balalaika.csv`` (or an empty DataFrame)."""
    df = _read_csv_safe(csv_path(podcasts_path))
    if df is None:
        return pd.DataFrame(columns=["filepath"])
    if "filepath" not in df.columns:
        df["filepath"] = ""
    return _normalize_filepath_column(df)


def ensure_main_csv(
    podcasts_path: os.PathLike | str,
    audio_paths: Optional[Iterable[os.PathLike | str]] = None,
) -> pd.DataFrame:
    """Create ``balalaika.csv`` if missing, optionally populating from audio paths.

    Returns the loaded DataFrame (potentially empty if ``audio_paths`` was not
    supplied and the CSV did not yet exist).
    """
    target = csv_path(podcasts_path)
    df = _read_csv_safe(target)

    if df is None or df.empty or "filepath" not in df.columns:
        if audio_paths is None:
            logger.info(f"{target} is missing; creating empty CSV.")
            df = pd.DataFrame(columns=["filepath"])
        else:
            paths = sorted({resolve_path(p) for p in audio_paths})
            df = pd.DataFrame({"filepath": paths})
            logger.info(
                f"{target} missing — bootstrapped with {len(df)} audio paths."
            )
        atomic_write_csv(df, target)
        return df

    return _normalize_filepath_column(df)


def upsert_columns(
    podcasts_path: os.PathLike | str,
    results_df: pd.DataFrame,
    value_columns: Sequence[str],
    *,
    drop_missing_files: bool = False,
    bootstrap_audio_paths: Optional[Iterable[os.PathLike | str]] = None,
) -> pd.DataFrame:
    """Merge ``results_df`` into ``balalaika.csv`` on ``filepath``.

    Args:
        podcasts_path: dataset root (the directory holding ``balalaika.csv``).
        results_df: incoming rows; must have a ``filepath`` column. Other
            columns are taken from ``value_columns``.
        value_columns: which columns from ``results_df`` to write into the
            main CSV. Existing values for these columns are overwritten with
            the new values for matching files; rows for files not yet in the
            CSV are appended.
        drop_missing_files: when True, rows whose ``filepath`` no longer
            exists on disk are dropped before saving (used by filter stages).
        bootstrap_audio_paths: optional list of audio paths to add to the
            CSV's universe before merging (so brand-new files appear even if
            this stage didn't produce a row for them yet).

    Returns the resulting DataFrame after the atomic write.
    """
    target = csv_path(podcasts_path)
    df = _read_csv_safe(target)
    if df is None or "filepath" not in df.columns:
        df = pd.DataFrame(columns=["filepath"])
    df = _normalize_filepath_column(df)

    if bootstrap_audio_paths is not None:
        boot = pd.DataFrame(
            {"filepath": [resolve_path(p) for p in bootstrap_audio_paths]}
        )
        boot = boot.drop_duplicates(subset="filepath")
        # Preserve any existing column values; only add brand-new rows.
        df = pd.concat([df, boot], ignore_index=True).drop_duplicates(
            subset="filepath", keep="first"
        )

    if results_df is not None and not results_df.empty:
        if "filepath" not in results_df.columns:
            raise ValueError("results_df must contain a 'filepath' column")
        results = _normalize_filepath_column(results_df.copy())
        # Keep only the columns we care about (plus filepath).
        present = [c for c in value_columns if c in results.columns]
        results = results[["filepath", *present]].drop_duplicates(
            subset="filepath", keep="last"
        )
        # Drop existing target columns so the merge overwrites cleanly.
        df = df.drop(columns=present, errors="ignore")
        df = df.merge(results, on="filepath", how="outer")

    if drop_missing_files and not df.empty:
        before = len(df)
        df = df[df["filepath"].apply(lambda p: bool(p) and Path(p).exists())]
        removed = before - len(df)
        if removed:
            logger.info(
                f"Pruned {removed} rows whose audio files no longer exist."
            )

    df = _reorder_columns(df)
    atomic_write_csv(df, target)
    return df


def unprocessed_paths(
    podcasts_path: os.PathLike | str,
    column: str,
    audio_paths: Iterable[os.PathLike | str],
) -> List[str]:
    """Return audio paths whose ``column`` value in the main CSV is missing.

    Files that aren't represented in the CSV at all are also returned so a
    fresh-disk-but-stale-CSV state still gets processed.
    """
    df = load_main_csv(podcasts_path)
    audio_resolved = [resolve_path(p) for p in audio_paths]
    audio_set = set(audio_resolved)

    if column not in df.columns or df.empty:
        return audio_resolved

    done_mask = df[column].notna()
    if df[column].dtype == object:
        done_mask &= df[column].astype(str).str.strip().ne("")
    done = set(df.loc[done_mask, "filepath"].tolist())

    return [p for p in audio_resolved if p not in done]


# ---------------------------------------------------------------------------
# Worker partial CSV streams
# ---------------------------------------------------------------------------

def _partial_path(podcasts_path: os.PathLike | str, prefix: str, rank: int) -> Path:
    return Path(podcasts_path) / f"{prefix}_part_{rank}.csv"


def list_partial_csvs(
    podcasts_path: os.PathLike | str, prefix: str
) -> List[Path]:
    """Return existing ``<prefix>_part_*.csv`` files for a dataset root."""
    return sorted(Path(podcasts_path).glob(f"{prefix}_part_*.csv"))


def read_partial_csvs(
    podcasts_path: os.PathLike | str, prefix: str
) -> pd.DataFrame:
    """Read and concatenate any ``<prefix>_part_*.csv`` files into one frame.

    Empty / unreadable parts are skipped (they may have been left as empty
    sentinel files by a worker that exited before producing rows).
    """
    parts = list_partial_csvs(podcasts_path, prefix)
    if not parts:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for p in parts:
        try:
            df = pd.read_csv(p, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        except Exception as exc:
            logger.warning(f"Skipping unreadable partial {p.name}: {exc}")
            continue
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    if "filepath" in merged.columns:
        merged = _normalize_filepath_column(merged)
        merged = merged.drop_duplicates(subset="filepath", keep="last")
    return merged


def delete_partial_csvs(podcasts_path: os.PathLike | str, prefix: str) -> int:
    """Remove all ``<prefix>_part_*.csv`` files; return the number deleted."""
    deleted = 0
    for p in list_partial_csvs(podcasts_path, prefix):
        try:
            p.unlink()
            deleted += 1
        except OSError as exc:
            logger.warning(f"Failed to delete {p}: {exc}")
    return deleted


def absorb_partial_csvs(
    podcasts_path: os.PathLike | str,
    prefix: str,
    value_columns: Sequence[str],
    *,
    drop_missing_files: bool = False,
    bootstrap_audio_paths: Optional[Iterable[os.PathLike | str]] = None,
) -> Tuple[pd.DataFrame, int]:
    """Merge any leftover partials into the main CSV and delete them.

    Returns ``(partials_df, rows_absorbed)``. ``partials_df`` is the raw
    concatenated partials (handy for stage audit accounting); the main CSV is
    updated only when there is something to merge or ``bootstrap_audio_paths``
    is given.
    """
    partials = read_partial_csvs(podcasts_path, prefix)

    if partials.empty and bootstrap_audio_paths is None:
        return partials, 0

    upsert_columns(
        podcasts_path,
        partials,
        value_columns=value_columns,
        drop_missing_files=drop_missing_files,
        bootstrap_audio_paths=bootstrap_audio_paths,
    )
    delete_partial_csvs(podcasts_path, prefix)
    return partials, int(len(partials))


def already_processed_from_partials(
    podcasts_path: os.PathLike | str,
    prefix: str,
    column: str,
) -> Set[str]:
    """Set of filepaths already scored in ``<prefix>_part_*.csv`` (resume aid)."""
    partials = read_partial_csvs(podcasts_path, prefix)
    if partials.empty or "filepath" not in partials.columns or column not in partials.columns:
        return set()
    mask = partials[column].notna()
    if partials[column].dtype == object:
        mask &= partials[column].astype(str).str.strip().ne("")
    return set(partials.loc[mask, "filepath"].astype(str).tolist())


class PartialCsvWriter:
    """Append-only CSV writer for a worker's incremental output.

    Rows are flushed after every ``write`` so a forced stop (``SIGINT`` /
    ``SIGKILL``) keeps the rows already produced. The writer is safe to use as
    a context manager and survives being instantiated many times for the same
    file (it appends, picks up the existing header).

    Field discovery: on first call to ``write``, the union of keys in *that*
    row defines the CSV header. Later rows that introduce new keys cause a
    log warning and fall back to the established header (extra keys are
    dropped, missing keys are written as empty). Keep your worker rows
    homogeneous to avoid surprises.
    """

    def __init__(
        self,
        podcasts_path: os.PathLike | str,
        prefix: str,
        rank: int,
        *,
        fieldnames: Optional[Sequence[str]] = None,
    ) -> None:
        self.path = _partial_path(podcasts_path, prefix, rank)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer: Optional[csv.DictWriter] = None
        self._fields: Optional[List[str]] = list(fieldnames) if fieldnames else None
        self._write_header_pending = False

    def _open_for(self, fields: Sequence[str]) -> None:
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        if not new_file and self._fields is None:
            try:
                with self.path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header:
                        fields = header
            except Exception:
                pass
        self._fields = list(fields)
        self._file = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fields)
        if new_file:
            self._writer.writeheader()
            self._file.flush()

    def write(self, row: Mapping[str, object]) -> None:
        if self._writer is None:
            self._open_for(self._fields if self._fields else list(row.keys()))
        # Drop unknown keys silently; log once per row only on header drift.
        unknown = [k for k in row.keys() if k not in self._fields]
        if unknown:
            logger.debug(
                f"Partial CSV {self.path.name}: dropping unexpected keys {unknown}"
            )
        clean = {k: row.get(k, "") for k in self._fields}
        self._writer.writerow(clean)
        self._file.flush()

    def already_done(self, key_column: str = "filepath") -> Set[str]:
        """Return the values of ``key_column`` already present in the partial."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return set()
        try:
            df = pd.read_csv(self.path, low_memory=False, usecols=[key_column])
        except (ValueError, pd.errors.EmptyDataError, FileNotFoundError):
            return set()
        except Exception as exc:
            logger.warning(f"Could not inspect {self.path}: {exc}")
            return set()
        if key_column not in df.columns:
            return set()
        return set(df[key_column].astype(str).map(resolve_path).tolist())

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
            finally:
                self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self) -> "PartialCsvWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@contextmanager
def partial_writer(
    podcasts_path: os.PathLike | str,
    prefix: str,
    rank: int,
    *,
    fieldnames: Optional[Sequence[str]] = None,
):
    """Convenience context manager around :class:`PartialCsvWriter`."""
    w = PartialCsvWriter(podcasts_path, prefix, rank, fieldnames=fieldnames)
    try:
        yield w
    finally:
        w.close()


# ---------------------------------------------------------------------------
# Filter-stage audit summary
# ---------------------------------------------------------------------------

def audit_from_filter_partials(
    partials_df: pd.DataFrame,
    *,
    deleted_column: str = "deleted",
    duration_column: str = "duration_s",
) -> Dict[str, float]:
    """Compute a stage audit dict from concatenated filter partial CSVs.

    Returns the keys consumed by :func:`src.utils.audit.record_stage_summary`
    plus ``files_deleted``.
    """
    audit: Dict[str, float] = {
        "files_in": 0,
        "files_out": 0,
        "hours_in": 0.0,
        "hours_out": 0.0,
        "files_deleted": 0,
    }
    if partials_df is None or partials_df.empty:
        return audit

    audit["files_in"] = int(len(partials_df))
    if duration_column in partials_df.columns:
        audit["hours_in"] = float(
            partials_df[duration_column].fillna(0.0).sum() / 3600.0
        )

    if deleted_column in partials_df.columns:
        deleted_mask = partials_df[deleted_column].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        ) | (partials_df[deleted_column] == True)  # noqa: E712
        audit["files_deleted"] = int(deleted_mask.sum())
        survived = partials_df[~deleted_mask]
    else:
        survived = partials_df

    audit["files_out"] = int(len(survived))
    if duration_column in survived.columns:
        audit["hours_out"] = float(
            survived[duration_column].fillna(0.0).sum() / 3600.0
        )
    return audit


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def discover_audio_paths(podcasts_path: os.PathLike | str) -> List[str]:
    """Resolve all audio paths under ``podcasts_path`` (lazy import)."""
    from src.utils.utils import get_audio_paths

    return [resolve_path(p) for p in get_audio_paths(str(podcasts_path))]


def files_in_csv(df: pd.DataFrame) -> Set[str]:
    if df is None or df.empty or "filepath" not in df.columns:
        return set()
    return set(df["filepath"].astype(str).map(resolve_path).tolist())
