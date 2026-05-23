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
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd
from loguru import logger

CSV_NAME = "balalaika.csv"

# Default knobs for the periodic merger. Overridable via the top-level `csv:`
# block of configs/config.yaml (see :func:`load_csv_settings`).
DEFAULT_FLUSH_EVERY_ROWS = 10_000
DEFAULT_FLUSH_EVERY_SECONDS = 300

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
    "denoised",
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
    """Write ``df`` to ``path`` atomically (tmp file + rename + fsync).

    A backup ``<path>.bak`` is kept so :func:`ensure_main_csv` can recover if
    the process is killed mid-write and leaves the CSV corrupt.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    bak = Path(str(path) + ".bak")

    if path.exists():
        shutil.copy2(path, bak)

    df.to_csv(tmp, index=False)
    if not tmp.exists():
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

    If the CSV is corrupt (missing, empty, or lacks a ``filepath`` column),
    the function first tries to restore from ``balalaika.csv.bak`` (kept by
    :func:`atomic_write_csv`).  Only when the backup is also unusable does it
    bootstrap a fresh CSV — and logs ``ERROR`` so the operator knows data was
    lost.

    Returns the loaded DataFrame (potentially empty if ``audio_paths`` was not
    supplied and the CSV did not yet exist).
    """
    target = csv_path(podcasts_path)
    bak = Path(str(target) + ".bak")
    df = _read_csv_safe(target)

    if df is None or df.empty or "filepath" not in df.columns:
        bak_df = _read_csv_safe(bak)
        if bak_df is not None and not bak_df.empty and "filepath" in bak_df.columns:
            logger.warning(
                f"{target.name} is corrupt — restored {len(bak_df)} rows from "
                f"{bak.name}"
            )
            atomic_write_csv(bak_df, target)
            df = _normalize_filepath_column(bak_df)
        elif audio_paths is None:
            logger.error(
                f"{target.name} and {bak.name} are both corrupt — "
                "creating empty CSV. Some column data may be permanently lost."
            )
            df = pd.DataFrame(columns=["filepath"])
        else:
            logger.error(
                f"{target.name} and {bak.name} are both corrupt — "
                "bootstrapping fresh CSV from audio tree. Some column data "
                "may be permanently lost."
            )
            paths = sorted({resolve_path(p) for p in audio_paths})
            df = pd.DataFrame({"filepath": paths})

        atomic_write_csv(df, target)
        return df

    return _normalize_filepath_column(df)


def _merge_results_into_df(
    df: pd.DataFrame,
    results: pd.DataFrame,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    """Pure in-memory upsert helper shared by ``upsert_columns`` / ``CsvState``.

    Semantics: for every ``filepath`` present in ``results``, *non-null* values
    in ``value_columns`` overwrite whatever is currently in ``df``. Existing
    values that are not re-supplied by ``results`` are preserved — important
    for the periodic merger, which feeds the state only the *tail* of each
    partial CSV instead of its full content.
    """
    if results is None or results.empty:
        return df
    if "filepath" not in results.columns:
        raise ValueError("results must contain a 'filepath' column")
    results = _normalize_filepath_column(results.copy())
    present = [c for c in value_columns if c in results.columns]
    results = results[["filepath", *present]].drop_duplicates(
        subset="filepath", keep="last"
    )

    if df is None or df.empty or "filepath" not in df.columns:
        df = pd.DataFrame(columns=["filepath"])

    for col in present:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")

    existing = set(df["filepath"].astype(str).tolist())
    new_rows = results[~results["filepath"].isin(existing)]
    if not new_rows.empty:
        df = pd.concat([df, new_rows], ignore_index=True)

    if present:
        df = df.set_index("filepath")
        df.update(results.set_index("filepath")[present])
        df = df.reset_index()

    return df


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

    df = _merge_results_into_df(df, results_df, value_columns)

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
# In-memory mirror + periodic merger
# ---------------------------------------------------------------------------

class CsvState:
    """In-memory mirror of ``balalaika.csv`` for fast periodic upserts.

    Reading + writing the whole CSV every flush is fine for small datasets but
    becomes a real cost at multi-million-row scale. ``CsvState`` keeps the
    DataFrame in RAM, applies upserts there, and only ever writes the file
    atomically.

    The state is owned by the *main* process. Worker processes still produce
    their own ``<prefix>_part_*.csv`` files; they never touch the main CSV
    directly.
    """

    def __init__(self, podcasts_path: os.PathLike | str) -> None:
        self.path = csv_path(podcasts_path)
        df = _read_csv_safe(self.path)
        if df is None or "filepath" not in df.columns:
            df = pd.DataFrame(columns=["filepath"])
        self.df = _normalize_filepath_column(df)

    def upsert(
        self,
        results_df: pd.DataFrame,
        value_columns: Sequence[str],
        *,
        drop_missing_files: bool = False,
        bootstrap_audio_paths: Optional[Iterable[os.PathLike | str]] = None,
    ) -> None:
        """Update the in-memory mirror; does NOT touch disk."""
        if bootstrap_audio_paths is not None:
            boot = pd.DataFrame(
                {"filepath": [resolve_path(p) for p in bootstrap_audio_paths]}
            ).drop_duplicates(subset="filepath")
            self.df = pd.concat([self.df, boot], ignore_index=True).drop_duplicates(
                subset="filepath", keep="first"
            )
        self.df = _merge_results_into_df(self.df, results_df, value_columns)
        if drop_missing_files and not self.df.empty:
            before = len(self.df)
            self.df = self.df[
                self.df["filepath"].apply(lambda p: bool(p) and Path(p).exists())
            ]
            removed = before - len(self.df)
            if removed:
                logger.info(
                    f"Pruned {removed} rows whose audio files no longer exist."
                )
        self.df = _reorder_columns(self.df)

    def save(self) -> None:
        """Flush the in-memory mirror to disk atomically."""
        atomic_write_csv(self.df, self.path)


class PeriodicCsvMerger:
    """Background-thread merger that keeps ``balalaika.csv`` fresh during a stage.

    A single thread runs in the main process and periodically reads each
    worker's partial CSV (``<prefix>_part_*.csv``) and folds new rows into an
    in-memory :class:`CsvState`, then atomically writes ``balalaika.csv``.

    A flush is triggered whenever **either** condition is met:

    * cumulative ``progress_counter`` advanced by ``flush_every_rows`` since
      the last flush (set ``0`` to disable);
    * ``flush_every_seconds`` elapsed since the last flush (set ``0`` to
      disable).

    Workers keep writing to their partials throughout — this thread never
    deletes them. The post-stage :func:`absorb_partial_csvs` does the final
    merge and cleanup, exactly as before. So the merger only *adds* freshness;
    if it is killed mid-way, no data is lost.

    Optimisations:

    * Per-partial byte-offset tracking — each iteration only reads the *new*
      tail of each partial instead of re-parsing the whole file.
    * The main CSV stays resident in :class:`CsvState`; we never re-read it
      from disk between flushes.
    """

    def __init__(
        self,
        podcasts_path: os.PathLike | str,
        prefix: str,
        value_columns: Sequence[str],
        *,
        progress_counter: Any = None,
        flush_every_rows: int = DEFAULT_FLUSH_EVERY_ROWS,
        flush_every_seconds: float = DEFAULT_FLUSH_EVERY_SECONDS,
        drop_missing_files: bool = False,
        bootstrap_audio_paths: Optional[Iterable[os.PathLike | str]] = None,
        poll_interval: float = 5.0,
    ) -> None:
        self.podcasts_path = Path(podcasts_path)
        self.prefix = prefix
        self.value_columns = list(value_columns)
        self.progress_counter = progress_counter
        self.flush_every_rows = max(0, int(flush_every_rows or 0))
        self.flush_every_seconds = max(0.0, float(flush_every_seconds or 0.0))
        self.drop_missing_files = drop_missing_files
        self.bootstrap_audio_paths = (
            [resolve_path(p) for p in bootstrap_audio_paths]
            if bootstrap_audio_paths is not None
            else None
        )
        self.poll_interval = max(0.5, float(poll_interval))

        self._state: Optional[CsvState] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        # filename -> next byte offset to read from in the partial CSV
        self._offsets: Dict[str, int] = {}
        # Rows we've buffered locally but not yet folded into balalaika.csv.
        self._pending: List[pd.DataFrame] = []
        self._pending_rows = 0
        self._last_flush_ts = 0.0
        self._enabled = self.flush_every_rows > 0 or self.flush_every_seconds > 0

    def _ingest_new_partial_rows(self) -> int:
        """Append fresh rows from each partial into ``self._pending``.

        Reads only the *new* tail of each partial since the previous call,
        so this is O(new bytes), not O(partial size). Returns the number of
        rows appended this iteration.
        """
        added = 0
        for path in list_partial_csvs(self.podcasts_path, self.prefix):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size == 0:
                continue
            offset = self._offsets.get(path.name, 0)
            if offset >= size:
                continue
            try:
                if offset == 0:
                    df = pd.read_csv(path, low_memory=False)
                else:
                    # Stream the appended tail, prepending the original header
                    # so pandas can parse it.
                    with path.open("rb") as f:
                        header = f.readline()
                        f.seek(offset)
                        tail = f.read()
                    if not tail.strip():
                        self._offsets[path.name] = size
                        continue
                    from io import BytesIO

                    df = pd.read_csv(
                        BytesIO(header + tail),
                        low_memory=False,
                        on_bad_lines="skip",
                    )
                self._offsets[path.name] = size
            except pd.errors.EmptyDataError:
                self._offsets[path.name] = size
                continue
            except Exception as exc:
                logger.debug(f"Periodic merger: skipping partial {path.name}: {exc}")
                continue
            if not df.empty:
                self._pending.append(df)
                added += int(len(df))
        self._pending_rows += added
        return added

    def _take_pending(self) -> pd.DataFrame:
        if not self._pending:
            return pd.DataFrame()
        merged = pd.concat(self._pending, ignore_index=True)
        self._pending.clear()
        self._pending_rows = 0
        if "filepath" in merged.columns:
            merged = _normalize_filepath_column(merged)
            merged = merged.drop_duplicates(subset="filepath", keep="last")
        return merged

    def flush_now(self, *, force: bool = False) -> int:
        """Synchronously merge new partial rows into the main CSV.

        Returns the number of rows folded in this flush. ``force=True`` flushes
        even when nothing changed (mainly useful for the post-stage call).
        """
        if self._state is None:
            self._state = CsvState(self.podcasts_path)
            if self.bootstrap_audio_paths:
                self._state.upsert(
                    pd.DataFrame(),
                    value_columns=self.value_columns,
                    bootstrap_audio_paths=self.bootstrap_audio_paths,
                )

        with self._lock:
            self._ingest_new_partial_rows()
            new_rows = self._take_pending()
            if new_rows.empty and not force:
                return 0
            if not new_rows.empty:
                self._state.upsert(
                    new_rows,
                    value_columns=self.value_columns,
                    drop_missing_files=self.drop_missing_files,
                )
            try:
                self._state.save()
            except Exception as exc:
                logger.error(f"Periodic merger: atomic save failed: {exc}")
                return 0
            self._last_flush_ts = time.time()
            return int(len(new_rows))

    def _should_flush(self) -> bool:
        if not self._enabled:
            return False
        now = time.time()
        if (
            self.flush_every_seconds > 0
            and now - self._last_flush_ts >= self.flush_every_seconds
            and self._pending_rows > 0
        ):
            return True
        if self.flush_every_rows > 0 and self._pending_rows >= self.flush_every_rows:
            return True
        # If a progress counter was supplied, also trigger off it. Useful when
        # rows are produced very slowly but the worker still wants to mark
        # progress in balalaika.csv (e.g. for large multi-row chunks).
        c = self.progress_counter
        if c is not None and self.flush_every_rows > 0:
            try:
                if int(c.value) // self.flush_every_rows > getattr(
                    self, "_last_counter_bucket", -1
                ):
                    self._last_counter_bucket = int(c.value) // self.flush_every_rows
                    return self._pending_rows > 0
            except AttributeError:
                pass
        return False

    def _loop(self) -> None:
        self._last_flush_ts = time.time()
        while not self._stop.is_set():
            try:
                self._ingest_new_partial_rows()
            except Exception as exc:
                logger.debug(f"Periodic merger: ingest failed: {exc}")
            if self._should_flush():
                try:
                    rows = self.flush_now()
                    if rows:
                        logger.debug(
                            f"Periodic merger: folded {rows} new rows into "
                            f"{csv_path(self.podcasts_path).name}"
                        )
                except Exception as exc:
                    logger.warning(f"Periodic merger flush failed: {exc}")
            self._stop.wait(self.poll_interval)

    def __enter__(self) -> "PeriodicCsvMerger":
        if self._enabled:
            self._thread = threading.Thread(
                target=self._loop,
                name=f"csv-merger-{self.prefix}",
                daemon=True,
            )
            self._thread.start()
            logger.info(
                f"Periodic CSV merger started: every {self.flush_every_rows} rows "
                f"or {self.flush_every_seconds}s — folds {self.prefix}_part_*.csv "
                f"into {csv_path(self.podcasts_path).name}."
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval * 2)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_csv_settings(config_path: Optional[os.PathLike | str]) -> Dict[str, float]:
    """Read the top-level ``csv:`` block from a YAML config.

    Returns a dict with ``flush_every_rows`` (int) and
    ``flush_every_seconds`` (float). Missing keys/files fall back to the
    documented defaults so stages never crash on a stale config.
    """
    settings = {
        "flush_every_rows": DEFAULT_FLUSH_EVERY_ROWS,
        "flush_every_seconds": DEFAULT_FLUSH_EVERY_SECONDS,
    }
    if not config_path:
        return settings
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning(f"Could not read csv settings from {config_path}: {exc}")
        return settings

    block = raw.get("csv") if isinstance(raw, dict) else None
    if not isinstance(block, dict):
        return settings
    try:
        settings["flush_every_rows"] = max(0, int(block.get(
            "flush_every_rows", DEFAULT_FLUSH_EVERY_ROWS
        )))
    except (TypeError, ValueError):
        pass
    try:
        settings["flush_every_seconds"] = max(0.0, float(block.get(
            "flush_every_seconds", DEFAULT_FLUSH_EVERY_SECONDS
        )))
    except (TypeError, ValueError):
        pass
    return settings


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
