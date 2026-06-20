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
import fcntl
import os
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd
from loguru import logger
from tqdm import tqdm

CSV_NAME = "balalaika.csv"
PARQUET_NAME = "balalaika.parquet"
AUDIO_EXTENSIONS: Tuple[str, ...] = (".mp3", ".wav", ".flac", ".ogg", ".opus")

# Default knobs for the periodic merger. Overridable via the top-level `csv:`
# block of configs/config.yaml (see :func:`load_csv_settings`).
DEFAULT_FLUSH_EVERY_ROWS = 10_000
DEFAULT_FLUSH_EVERY_SECONDS = 300
CSV_WRITE_CHUNK_ROWS = 100_000
_CSV_THREAD_LOCK = threading.RLock()

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
    "score_bonafide",
    "score_spoof",
    "denoised",
)


def parquet_path(podcasts_path: os.PathLike | str) -> Path:
    """Return the path to ``balalaika.parquet`` for a dataset root."""
    return Path(podcasts_path) / PARQUET_NAME


def state_format() -> str:
    """The pipeline state format. Always ``"parquet"``.

    CSV state was removed (it is an inefficient redundant copy); the live state
    is ``balalaika.parquet`` only and no ``balalaika.csv`` is produced. The
    function is kept (returning a constant) so existing format-aware call sites
    stay valid.
    """
    return "parquet"


def state_path(podcasts_path: os.PathLike | str) -> Path:
    """Return the active pipeline-state file (``balalaika.parquet``).

    All state ops (load / atomic write / flush / absorb / drop_missing / narrow
    reads) operate on this file.
    """
    return parquet_path(podcasts_path)


def resolve_path(p: os.PathLike | str) -> str:
    """Normalise a filesystem path to an absolute string."""
    return str(Path(p).resolve())


def normalize_path_string(p: os.PathLike | str) -> str:
    """Fast path normalisation for already-absolute CSV filepaths.

    ``Path.resolve()`` performs filesystem work and is prohibitively expensive
    when repeated over tens of millions of rows. Pipeline CSVs store absolute
    paths, so only relative paths need resolution. Absolute paths are trusted
    verbatim (no ``..`` collapsing) — same contract as the original
    implementation, but without constructing a ``Path`` object per call.
    """
    path = str(p).strip()
    if not path:
        return ""
    return path if os.path.isabs(path) else resolve_path(path)


def _sequence_total(values: Iterable[object]) -> Optional[int]:
    try:
        return len(values)  # type: ignore[arg-type]
    except TypeError:
        return None


def normalize_path_values(
    values: Iterable[os.PathLike | str],
    *,
    desc: str,
    drop_empty: bool = False,
) -> List[str]:
    """Normalise many path values.

    Hot path for multi-million-row CSV passes: plain string ops, no per-row
    tqdm/Path overhead. ``desc`` is kept for signature compatibility and used
    only for a summary debug log.
    """
    out: List[str] = []
    append = out.append
    isabs = os.path.isabs
    for raw in values:
        path = str(raw).strip()
        if not path:
            if not drop_empty:
                append("")
            continue
        append(path if isabs(path) else resolve_path(path))
    logger.debug(f"{desc}: normalized {len(out)} path value(s).")
    return out


def _normalize_path_series(values: pd.Series) -> pd.Series:
    """Vectorised :func:`normalize_path_string` over a pandas Series."""
    s = values.astype(str).str.strip()
    needs_resolve = ~(s.str.startswith(os.sep) | s.eq(""))
    if needs_resolve.any():
        s.loc[needs_resolve] = [resolve_path(p) for p in s.loc[needs_resolve]]
    return s


def _filepath_is_canonical(col: pd.Series) -> bool:
    """Cheap check: is every value already an absolute, stripped path string?

    Steady-state pipeline CSVs store absolute, already-stripped paths, so the
    expensive ``astype(str)`` materialisation + per-row resolve done by
    :func:`_normalize_path_series` is almost always a no-op that produces a
    value-identical Series — but at the cost of a fresh object column the size
    of the whole frame. This guard answers "would normalization change
    anything?" so we can skip the duplicate (and the enclosing ``df.copy()``)
    entirely when it wouldn't.

    A value is left unchanged by normalization iff it is already a ``str`` that
    is absolute and free of surrounding whitespace. Any non-string (NaN, None,
    numbers), relative path, empty string, or whitespace-padded value forces
    the full normalization path.
    """
    if col.dtype != object:
        # Numeric / bool columns stringify to a different representation.
        return False
    sep = os.sep
    for v in col.to_numpy():
        if type(v) is not str:
            return False  # NaN/None/numbers -> astype(str) changes them
        if not v.startswith(sep):
            return False  # relative path (incl. "") -> would be resolved
        if v != v.strip():
            return False  # surrounding whitespace would be stripped
    return True


def _paths_exist_mask(paths: Sequence[str], *, desc: str) -> List[bool]:
    """Existence check for many paths with one scandir per unique directory.

    Equivalent to ``os.path.exists(p)`` per path, but instead of one stat
    syscall per row (O(N) syscalls — minutes on multi-million-row CSVs) it
    lists each distinct parent directory once and answers from the name set.
    Symlink entries are verified with a real ``os.path.exists`` so dangling
    symlinks still read as missing, matching the per-path semantics.
    """
    if len(paths) < 10_000:
        exists = os.path.exists
        return [bool(p) and exists(p) for p in paths]

    names_cache: Dict[str, Set[str]] = {}

    def dir_names(d: str) -> Set[str]:
        cached = names_cache.get(d)
        if cached is not None:
            return cached
        present: Set[str] = set()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_symlink():
                            if os.path.exists(entry.path):
                                present.add(entry.name)
                        else:
                            present.add(entry.name)
                    except OSError:
                        continue
        except OSError:
            pass  # directory itself missing/unreadable -> nothing exists in it
        names_cache[d] = present
        return present

    split = os.path.split
    mask: List[bool] = []
    append = mask.append
    for p in tqdm(paths, desc=desc, mininterval=1.0):
        if not p:
            append(False)
            continue
        d, name = split(p)
        append(name in dir_names(d))
    return mask


# ---------------------------------------------------------------------------
# Atomic CSV read/write helpers
# ---------------------------------------------------------------------------

def _pandas_is_cudf_proxy() -> bool:
    try:
        import cudf.pandas as cudf_pandas

        is_proxy_object = getattr(cudf_pandas, "is_proxy_object", None)
        if callable(is_proxy_object) and is_proxy_object(pd.DataFrame()):
            return True
    except Exception:
        pass

    try:
        return "cudf" in type(pd.DataFrame()).__module__
    except Exception:
        return False


_FORCE_C_ENGINE = os.environ.get("BALALAIKA_CSV_ENGINE", "").lower() == "c"


def fast_read_csv(path, **kwargs) -> pd.DataFrame:
    """``pd.read_csv`` with the multithreaded pyarrow parser when safe.

    The pyarrow engine reads large CSVs ~4x faster than the default C engine.
    Float values that were written with full 17-digit precision may differ by
    1 ULP from the C parser — bounded, non-cumulative, and far below the
    measurement noise of any score stored in balalaika.csv. Set
    ``BALALAIKA_CSV_ENGINE=c`` to force the old parser. When cudf.pandas is
    active the call is left untouched so it can route to GPU.
    """
    if _FORCE_C_ENGINE or _pandas_is_cudf_proxy():
        return pd.read_csv(path, low_memory=False, **kwargs)
    try:
        return pd.read_csv(path, engine="pyarrow", **kwargs)
    except (ValueError, TypeError, ImportError):
        # unsupported kwarg combination or missing pyarrow -> C engine
        return pd.read_csv(path, low_memory=False, **kwargs)


def _read_parquet(path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Read a parquet state file (optionally projecting ``columns``).

    Column projection is genuinely cheap in parquet (only the requested column
    chunks leave the disk), which is what makes the narrow reads in
    :func:`unprocessed_paths` / the duration cache so much lighter in parquet
    mode than the equivalent CSV ``usecols`` read.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=list(columns) if columns is not None else None)
    return table.to_pandas()


def _is_parquet_path(path: Path) -> bool:
    """True for the parquet state file and its ``.bak`` / ``.tmp`` siblings.

    The backup/temp helpers append a suffix (``balalaika.parquet.bak``,
    ``balalaika.parquet.tmp.<pid>``), so a plain ``.suffix`` check is not
    enough — match on the ``.parquet`` component anywhere in the name.
    """
    name = path.name.lower()
    return name.endswith(".parquet") or ".parquet." in name


def _read_state_body(path: Path) -> pd.DataFrame:
    """Read a state file, dispatching on its suffix (parquet vs CSV)."""
    if _is_parquet_path(path):
        return _read_parquet(path)
    return fast_read_csv(path)


def _state_header(path: Path) -> Optional[List[str]]:
    """Return the column names of a state file without reading the body.

    CSV: read the first line. Parquet: read the schema (metadata only). Returns
    ``None`` if the file is missing/unreadable.
    """
    if not path.exists():
        return None
    try:
        if _is_parquet_path(path):
            import pyarrow.parquet as pq

            return list(pq.read_schema(path).names)
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(next(csv.reader(f)))
    except Exception:
        return None


def _read_state_narrow(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read only ``columns`` from a state file (parquet projection / CSV usecols)."""
    if _is_parquet_path(path):
        return _read_parquet(path, columns=list(columns))
    return fast_read_csv(path, usecols=list(columns))


def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    """Best-effort read; tolerates a stale ``.tmp`` left by an earlier crash.

    Works for both CSV and parquet state files (dispatch on suffix). Named for
    its historical CSV role; the body reader handles either format.
    """
    if path.exists():
        try:
            logger.info(f"Reading state {path.name}...")
            df = _read_state_body(path)
            logger.info(f"Read {len(df)} rows from {path.name}.")
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except Exception as exc:
            logger.warning(f"Failed to read {path}: {exc}; trying tmp fallback.")

    tmp_candidates = [path.with_suffix(path.suffix + ".tmp")]
    tmp_candidates.extend(
        sorted(
            path.parent.glob(f"{path.name}.tmp.*"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        )
    )
    for tmp in tmp_candidates:
        if not tmp.exists():
            continue
        try:
            logger.info(f"Reading state fallback {tmp.name}...")
            df = _read_state_body(tmp)
            logger.info(f"Recovered {len(df)} rows from leftover {tmp.name}.")
            return df
        except Exception as exc:
            logger.warning(f"Could not recover {tmp}: {exc}")

    return None


@contextmanager
def _csv_write_lock(path: Path):
    """Serialize read/merge/write cycles for one main CSV across threads/processes."""
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _CSV_THREAD_LOCK:
        with lock_path.open("a", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _copy_file_with_progress(src: Path, dst: Path, *, desc: str) -> None:
    total = src.stat().st_size
    with src.open("rb") as fsrc, dst.open("wb") as fdst:
        with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
            while True:
                chunk = fsrc.read(16 * 1024 * 1024)
                if not chunk:
                    break
                fdst.write(chunk)
                bar.update(len(chunk))
    shutil.copystat(src, dst)


def _write_csv_with_progress(df: pd.DataFrame, path: Path, *, desc: str) -> None:
    total_rows = len(df)
    if total_rows == 0:
        df.to_csv(path, index=False)
        return

    # Large writes go through pyarrow's multithreaded CSV writer (~4.5x faster
    # than pandas to_csv; round-trip values/dtypes verified identical, string
    # fields come out RFC-4180-quoted). Any conversion problem (e.g. truly
    # mixed-type object columns) falls back to the classic pandas writer.
    if total_rows >= 200_000 and not _FORCE_C_ENGINE and not _pandas_is_cudf_proxy():
        try:
            import pyarrow as pa
            import pyarrow.csv as pacsv

            table = pa.Table.from_pandas(df, preserve_index=False)
            pacsv.write_csv(
                table,
                str(path),
                write_options=pacsv.WriteOptions(quoting_style="needed"),
            )
            return
        except Exception as exc:
            logger.debug(f"pyarrow CSV write fell back to pandas: {exc}")

    total_chunks = (total_rows + CSV_WRITE_CHUNK_ROWS - 1) // CSV_WRITE_CHUNK_ROWS
    with path.open("w", encoding="utf-8", newline="") as f:
        for start in tqdm(
            range(0, total_rows, CSV_WRITE_CHUNK_ROWS),
            total=total_chunks,
            desc=desc,
        ):
            df.iloc[start:start + CSV_WRITE_CHUNK_ROWS].to_csv(
                f,
                index=False,
                header=start == 0,
            )


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to a parquet state file (snappy compression).

    Parquet preserves dtypes natively, so the parquet state file round-trips
    int/float/bool/str/NaN columns exactly — no string round-trip / dtype
    re-inference like CSV. snappy keeps writes fast and files compact.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(path), compression="snappy")


def _write_state_body(df: pd.DataFrame, path: Path) -> None:
    """Serialize ``df`` to ``path`` dispatching on suffix (parquet vs CSV)."""
    if _is_parquet_path(path):
        _write_parquet(df, path)
    else:
        _write_csv_with_progress(df, path, desc=f"write_{path.name}")


def _atomic_write_csv_unlocked(df: pd.DataFrame, path: Path) -> None:
    """Atomically write ``df`` to ``path`` (tmp + hardlink .bak + fsync + rename).

    Format-agnostic: ``balalaika.csv`` writes CSV, ``balalaika.parquet`` writes
    snappy parquet. The atomicity machinery (tmp file, hardlink backup, fsync,
    rename) is identical for both.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    bak = Path(str(path) + ".bak")

    if path.exists():
        # Hardlink instead of byte-copy: after os.replace() swaps `path` to the
        # new inode, `bak` still references the previous generation — identical
        # backup semantics at O(1) cost instead of re-reading/writing the whole
        # multi-GB state file on every flush. Falls back to a copy on
        # filesystems without hardlink support.
        try:
            bak.unlink(missing_ok=True)
            os.link(path, bak)
        except OSError:
            _copy_file_with_progress(path, bak, desc=f"backup_{path.name}")

    _write_state_body(df, tmp)
    try:
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
    except OSError:
        pass
    os.replace(tmp, path)


def atomic_write_csv(df: pd.DataFrame, path: os.PathLike | str) -> None:
    """Write ``df`` to ``path`` atomically (tmp file + rename + fsync).

    A backup ``<path>.bak`` is kept so :func:`ensure_main_csv` can recover if
    the process is killed mid-write and leaves the CSV corrupt. The whole write
    is serialized with a per-CSV lock so periodic and final merges cannot race.
    """
    path = Path(path)
    with _csv_write_lock(path):
        _atomic_write_csv_unlocked(df, path)


# ---------------------------------------------------------------------------
# Main-CSV operations
# ---------------------------------------------------------------------------

def _normalize_filepath_column(df: pd.DataFrame, *, owned: bool = False) -> pd.DataFrame:
    """Return ``df`` with its ``filepath`` column normalized.

    When ``owned`` is False (the default, for frames an external caller may
    still alias) a copy is made before mutating — but only if normalization
    would actually change a value (:func:`_filepath_is_canonical`). Steady-state
    pipeline CSVs hold absolute, stripped paths, so the common case skips both
    the full-frame ``df.copy()`` and the per-row resolve pass entirely.

    When ``owned`` is True the caller guarantees no other reference to ``df``
    exists (it was just read inside the same lock), so the column is assigned
    in place — no defensive copy, ~one full frame less peak RAM per CSV touch.
    """
    if df is None or df.empty:
        return df
    if "filepath" in df.columns:
        if _filepath_is_canonical(df["filepath"]):
            return df  # nothing to do; no copy needed
        if not owned:
            df = df.copy()
        df["filepath"] = _normalize_path_series(df["filepath"])
    return df


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    base = [c for c in BASE_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in base]
    return df[base + extras]


# ---------------------------------------------------------------------------
# State-format interop (parquet mode)
# ---------------------------------------------------------------------------

def load_main_csv(podcasts_path: os.PathLike | str) -> pd.DataFrame:
    """Return the current pipeline state (or an empty DataFrame).

    Reads ``balalaika.parquet`` (the only state format).
    """
    with _csv_write_lock(state_path(podcasts_path)):
        df = _read_csv_safe(state_path(podcasts_path))
    if df is None:
        return pd.DataFrame(columns=["filepath"])
    if "filepath" not in df.columns:
        df["filepath"] = ""
    # Freshly read frame, no external alias yet -> normalize in place.
    return _normalize_filepath_column(df, owned=True)


def _normalize_state_dtypes_to_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a parquet-loaded frame to the dtypes a CSV read would yield.

    Parquet preserves pandas' nullable extension dtypes (``Int64`` / ``boolean``
    / ``string``) and Arrow-backed types, but every CSV consumer in the pipeline
    was written against the numpy-backed dtypes ``fast_read_csv`` produces:

    * nullable integer -> ``int64`` when no nulls, else ``float64`` (CSV loses
      integerness once a column has a hole, exactly like ``pd.read_csv``);
    * nullable boolean -> ``object`` with ``True`` / ``False`` / ``None``;
    * nullable / Arrow string -> ``object`` with ``None`` for missing.

    Operates in place on a frame the caller owns; returns it for chaining.
    """
    import pandas.api.types as ptypes

    if df is None or df.empty:
        return df
    for col in df.columns:
        s = df[col]
        dtype = s.dtype
        if not ptypes.is_extension_array_dtype(dtype):
            continue
        if ptypes.is_integer_dtype(dtype):
            df[col] = s.astype("int64") if not s.isna().any() else s.astype("float64")
        elif ptypes.is_bool_dtype(dtype):
            df[col] = s.astype(object).where(s.notna(), None)
        elif ptypes.is_string_dtype(dtype):
            df[col] = s.astype(object).where(s.notna(), None)
    return df


def read_state_dataframe(podcasts_path: os.PathLike | str) -> pd.DataFrame:
    """Read the active pipeline state for a late-stage consumer (read-only).

    Mirrors :func:`report.py`'s precedence: in parquet mode prefer
    ``balalaika.parquet`` when it exists, else fall back to the ``balalaika.csv``
    export; csv mode always reads the CSV. This is the canonical reader for
    stages that run *after* the direct upserters (``ensure_audio_durations`` &
    co.) which write only the parquet state without refreshing the CSV export —
    so the hardcoded CSV would be stale.

    Unlike :func:`load_main_csv` this never migrates or writes; the parquet frame
    is normalized to CSV-equivalent dtypes (see
    :func:`_normalize_state_dtypes_to_csv`) so consumers see the dtypes they
    always got from ``fast_read_csv``. Returns an empty (``filepath``-only) frame
    when no state file exists.
    """
    parquet = parquet_path(podcasts_path)
    if parquet.exists():
        df = _read_csv_safe(parquet)
        if df is not None:
            df = _normalize_state_dtypes_to_csv(df)
    else:
        df = None
    if df is None:
        return pd.DataFrame(columns=["filepath"])
    if "filepath" not in df.columns:
        df["filepath"] = ""
    # Freshly read frame, no external alias yet -> normalize in place.
    return _normalize_filepath_column(df, owned=True)


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
    target = state_path(podcasts_path)
    with _csv_write_lock(target):
        bak = Path(str(target) + ".bak")
        df = _read_csv_safe(target)

        if df is None or df.empty or "filepath" not in df.columns:
            bak_df = _read_csv_safe(bak)
            if bak_df is not None and not bak_df.empty and "filepath" in bak_df.columns:
                logger.warning(
                    f"{target.name} is corrupt — restored {len(bak_df)} rows from "
                    f"{bak.name}"
                )
                _atomic_write_csv_unlocked(bak_df, target)
                df = _normalize_filepath_column(bak_df, owned=True)
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
                paths = sorted(
                    set(
                        normalize_path_values(
                            audio_paths,
                            desc="bootstrap_audio_paths",
                            drop_empty=True,
                        )
                    )
                )
                df = pd.DataFrame({"filepath": paths})

            _atomic_write_csv_unlocked(df, target)
            return df

        return _normalize_filepath_column(df, owned=True)


def upsert_columns(
    podcasts_path: os.PathLike | str,
    results_df: pd.DataFrame,
    value_columns: Sequence[str],
    *,
    drop_missing_files: bool = False,
    bootstrap_audio_paths: Optional[Iterable[os.PathLike | str]] = None,
    preserve_existing: bool = True,
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
        preserve_existing: existing values outside ``results_df`` are always
            retained. When True, null incoming values also cannot erase a
            value already stored for the same filepath. When False, incoming
            rows replace matching values, including with null.

    Returns the resulting DataFrame after the atomic write.
    """
    target = state_path(podcasts_path)
    with _csv_write_lock(target):
        df = _read_csv_safe(target)
        if df is None or "filepath" not in df.columns:
            df = pd.DataFrame(columns=["filepath"])
        # `df` was just read inside the lock; no external alias -> in place.
        df = _normalize_filepath_column(df, owned=True)

        if bootstrap_audio_paths is not None:
            boot = pd.DataFrame(
                {
                    "filepath": normalize_path_values(
                        bootstrap_audio_paths,
                        desc="bootstrap_audio_paths",
                        drop_empty=True,
                    )
                }
            )
            boot = boot.drop_duplicates(subset="filepath")
            # Preserve any existing column values; only add brand-new rows.
            df = pd.concat([df, boot], ignore_index=True).drop_duplicates(
                subset="filepath", keep="first"
            )

        if results_df is not None and not results_df.empty:
            if "filepath" not in results_df.columns:
                raise ValueError("results_df must contain a 'filepath' column")
            # Normalize without an unconditional full-frame copy: the guard
            # copies only when a value actually changes, and the column slice
            # below produces a fresh frame we own regardless, so the caller's
            # results_df is never mutated.
            results = _normalize_filepath_column(results_df)
            present = [c for c in value_columns if c in results.columns]
            results = results[["filepath", *present]].drop_duplicates(
                subset="filepath", keep="last"
            )
            existing_columns = set(df.columns)
            incoming_marker = "__balalaika_incoming_row__"
            while incoming_marker in existing_columns or incoming_marker in results.columns:
                incoming_marker = f"_{incoming_marker}"
            results[incoming_marker] = True
            df = df.merge(
                results,
                on="filepath",
                how="outer",
                suffixes=("", "__incoming"),
            )
            incoming_rows = df[incoming_marker].eq(True)

            for col in present:
                if col not in existing_columns:
                    continue
                incoming_col = f"{col}__incoming"
                if preserve_existing:
                    df[col] = df[incoming_col].combine_first(df[col])
                else:
                    updated = df[col].astype(object)
                    updated.loc[incoming_rows] = df.loc[incoming_rows, incoming_col]
                    df[col] = updated.infer_objects(copy=False)
                df = df.drop(columns=[incoming_col])

            df = df.drop(columns=[incoming_marker])

        if drop_missing_files and not df.empty:
            before = len(df)
            existing_mask = _paths_exist_mask(
                df["filepath"].astype(str).tolist(),
                desc="check_existing_files",
            )
            df = df[existing_mask]
            removed = before - len(df)
            if removed:
                logger.info(
                    f"Pruned {removed} rows whose audio files no longer exist."
                )

        df = _reorder_columns(df)
        _atomic_write_csv_unlocked(df, target)
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
    logger.info(f"Loading main state to find unprocessed paths for column '{column}'.")
    df = None
    main_path = state_path(podcasts_path)
    # Parquet column projection is genuinely cheap (only the 2 needed column
    # chunks leave the disk); the CSV path sniffs the header first so a missing
    # column skips the body read entirely.
    header = _state_header(main_path)

    audio_resolved = normalize_path_values(
        audio_paths, desc=f"resolve_{column}_paths", drop_empty=True
    )

    if header is not None and "filepath" in header:
        if column not in header:
            logger.info(
                f"Column '{column}' is missing from the state header; "
                f"all {len(audio_resolved)} paths are pending."
            )
            return audio_resolved
        try:
            # Only the 2 needed columns: a full-width read of a production
            # state file holds every text/score column in RAM just to drop it.
            df = _read_state_narrow(main_path, ["filepath", column])
            df["filepath"] = _normalize_path_series(df["filepath"])
        except Exception as exc:
            logger.warning(f"Narrow state read failed ({exc}); falling back to full read.")
            df = None

    if df is None:
        df = load_main_csv(podcasts_path)

    if column not in df.columns or df.empty:
        logger.info(
            f"Column '{column}' is missing or CSV is empty; "
            f"all {len(audio_resolved)} paths are pending."
        )
        return audio_resolved

    logger.info(f"Building done set for column '{column}'.")
    done_mask = df[column].notna()
    # Blank / whitespace-only strings count as pending. Parquet returns string
    # columns as the pandas ``string`` dtype (not ``object``), so check both —
    # otherwise an empty-string "not done" marker would be read as done.
    if df[column].dtype == object or pd.api.types.is_string_dtype(df[column].dtype):
        done_mask &= df[column].astype(str).str.strip().ne("")
    done = set(df.loc[done_mask, "filepath"].tolist())

    pending = [path for path in audio_resolved if path not in done]

    logger.info(
        f"Column '{column}': {len(done)} done, {len(pending)} pending "
        f"out of {len(audio_resolved)} audio paths."
    )
    return pending


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
    for p in tqdm(parts, total=len(parts), desc=f"read_{prefix}_partials"):
        try:
            df = fast_read_csv(p)
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
        # Freshly concatenated frame -> normalize in place.
        merged = _normalize_filepath_column(merged, owned=True)
        merged = merged.drop_duplicates(subset="filepath", keep="last")
    return merged


def delete_partial_csvs(podcasts_path: os.PathLike | str, prefix: str) -> int:
    """Remove all ``<prefix>_part_*.csv`` files; return the number deleted."""
    deleted = 0
    parts = list_partial_csvs(podcasts_path, prefix)
    for p in tqdm(parts, total=len(parts), desc=f"delete_{prefix}_partials"):
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
    preserve_existing: bool = True,
) -> Tuple[pd.DataFrame, int]:
    """Merge any leftover partials into the main state and delete them.

    Returns ``(partials_df, rows_absorbed)``. ``partials_df`` is the raw
    concatenated partials (handy for stage audit accounting); the main state
    (``balalaika.parquet``) is updated only when there is something to merge or
    ``bootstrap_audio_paths`` is given.
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
        preserve_existing=preserve_existing,
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
        return set(
            normalize_path_values(
                df[key_column].astype(str).tolist(),
                desc=f"partial_{self.path.stem}_already_done",
                drop_empty=True,
            )
        )

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
# Periodic merger
# ---------------------------------------------------------------------------

def _count_partial_rows(podcasts_path: os.PathLike | str, prefix: str) -> int:
    """Count data rows across all ``<prefix>_part_*.csv`` (newline-based, no parsing).

    Header is excluded by subtracting one per non-empty file. This is the
    cheapest way to know "did the workers produce at least N more rows since
    last flush" — no pandas, no string parsing.
    """
    total = 0
    for p in list_partial_csvs(podcasts_path, prefix):
        try:
            n = 0
            last = b""
            with p.open("rb") as f:
                while True:
                    chunk = f.read(8 << 20)
                    if not chunk:
                        break
                    n += chunk.count(b"\n")
                    last = chunk[-1:]
            if last and last != b"\n":
                n += 1  # final line without trailing newline still counts
        except OSError:
            continue
        if n > 0:
            total += n - 1
    return total


def _partials_signature(podcasts_path: os.PathLike | str, prefix: str) -> Tuple[Tuple[str, int], ...]:
    """Cheap (name, size) fingerprint of all ``<prefix>_part_*.csv`` files.

    Partials are append-only (``PartialCsvWriter`` only ever appends), so the
    set of partial paths plus each one's byte size uniquely identifies "the
    workers have produced no further rows since the last observation". One
    ``stat`` per partial — no read, no parse. Used by the periodic merger to
    skip a full read+upsert+rewrite cycle that would reproduce a byte-identical
    ``balalaika.csv``.
    """
    sig: List[Tuple[str, int]] = []
    for p in list_partial_csvs(podcasts_path, prefix):
        try:
            sig.append((p.name, p.stat().st_size))
        except OSError:
            continue
    return tuple(sig)


class PeriodicCsvMerger:
    """Background-thread merger that keeps ``balalaika.csv`` fresh during a stage.

    Design goals (kept deliberately minimal):

    * One daemon thread in the main process.
    * Every ``poll_interval`` seconds, count the data rows on disk across all
      worker partials (a cheap byte-level newline count — no pandas).
    * Once the count has grown by ``flush_every_rows`` since the last flush
      (or ``flush_every_seconds`` elapsed), call the existing on-disk
      :func:`upsert_columns` exactly once. No in-memory mirror, no tail-byte
      reading — just one straightforward merge of the partials into
      ``balalaika.csv``.
    * Partials are never deleted by the merger; the post-stage
      :func:`absorb_partial_csvs` still owns cleanup. So losing the merger
      thread mid-flight cannot lose any data.

    The trade-off vs. the previous in-memory mirror is on purpose: re-reading
    ``balalaika.csv`` on each flush is a one-off pandas pass instead of a
    long-lived multi-GB RAM resident DataFrame. For a 10 000-row flush
    threshold this means at most one extra read per ~10 000 rows of progress,
    which dominates *much* less CPU/RAM than the previous design.
    """

    def __init__(
        self,
        podcasts_path: os.PathLike | str,
        prefix: str,
        value_columns: Sequence[str],
        *,
        flush_every_rows: int = DEFAULT_FLUSH_EVERY_ROWS,
        flush_every_seconds: float = DEFAULT_FLUSH_EVERY_SECONDS,
        drop_missing_files: bool = False,
        bootstrap_audio_paths: Optional[Iterable[os.PathLike | str]] = None,
        preserve_existing: bool = True,
        poll_interval: float = 30.0,
    ) -> None:
        self.podcasts_path = Path(podcasts_path)
        self.prefix = prefix
        self.value_columns = list(value_columns)
        self.flush_every_rows = max(0, int(flush_every_rows or 0))
        self.flush_every_seconds = max(0.0, float(flush_every_seconds or 0.0))
        self.drop_missing_files = drop_missing_files
        self.bootstrap_audio_paths = (
            list(bootstrap_audio_paths)
            if bootstrap_audio_paths is not None
            else None
        )
        self.preserve_existing = preserve_existing
        self.poll_interval = max(5.0, float(poll_interval))

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_flush_ts = 0.0
        self._last_flushed_rows = 0
        # (name, size) fingerprint of the partials at the last actual flush, so
        # a triggered flush that would re-merge byte-identical partials (no new
        # rows since last time) skips the full read+upsert+rewrite cycle.
        self._last_flushed_sig: Tuple[Tuple[str, int], ...] = ()
        self._enabled = self.flush_every_rows > 0 or self.flush_every_seconds > 0

    def _flush_once(self) -> int:
        """Read every partial in full and fold it into ``balalaika.csv``.

        Returns the number of partial rows merged (0 when there's nothing new).

        ``drop_missing_files`` is deliberately NOT applied here even when the
        caller requested it: pruning runs ``_paths_exist_mask`` over the whole
        CSV, i.e. one ``scandir`` per audio directory of the dataset, on every
        flush — a full metadata sweep of the HDD that competes with the
        stage's own audio reads. Every caller that prunes already does a final
        ``absorb_partial_csvs(drop_missing_files=True)`` after the stage, so
        deferring the prune leaves the final CSV identical; only mid-stage
        snapshots may briefly keep rows for files deleted during the run.
        """
        partials = read_partial_csvs(self.podcasts_path, self.prefix)
        if partials.empty and self.bootstrap_audio_paths is None:
            return 0
        upsert_columns(
            self.podcasts_path,
            partials,
            value_columns=self.value_columns,
            drop_missing_files=False,
            bootstrap_audio_paths=self.bootstrap_audio_paths,
            preserve_existing=self.preserve_existing,
        )
        return int(len(partials))

    def _loop(self) -> None:
        self._last_flush_ts = time.time()
        while not self._stop.wait(self.poll_interval):
            try:
                current_rows = _count_partial_rows(self.podcasts_path, self.prefix)
            except Exception as exc:
                logger.debug(f"Periodic merger: row count failed: {exc}")
                continue

            now = time.time()
            should_flush = False
            if (
                self.flush_every_rows > 0
                and current_rows - self._last_flushed_rows >= self.flush_every_rows
            ):
                should_flush = True
            elif (
                self.flush_every_seconds > 0
                and now - self._last_flush_ts >= self.flush_every_seconds
                and current_rows > self._last_flushed_rows
            ):
                should_flush = True

            if not should_flush:
                continue
            # Skip the whole read+upsert+rewrite when the partials are byte-for-
            # byte unchanged since the last flush: re-merging them would only
            # reproduce the current balalaika.csv. Append-only partials make a
            # (name, size) signature a sound "nothing new" test. (When a
            # bootstrap path list is configured the first flush must still run
            # to inject those rows, so only skip once a flush has happened.)
            sig = _partials_signature(self.podcasts_path, self.prefix)
            if self._last_flushed_sig and sig == self._last_flushed_sig:
                self._last_flush_ts = now
                self._last_flushed_rows = current_rows
                continue
            try:
                merged = self._flush_once()
            except Exception as exc:
                logger.warning(f"Periodic CSV flush failed: {exc}")
                continue
            self._last_flush_ts = now
            self._last_flushed_rows = current_rows
            self._last_flushed_sig = sig
            if merged:
                logger.info(
                    f"balalaika.csv refreshed: {merged} rows from "
                    f"{self.prefix}_part_*.csv folded in."
                )

    def __enter__(self) -> "PeriodicCsvMerger":
        if self._enabled:
            self._thread = threading.Thread(
                target=self._loop,
                name=f"csv-merger-{self.prefix}",
                daemon=True,
            )
            self._thread.start()
            logger.info(
                f"Periodic CSV merger: every {self.flush_every_rows} rows or "
                f"{self.flush_every_seconds}s (poll {self.poll_interval}s)."
            )
            if self.drop_missing_files:
                logger.info(
                    "Periodic CSV merger: missing-file pruning is deferred to "
                    "the final absorb (per-flush pruning would rescan every "
                    "audio directory)."
                )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            if self._thread.is_alive():
                logger.info(
                    "Periodic CSV merger stopping; waiting for any active "
                    "balalaika.csv flush to finish."
                )
            self._thread.join()


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

def _normalise_audio_paths_source(source: object) -> str:
    value = str(source or "auto").strip().lower().replace("-", "_")
    aliases = {
        "balalaika.csv": "csv",
        "balalaika_csv": "csv",
        "filesystem": "rglob",
        "fs": "rglob",
        "file_system": "rglob",
    }
    value = aliases.get(value, value)
    if value not in {"auto", "csv", "rglob"}:
        logger.warning(
            f"Unknown runtime.audio_paths_source={source!r}; using 'auto'. "
            "Expected one of: auto, csv, rglob."
        )
        return "auto"
    return value


def _runtime_audio_paths_source(config_path: Optional[os.PathLike | str]) -> str:
    if not config_path:
        return "rglob"
    try:
        from src.utils.runtime_env import runtime_cfg

        return _normalise_audio_paths_source(
            runtime_cfg(str(config_path)).get("audio_paths_source", "auto")
        )
    except Exception as exc:
        logger.warning(f"Could not read runtime.audio_paths_source: {exc}; using 'rglob'.")
        return "rglob"


def _path_suffix_lower(path: str) -> str:
    """``Path(path).suffix.lower()`` without the Path object cost."""
    name = os.path.basename(path)
    dot = name.rfind(".")
    if dot <= 0:  # no dot, or hidden file like '.wav' (no real suffix)
        return ""
    return name[dot:].lower()


def _dedupe_paths(paths: Iterable[os.PathLike | str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    append = out.append
    add = seen.add
    isabs = os.path.isabs
    for raw in paths:
        if raw is None:
            continue
        path = str(raw).strip()
        if not path:
            continue
        if _path_suffix_lower(path) not in AUDIO_EXTENSIONS:
            continue
        resolved = path if isabs(path) else resolve_path(path)
        if resolved not in seen:
            add(resolved)
            append(resolved)
    return out


def _audio_paths_from_csv(podcasts_path: os.PathLike | str) -> List[str]:
    # Load filepaths from the parquet state (column projection is cheap).
    target = state_path(podcasts_path)
    if not target.exists():
        logger.info(f"{target.name} not found; cannot load audio paths from state.")
        return []
    try:
        df = _read_state_narrow(target, ["filepath"])
    except (ValueError, pd.errors.EmptyDataError):
        logger.warning(f"{target.name} has no usable filepath column.")
        return []
    except Exception as exc:
        logger.warning(f"Could not load audio paths from {target}: {exc}")
        return []

    paths = _dedupe_paths(df["filepath"].dropna().astype(str))
    logger.info(f"Loaded {len(paths)} audio paths from {target.name}.")
    return paths


def _audio_paths_from_rglob(podcasts_path: os.PathLike | str) -> List[str]:
    from src.utils.utils import get_audio_paths

    paths = _dedupe_paths(get_audio_paths(str(podcasts_path)))
    logger.info(f"Discovered {len(paths)} audio paths via rglob.")
    return paths


def discover_audio_paths(
    podcasts_path: os.PathLike | str,
    *,
    config_path: Optional[os.PathLike | str] = None,
    source: Optional[str] = None,
) -> List[str]:
    """Resolve audio paths using runtime.audio_paths_source.

    Sources:
    * ``rglob``: scan the filesystem recursively.
    * ``csv``: trust ``balalaika.csv`` as the source of filepaths.
    * ``auto``: prefer ``balalaika.csv`` when populated, otherwise fall back to rglob.
    """
    selected = _normalise_audio_paths_source(source) if source is not None else _runtime_audio_paths_source(config_path)

    if selected in {"auto", "csv"}:
        paths = _audio_paths_from_csv(podcasts_path)
        if paths or selected == "csv":
            return paths
        logger.info("Falling back to rglob because balalaika.csv did not provide audio paths.")

    return _audio_paths_from_rglob(podcasts_path)


def files_in_csv(df: pd.DataFrame) -> Set[str]:
    if df is None or df.empty or "filepath" not in df.columns:
        return set()
    return set(
        normalize_path_values(
            df["filepath"].astype(str).tolist(),
            desc="files_in_csv",
            drop_empty=True,
        )
    )
