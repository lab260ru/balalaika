"""One JSON sidecar per chunk: ``<stem>.json`` next to the audio.

Replaces the per-stage ``*_<model>.txt`` / ``*_<model>.tst`` / ``*_rover.txt`` /
``*_punct.txt`` / ``*_accent.txt`` / ``*_rover_phonemes.txt`` files with a
single JSON document per chunk. Stages write their own key(s) incrementally; a
missing or empty key means "that stage has not produced output for this chunk
yet", which is how resume/skip stays identical to the old per-file existence
checks.

Schema (all keys optional, written by their producing stage):

    {
      "asr":    {"<model>": "<text>", ...},      # stage 8 per-model ASR
      "asr_ts": {"<model>": "<timestamps>", ...},# stage 8, when with_timestamps
      "rover":  "<text>",                         # stage 8 ROVER consensus
      "punct":  "<text>",                         # stage 9
      "accent": "<text>",                         # stage 10
      "rover_phonemes": "<text>"                  # stage 11
    }

Chunk *metadata* (start/end/duration/scores) lives in the parquet state, not
here — no duplication.

**Concurrency.** Stages run sequentially and each stage's workers process
disjoint work shards, so no two processes ever write the same chunk JSON at the
same time. :func:`update_chunk_json` is a read-modify-write guarded by an atomic
``tmp + os.replace`` so a crash can never leave a truncated/partial document and
a re-run simply re-reads the last complete version.
"""
from __future__ import annotations

import errno
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.utils.csv_manager import discover_audio_paths
from src.utils.utils import get_audio_paths

JSON_SUFFIX = ".json"


def chunk_json_path(audio_path: Path | str) -> Path:
    """``foo.flac`` -> ``foo.json`` next to the audio (one extension stripped)."""
    p = audio_path if isinstance(audio_path, Path) else Path(audio_path)
    return p.with_name(f"{p.stem}{JSON_SUFFIX}")


def read_chunk_json(path: Path | str) -> Dict[str, Any]:
    """Parse a chunk JSON, returning ``{}`` if missing or corrupt.

    Tolerates a truncated/corrupt file (e.g. a write killed before this module
    existed) rather than crashing a worker — the chunk is then simply treated
    as not-yet-processed and reprocessed.
    """
    p = path if isinstance(path, Path) else Path(path)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        if exc.errno == errno.ENAMETOOLONG:
            return {}
        raise
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning(f"Corrupt chunk JSON, treating as empty: {p}")
        return {}
    return data if isinstance(data, dict) else {}


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``updates`` into ``base`` in place; nested dicts merge key-wise."""
    for key, value in updates.items():
        if (
            isinstance(value, dict)
            and isinstance(base.get(key), dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """tmp + ``os.replace`` so a killed worker never publishes a partial file.

    The staging file is a UNIQUE ``tempfile.mkstemp`` name in the destination
    directory (cleaned up on error), mirroring the discipline of the old
    ``transcription._write_text_atomic`` peer writers.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def update_chunk_json(audio_path: Path | str, updates: Dict[str, Any]) -> None:
    """Atomic read-modify-write: deep-merge ``updates`` into ``<stem>.json``."""
    path = chunk_json_path(audio_path)
    data = read_chunk_json(path)
    _deep_merge(data, updates)
    _atomic_write_json(path, data)


def get_field(data: Dict[str, Any], dotted_key: str) -> Any:
    """Read a nested value by dotted path (``"asr.giga_ctc"``); ``None`` if absent."""
    cur: Any = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _value_complete(value: Any, *, retry_empty: bool) -> bool:
    """A field is complete iff present and (not ``retry_empty`` or non-empty)."""
    if value is None:
        return False
    if retry_empty and isinstance(value, str):
        return value != ""
    return True


def field_complete(
    audio_path: Path | str,
    dotted_key: str,
    *,
    retry_empty: bool = False,
) -> bool:
    """Whether ``<stem>.json`` already holds ``dotted_key`` (resume check)."""
    data = read_chunk_json(chunk_json_path(audio_path))
    return _value_complete(get_field(data, dotted_key), retry_empty=retry_empty)


class ChunkJsonCache:
    """Existence + parse cache for chunk JSONs, the analogue of ``DirNameCache``.

    One ``os.scandir`` per directory answers "does ``<stem>.json`` exist" without
    a stat per file, and each JSON is parsed at most once (memoised per path).
    Pending scans over millions of chunks therefore do one directory listing per
    directory and one small read per existing JSON, instead of the old
    O(#sidecars) existence probes.

    A JSON name whose byte length exceeds the directory's ``NAME_MAX`` can never
    appear in a listing, so — matching the old per-file ``ENAMETOOLONG``
    handling — it is treated as already complete (skip once, forever) and warned
    about once.

    Not picklable/shareable across processes — every worker builds its own; the
    per-directory scandir still amortizes across all the files a shard touches in
    that directory.
    """

    _MISSING = object()
    _DEFAULT_NAME_MAX = 255

    def __init__(self) -> None:
        self._names: Dict[str, set] = {}
        self._parsed: Dict[str, Dict[str, Any]] = {}
        self._name_max: Dict[str, int] = {}
        self._warned_too_long: set = set()

    def _dir_names(self, d: str) -> set:
        cached = self._names.get(d)
        if cached is not None:
            return cached
        present: set = set()
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
            pass
        self._names[d] = present
        return present

    def _name_too_long(self, d: str, name: str) -> bool:
        limit = self._name_max.get(d)
        if limit is None:
            try:
                limit = os.pathconf(d, "PC_NAME_MAX")
            except (OSError, ValueError, AttributeError):
                limit = self._DEFAULT_NAME_MAX
            self._name_max[d] = limit
        if len(os.fsencode(name)) <= limit:
            return False
        full = os.path.join(d, name)
        if full not in self._warned_too_long:
            self._warned_too_long.add(full)
            logger.warning(
                f"Chunk JSON name exceeds NAME_MAX; treating as complete / "
                f"skipping: {full}"
            )
        return True

    def json_exists(self, audio_path: Path | str) -> bool:
        jp = chunk_json_path(audio_path)
        d, name = os.path.split(str(jp))
        if self._name_too_long(d, name):
            return True
        return name in self._dir_names(d)

    def get(self, audio_path: Path | str) -> Dict[str, Any]:
        """Parsed JSON for ``audio_path`` (``{}`` if missing/corrupt), memoised."""
        jp = chunk_json_path(audio_path)
        key = str(jp)
        cached = self._parsed.get(key, self._MISSING)
        if cached is not self._MISSING:
            return cached  # type: ignore[return-value]
        data = read_chunk_json(jp)
        self._parsed[key] = data
        return data

    def field_complete(
        self,
        audio_path: Path | str,
        dotted_key: str,
        *,
        retry_empty: bool = False,
    ) -> bool:
        jp = chunk_json_path(audio_path)
        d, name = os.path.split(str(jp))
        if self._name_too_long(d, name):
            return True
        if name not in self._dir_names(d):
            return False
        return _value_complete(
            get_field(self.get(audio_path), dotted_key), retry_empty=retry_empty
        )


def pending_chunks(
    podcasts_path: str | Path,
    *,
    out_field: str,
    in_field: Optional[str] = None,
    config_path: str | Path | None = None,
    retry_empty: bool = False,
) -> List[Path]:
    """Audio paths whose chunk JSON still needs ``out_field``.

    With ``in_field`` set, an audio is only pending if its JSON already holds
    ``in_field`` (the upstream stage's output) — the JSON analogue of the old
    "input sidecar exists, output sidecar missing" chain. Returns **audio**
    paths; the stage derives the JSON with :func:`chunk_json_path`.
    """
    audio = (
        discover_audio_paths(podcasts_path, config_path=config_path)
        if config_path
        else get_audio_paths(str(podcasts_path))
    )
    cache = ChunkJsonCache()
    out: List[Path] = []
    for raw in audio:
        a = Path(raw)
        if in_field is not None and not cache.field_complete(
            a, in_field, retry_empty=retry_empty
        ):
            continue
        if cache.field_complete(a, out_field, retry_empty=retry_empty):
            continue
        out.append(a)
    return out
