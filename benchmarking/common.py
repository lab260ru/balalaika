from __future__ import annotations

import os
import shutil
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRANSCRIPTION_MODELS: tuple[str, ...] = (
    "giga_ctc",
    "giga_rnnt",
    "giga_ctc_lm",
    "tone",
    "vosk",
    "vosk_small",
    "parakeet_v2",
    "parakeet_v3",
    "canary",
    "whisper_base",
    "whisper_turbo",
)

COLLATE_SIDECARS: tuple[str, ...] = (
    "_rover.txt",
    "_punct.txt",
    "_accent.txt",
    "_rover_phonemes.txt",
)

CLK_TCK = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def preserve_repeat_artifacts(dataset_root: Path, repeat_root: Path) -> List[str]:
    artifacts_dir = repeat_root / "artifacts"
    preserved_paths: List[str] = []

    for filename in ("balalaika.csv", "balalaika.parquet"):
        source_path = dataset_root / filename
        if not source_path.exists():
            continue
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        destination_path = artifacts_dir / filename
        shutil.copy2(source_path, destination_path)
        preserved_paths.append(str(destination_path))

    return preserved_paths


def ensure_dict(mapping: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        value = {}
        mapping[key] = value
    return value


def load_full_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {config_path} must be a YAML mapping")
    return data


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def run_git(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def get_audio_duration(path: Path) -> float:
    try:
        import torchaudio

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = torchaudio.info(str(path))
        if info.sample_rate > 0 and info.num_frames > 0:
            return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        pass

    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate > 0 and info.frames > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass

    raise RuntimeError(f"Unable to determine duration for {path}")


def sidecar_path(audio_path: Path, suffix: str) -> Path:
    return audio_path.with_name(f"{audio_path.stem}{suffix}")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def summarize_numeric(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return {"avg": None, "max": None, "min": None}
    return {
        "avg": sum(filtered) / len(filtered),
        "max": max(filtered),
        "min": min(filtered),
    }


def parse_gpu_ids(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    values: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values or None
