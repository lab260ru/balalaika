"""Detect whether a dataset directory lives on a rotational (HDD) disk and
derive I/O-concurrency bounds from it.

A single spindle serves ~100 random IOPS; every additional process that
reads audio concurrently multiplies seek distance instead of throughput.
Stages therefore clamp their DataLoader / probe-pool worker counts when the
dataset resolves to a rotational device. SSDs keep the configured values.

Resolution order for the profile (highest priority first):

1. ``$BALALAIKA_IO_PROFILE`` (``hdd``/``ssd``),
2. ``runtime.io_profile: hdd|ssd`` in the YAML config — read through
   :func:`src.utils.runtime_env.runtime_cfg`, the single source of truth for
   the ``runtime`` block (env-over-YAML precedence + caching),
3. ``auto``: sysfs rotational flag of the device backing the dataset path
   (partitions and device-mapper/LVM stacks are walked to the physical
   disks; a stack containing any rotational member counts as ``hdd``),
4. unknown hardware (containers without sysfs, network mounts): ``ssd``,
   i.e. no behavior change.

``auto`` / empty at either the env or YAML layer means "fall through to the
next source"; only ``hdd``/``ssd`` short-circuit the chain.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from loguru import logger

from src.utils.runtime_env import runtime_cfg

IO_PROFILE_ENV = "BALALAIKA_IO_PROFILE"
CONFIG_PATH_ENV = "BALALAIKA_CONFIG_PATH"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"

# Conservative concurrency caps for one spindle. Loader workers keep some
# parallelism because consecutive batches read path-adjacent files (short
# seeks) and decode is CPU-heavy; metadata/header probes gain nothing from
# fan-out and are capped harder.
HDD_MAX_LOADER_WORKERS = 4
HDD_MAX_PROBE_WORKERS = 2

_VALID_PROFILES = {"hdd", "ssd"}


def _read_rotational(disk_sys_path: Path) -> Optional[bool]:
    try:
        raw = (disk_sys_path / "queue" / "rotational").read_text().strip()
    except OSError:
        return None
    if raw == "1":
        return True
    if raw == "0":
        return False
    return None


def _rotational_for_sys_block(sys_path: Path, depth: int = 0) -> Optional[bool]:
    """Rotational flag for a resolved ``/sys/.../block/<name>`` node."""
    if depth > 8:  # cycle guard for pathological dm stacks
        return None

    direct = _read_rotational(sys_path)

    slaves_dir = sys_path / "slaves"
    slave_flags = []
    try:
        slave_names = sorted(os.listdir(slaves_dir))
    except OSError:
        slave_names = []
    for name in slave_names:
        flag = _rotational_for_sys_block(Path("/sys/class/block") / name, depth + 1)
        if flag is not None:
            slave_flags.append(flag)
    if slave_flags:
        # dm/LVM/RAID: pessimistically rotational if any member is.
        return any(slave_flags)

    if direct is not None:
        return direct

    # Partition nodes have no queue/ of their own; the parent is the disk.
    parent = sys_path.parent
    if (parent / "queue").is_dir():
        return _read_rotational(parent)
    return None


@lru_cache(maxsize=None)
def is_rotational(path: str) -> Optional[bool]:
    """Best-effort rotational flag for the device backing ``path``.

    Returns ``None`` when sysfs cannot answer (non-Linux, containers,
    network filesystems).
    """
    try:
        st = os.stat(path)
        node = Path(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}")
        return _rotational_for_sys_block(node.resolve())
    except OSError:
        return None


def _configured_io_profile() -> Optional[str]:
    """``runtime.io_profile`` from the YAML config, or ``None`` if unset.

    Reads through :func:`runtime_cfg` (the single source of truth for the
    ``runtime`` block, with its own caching) rather than re-parsing the YAML
    here. The config path follows the ``BALALAIKA_CONFIG_PATH`` env var the
    shell bootstrap exports, defaulting to ``configs/config.yaml``; a missing
    file yields the ``"auto"`` default, i.e. no behavior change.
    """
    config_path = os.environ.get(CONFIG_PATH_ENV) or str(DEFAULT_CONFIG_PATH)
    return runtime_cfg(config_path).get("io_profile")


@lru_cache(maxsize=None)
def resolve_io_profile(dataset_path: str, configured: Optional[str] = None) -> str:
    """Return ``"hdd"`` or ``"ssd"`` for ``dataset_path``.

    ``configured`` is an explicit per-call override; when it is ``None`` the
    YAML ``runtime.io_profile`` is consulted, so the documented knob takes
    effect for every stage without threading the config path through each
    caller. Precedence: ``configured`` / ``$BALALAIKA_IO_PROFILE`` > YAML
    ``runtime.io_profile`` > sysfs auto-detect.
    """
    for source, value in (
        ("explicit override", configured),
        (IO_PROFILE_ENV, os.environ.get(IO_PROFILE_ENV)),
        ("config runtime.io_profile", _configured_io_profile()),
    ):
        if value is None:
            continue
        value = str(value).strip().lower()
        if value in _VALID_PROFILES:
            logger.info(f"I/O profile for {dataset_path}: {value} (from {source}).")
            return value
        if value not in ("", "auto"):
            logger.warning(f"Ignoring unknown I/O profile {value!r} from {source}.")

    rotational = is_rotational(str(dataset_path))
    if rotational is None:
        logger.debug(f"I/O profile for {dataset_path}: device type unknown, assuming ssd.")
        return "ssd"
    profile = "hdd" if rotational else "ssd"
    logger.info(f"I/O profile for {dataset_path}: {profile} (auto-detected).")
    return profile


def clamp_loader_workers(num_workers: int, file_paths) -> int:
    """Clamp DataLoader workers for the disk that actually holds the files.

    Anchors detection on the first file's directory so every stage gets the
    right answer even when datasets and code live on different devices.
    """
    if num_workers <= 0 or not file_paths:
        return int(num_workers)
    anchor = os.path.dirname(str(file_paths[0])) or "."
    return effective_workers(num_workers, resolve_io_profile(anchor), role="loader")


def effective_workers(configured: int, profile: str, *, role: str = "loader") -> int:
    """Clamp a worker count for the given profile.

    ``role`` is ``"loader"`` (DataLoader / decode workers) or ``"probe"``
    (metadata/header scans). Worker counts never increase; on ``ssd`` the
    configured value passes through untouched.
    """
    configured = int(configured)
    if profile != "hdd":
        return configured
    cap = HDD_MAX_PROBE_WORKERS if role == "probe" else HDD_MAX_LOADER_WORKERS
    if configured > cap:
        logger.info(
            f"HDD I/O profile: clamping {role} workers {configured} -> {cap} "
            "(set runtime.io_profile: ssd or BALALAIKA_IO_PROFILE=ssd to disable)."
        )
        return cap
    return configured
