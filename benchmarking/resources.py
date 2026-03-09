from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .common import CLK_TCK, PAGE_SIZE, REPO_ROOT, summarize_numeric, utc_now


def build_runtime_library_paths() -> List[str]:
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.prefix) / "lib" / f"python{python_version}" / "site-packages"
    nvidia_root = site_packages / "nvidia"

    library_dirs: List[Path] = []
    for candidate in [
        nvidia_root / "cu13" / "lib",
        nvidia_root / "cublas" / "lib",
        nvidia_root / "cudnn" / "lib",
        nvidia_root / "cuda_runtime" / "lib",
        nvidia_root / "cuda_nvrtc" / "lib",
        nvidia_root / "cufft" / "lib",
        nvidia_root / "nvjitlink" / "lib",
        nvidia_root / "cusolver" / "lib",
        nvidia_root / "cusparse" / "lib",
        nvidia_root / "cusparselt" / "lib",
        nvidia_root / "nccl" / "lib",
        site_packages / "tensorrt_libs",
    ]:
        if candidate.exists():
            library_dirs.append(candidate)

    seen: set[str] = set()
    resolved: List[str] = []
    for path in library_dirs:
        path_str = str(path)
        if path_str in seen:
            continue
        seen.add(path_str)
        resolved.append(path_str)
    return resolved


def nvidia_smi_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    return env


def proc_root() -> Path:
    return Path("/proc")


def system_cpu_snapshot() -> Optional[tuple[int, int]]:
    try:
        first_line = (proc_root() / "stat").read_text(encoding="utf-8").splitlines()[0]
    except Exception:
        return None

    parts = first_line.split()
    if not parts or parts[0] != "cpu":
        return None

    values = [int(value) for value in parts[1:]]
    total = sum(values)
    idle = values[3] + values[4] if len(values) > 4 else values[3]
    return total, idle


def system_memory_snapshot() -> Dict[str, Optional[float]]:
    meminfo_path = proc_root() / "meminfo"
    values: Dict[str, int] = {}
    try:
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            key, _, raw_value = line.partition(":")
            parts = raw_value.strip().split()
            if not parts:
                continue
            values[key] = int(parts[0]) * 1024
    except Exception:
        return {"used_gb": None, "percent": None}

    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if not total or available is None:
        return {"used_gb": None, "percent": None}

    used = total - available
    return {
        "used_gb": used / (1024 ** 3),
        "percent": (used / total) * 100.0,
    }


def read_process_stat(pid: int) -> Optional[Dict[str, int]]:
    stat_path = proc_root() / str(pid) / "stat"
    statm_path = proc_root() / str(pid) / "statm"

    try:
        stat_text = stat_path.read_text(encoding="utf-8")
        statm_text = statm_path.read_text(encoding="utf-8")
    except Exception:
        return None

    right_paren = stat_text.rfind(")")
    if right_paren == -1:
        return None
    fields = stat_text[right_paren + 2 :].split()
    if len(fields) < 13:
        return None

    try:
        ppid = int(fields[1])
        utime = int(fields[11])
        stime = int(fields[12])
        resident_pages = int(statm_text.split()[1])
    except (IndexError, ValueError):
        return None

    return {
        "pid": pid,
        "ppid": ppid,
        "cpu_ticks": utime + stime,
        "rss_bytes": resident_pages * PAGE_SIZE,
    }


def process_tree_snapshot(root_pid: int) -> Dict[int, Dict[str, int]]:
    stats: Dict[int, Dict[str, int]] = {}
    children: Dict[int, List[int]] = {}

    for entry in proc_root().iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        stat = read_process_stat(pid)
        if not stat:
            continue
        stats[pid] = stat
        children.setdefault(stat["ppid"], []).append(pid)

    if root_pid not in stats:
        return {}

    result: Dict[int, Dict[str, int]] = {}
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in result:
            continue
        stat = stats.get(pid)
        if not stat:
            continue
        result[pid] = stat
        pending.extend(children.get(pid, []))

    return result


def query_gpu_inventory() -> Dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            env=nvidia_smi_env(),
            check=True,
        )
    except Exception as exc:
        return {"gpus": [], "error": str(exc)}

    gpus: List[Dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": float(parts[2]),
                }
            )
        except ValueError:
            continue
    return {"gpus": gpus, "error": None}


def filter_gpu_inventory(inventory: Dict[str, Any], gpu_ids: Optional[List[int]]) -> Dict[str, Any]:
    if gpu_ids is None:
        return inventory
    if gpu_ids == []:
        return {"gpus": [], "error": inventory.get("error")}
    gpus = [gpu for gpu in inventory.get("gpus", []) if gpu.get("index") in gpu_ids]
    return {"gpus": gpus, "error": inventory.get("error")}


class ResourceSampler:
    def __init__(self, pid: int, gpu_ids: Optional[List[int]], interval_sec: float) -> None:
        self.pid = pid
        self.gpu_ids = gpu_ids
        self.interval_sec = interval_sec
        self.samples: List[Dict[str, Any]] = []
        self.gpu_query_error: Optional[str] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cpu_count = max(os.cpu_count() or 1, 1)
        self._prev_process_cpu_ticks: Dict[int, int] = {}
        self._prev_process_sample_time: Optional[float] = None
        self._prev_system_cpu_snapshot: Optional[tuple[int, int]] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"resource-sampler-{self.pid}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=max(self.interval_sec * 2, 1.0))

    def _query_gpu(self) -> List[Dict[str, Any]]:
        if self.gpu_ids == []:
            return []
        command = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
                env=nvidia_smi_env(),
                check=True,
            )
        except Exception as exc:
            if self.gpu_query_error is None:
                self.gpu_query_error = str(exc)
            return []

        metrics: List[Dict[str, Any]] = []
        for line in result.stdout.splitlines():
            parts = [item.strip() for item in line.split(",")]
            if len(parts) != 4:
                continue
            try:
                index = int(parts[0])
                if self.gpu_ids is not None and index not in self.gpu_ids:
                    continue
                metrics.append(
                    {
                        "index": index,
                        "gpu_util_percent": float(parts[1]),
                        "memory_used_mb": float(parts[2]),
                        "memory_total_mb": float(parts[3]),
                    }
                )
            except ValueError:
                continue
        return metrics

    def _run(self) -> None:
        while not self._stop_event.is_set():
            now = time.monotonic()
            process_stats = process_tree_snapshot(self.pid)
            cpu_ticks_total = sum(stat["cpu_ticks"] for stat in process_stats.values())
            rss_bytes = sum(stat["rss_bytes"] for stat in process_stats.values())

            cpu_percent_raw = 0.0
            if self._prev_process_sample_time is not None:
                elapsed = now - self._prev_process_sample_time
                if elapsed > 0:
                    previous_total = sum(
                        self._prev_process_cpu_ticks.get(pid, stat["cpu_ticks"])
                        for pid, stat in process_stats.items()
                    )
                    delta_ticks = max(cpu_ticks_total - previous_total, 0)
                    cpu_percent_raw = (delta_ticks / (elapsed * CLK_TCK)) * 100.0

            self._prev_process_cpu_ticks = {pid: stat["cpu_ticks"] for pid, stat in process_stats.items()}
            self._prev_process_sample_time = now

            system_cpu_percent = 0.0
            current_system_cpu_snapshot = system_cpu_snapshot()
            if self._prev_system_cpu_snapshot and current_system_cpu_snapshot:
                prev_total, prev_idle = self._prev_system_cpu_snapshot
                total, idle = current_system_cpu_snapshot
                total_delta = total - prev_total
                idle_delta = idle - prev_idle
                if total_delta > 0:
                    system_cpu_percent = (1.0 - (idle_delta / total_delta)) * 100.0
            self._prev_system_cpu_snapshot = current_system_cpu_snapshot

            memory_snapshot = system_memory_snapshot()

            sample = {
                "timestamp_utc": utc_now(),
                "process_count": len(process_stats),
                "cpu_percent_raw": cpu_percent_raw,
                "cpu_util_percent": cpu_percent_raw / self._cpu_count,
                "cpu_cores_used": cpu_percent_raw / 100.0,
                "rss_bytes": rss_bytes,
                "rss_gb": rss_bytes / (1024 ** 3),
                "system_cpu_percent": system_cpu_percent,
                "system_ram_used_gb": memory_snapshot["used_gb"],
                "system_ram_percent": memory_snapshot["percent"],
                "gpus": self._query_gpu(),
            }
            self.samples.append(sample)
            self._stop_event.wait(self.interval_sec)


def summarize_gpu_samples(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    per_gpu: Dict[str, Dict[str, List[float]]] = {}
    all_utils: List[float] = []
    all_vram_mb: List[float] = []

    for sample in samples:
        for gpu in sample.get("gpus", []):
            gpu_key = str(gpu["index"])
            bucket = per_gpu.setdefault(gpu_key, {"gpu_util_percent": [], "memory_used_mb": []})
            bucket["gpu_util_percent"].append(gpu["gpu_util_percent"])
            bucket["memory_used_mb"].append(gpu["memory_used_mb"])
            all_utils.append(gpu["gpu_util_percent"])
            all_vram_mb.append(gpu["memory_used_mb"])

    return {
        "gpu_util_percent": summarize_numeric(all_utils),
        "gpu_vram_mb": summarize_numeric(all_vram_mb),
        "per_gpu": {
            gpu_key: {
                "gpu_util_percent": summarize_numeric(values["gpu_util_percent"]),
                "memory_used_mb": summarize_numeric(values["memory_used_mb"]),
            }
            for gpu_key, values in per_gpu.items()
        },
    }


def summarize_resource_samples(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "sample_count": len(samples),
        "cpu_util_percent": summarize_numeric(sample["cpu_util_percent"] for sample in samples),
        "cpu_cores_used": summarize_numeric(sample["cpu_cores_used"] for sample in samples),
        "rss_gb": summarize_numeric(sample["rss_gb"] for sample in samples),
        "system_cpu_percent": summarize_numeric(sample["system_cpu_percent"] for sample in samples),
        "system_ram_used_gb": summarize_numeric(sample["system_ram_used_gb"] for sample in samples),
        "system_ram_percent": summarize_numeric(sample["system_ram_percent"] for sample in samples),
    }
    summary.update(summarize_gpu_samples(samples))
    return summary
