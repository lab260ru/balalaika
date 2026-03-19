from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeAlias


@dataclass(frozen=True)
class SampleRecord:
    audio_path: str
    relative_path: str
    duration_sec: float
    copied_sidecars: tuple[str, ...]


@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: tuple[str, ...]


ConfigMutator: TypeAlias = Callable[[Dict[str, Any], argparse.Namespace, Path], None]


@dataclass(frozen=True)
class TargetSpec:
    name: str
    description: str
    modules: tuple[str, ...]
    required_sidecars: tuple[str, ...] = ()
    copied_sidecars: tuple[str, ...] = ()
    mutator: Optional[ConfigMutator] = None
    min_input_duration_from_config: Optional[tuple[str, ...]] = None
    uses_gpu: bool = True
