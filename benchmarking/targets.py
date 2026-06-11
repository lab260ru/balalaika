from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .common import COLLATE_SIDECARS, TRANSCRIPTION_MODELS, ensure_dict
from .models import CommandSpec, ConfigMutator, TargetSpec


def build_commands(modules: Sequence[str]) -> List[CommandSpec]:
    return [
        CommandSpec(
            name=module,
            argv=(sys.executable, "-m", module, "--config_path"),
        )
        for module in modules
    ]


def base_mutator(config: Dict[str, Any], _: argparse.Namespace, work_dataset_path: Path) -> None:
    config["cache_path"] = str(work_dataset_path.parent / "cache")


def set_dataset_everywhere(config: Dict[str, Any], work_dataset_path: Path) -> None:
    dataset_path = str(work_dataset_path)
    for key in ("download", "preprocess", "separation", "transcription", "punctuation", "accent", "phonemizer"):
        section = ensure_dict(config, key)
        section["podcasts_path"] = dataset_path


def patch_preprocess(config: Dict[str, Any], args: argparse.Namespace, work_dataset_path: Path) -> None:
    base_mutator(config, args, work_dataset_path)
    section = ensure_dict(config, "preprocess")
    section["podcasts_path"] = str(work_dataset_path)
    if args.cpu_workers_total is not None:
        section["num_workers"] = args.cpu_workers_total
    elif args.cpu_workers_per_gpu is not None:
        section["num_workers"] = args.cpu_workers_per_gpu


def patch_separation_common(config: Dict[str, Any], args: argparse.Namespace, work_dataset_path: Path) -> Dict[str, Any]:
    base_mutator(config, args, work_dataset_path)
    section = ensure_dict(config, "separation")
    section["podcasts_path"] = str(work_dataset_path)
    section["cache_path"] = str(work_dataset_path.parent / "cache")
    return section


def patch_separation_music_detect(config: Dict[str, Any], args: argparse.Namespace, work_dataset_path: Path) -> None:
    section = patch_separation_common(config, args, work_dataset_path)
    music_detect = ensure_dict(section, "music_detect")
    music_detect["cache_path"] = str(work_dataset_path.parent / "cache")
    if args.cpu_workers_per_gpu is not None:
        music_detect["num_workers"] = args.cpu_workers_per_gpu
    if args.batch_size_override is not None:
        music_detect["bs"] = args.batch_size_override


def patch_separation_nisqa(config: Dict[str, Any], args: argparse.Namespace, work_dataset_path: Path) -> None:
    section = patch_separation_common(config, args, work_dataset_path)
    nisqa = ensure_dict(section, "nisqa")
    if args.batch_size_override is not None:
        nisqa["bs"] = args.batch_size_override
    if args.cpu_workers_per_gpu is not None:
        nisqa["num_workers"] = args.cpu_workers_per_gpu
    section["bs"] = nisqa.get("bs", 32)
    section["num_workers_nisqa"] = nisqa.get("num_workers", 4)
    section["nisqa_config_path"] = nisqa.get("nisqa_config_path", "./configs/nisqa_b.yaml")


def patch_separation_distillmos(config: Dict[str, Any], args: argparse.Namespace, work_dataset_path: Path) -> None:
    patch_separation_common(config, args, work_dataset_path)


def patch_separation_diarization(config: Dict[str, Any], args: argparse.Namespace, work_dataset_path: Path) -> None:
    section = patch_separation_common(config, args, work_dataset_path)
    diarization = ensure_dict(section, "diarization")
    if args.disable_diarization:
        diarization["enabled"] = False
    if args.cpu_workers_per_gpu is not None:
        diarization["num_workers"] = args.cpu_workers_per_gpu


def patch_separation_silence(config: Dict[str, Any], args: argparse.Namespace, work_dataset_path: Path) -> None:
    section = patch_separation_common(config, args, work_dataset_path)
    silence_detect = ensure_dict(section, "silence_detect")
    if args.cpu_workers_per_gpu is not None:
        silence_detect["num_workers"] = args.cpu_workers_per_gpu


def patch_separation_stage(config: Dict[str, Any], args: argparse.Namespace, work_dataset_path: Path) -> None:
    patch_separation_common(config, args, work_dataset_path)
    patch_separation_music_detect(config, args, work_dataset_path)
    patch_separation_nisqa(config, args, work_dataset_path)
    patch_separation_diarization(config, args, work_dataset_path)
    patch_separation_silence(config, args, work_dataset_path)


def transcription_batch_section(model_name: str) -> str:
    if "giga" in model_name:
        return "giga"
    if "vosk" in model_name:
        return "vosk"
    return model_name


def patch_transcription_common(config: Dict[str, Any], args: argparse.Namespace, work_dataset_path: Path) -> Dict[str, Any]:
    base_mutator(config, args, work_dataset_path)
    section = ensure_dict(config, "transcription")
    section["podcasts_path"] = str(work_dataset_path)
    return section


def patch_transcription_stage(config: Dict[str, Any], args: argparse.Namespace, work_dataset_path: Path) -> None:
    section = patch_transcription_common(config, args, work_dataset_path)
    if args.batch_size_override is not None:
        for model_name in section.get("model_names", []):
            model_section = ensure_dict(section, transcription_batch_section(str(model_name)))
            model_section["batch_size"] = args.batch_size_override


def patch_transcription_model(model_name: str) -> ConfigMutator:
    def mutator(config: Dict[str, Any], args: argparse.Namespace, work_dataset_path: Path) -> None:
        section = patch_transcription_common(config, args, work_dataset_path)
        section["model_names"] = [model_name]
        section["consensus_num"] = 0
        section["use_rover"] = False
        if args.batch_size_override is not None:
            model_section = ensure_dict(section, transcription_batch_section(model_name))
            model_section["batch_size"] = args.batch_size_override

    return mutator


def patch_punctuation(config: Dict[str, Any], args: argparse.Namespace, work_dataset_path: Path) -> None:
    base_mutator(config, args, work_dataset_path)
    section = ensure_dict(config, "punctuation")
    section["podcasts_path"] = str(work_dataset_path)
    if args.cpu_workers_per_gpu is not None:
        section["num_workers"] = args.cpu_workers_per_gpu
    if args.model_name_override:
        section["model_name"] = args.model_name_override


def patch_accent(config: Dict[str, Any], args: argparse.Namespace, work_dataset_path: Path) -> None:
    base_mutator(config, args, work_dataset_path)
    section = ensure_dict(config, "accent")
    section["podcasts_path"] = str(work_dataset_path)
    if args.cpu_workers_per_gpu is not None:
        section["num_workers"] = args.cpu_workers_per_gpu
    if args.model_name_override:
        section["model_name"] = args.model_name_override


def patch_phonemizer(config: Dict[str, Any], args: argparse.Namespace, work_dataset_path: Path) -> None:
    base_mutator(config, args, work_dataset_path)
    section = ensure_dict(config, "phonemizer")
    section["podcasts_path"] = str(work_dataset_path)
    if args.cpu_workers_per_gpu is not None:
        section["num_workers"] = args.cpu_workers_per_gpu


def patch_collate(config: Dict[str, Any], args: argparse.Namespace, work_dataset_path: Path) -> None:
    base_mutator(config, args, work_dataset_path)
    section = ensure_dict(config, "download")
    section["podcasts_path"] = str(work_dataset_path)
    if args.cpu_workers_total is not None:
        section["num_workers"] = args.cpu_workers_total
    elif args.cpu_workers_per_gpu is not None:
        section["num_workers"] = args.cpu_workers_per_gpu


def patch_pipeline(config: Dict[str, Any], args: argparse.Namespace, work_dataset_path: Path) -> None:
    set_dataset_everywhere(config, work_dataset_path)
    patch_preprocess(config, args, work_dataset_path)
    patch_separation_stage(config, args, work_dataset_path)
    patch_transcription_stage(config, args, work_dataset_path)
    patch_punctuation(config, args, work_dataset_path)
    patch_accent(config, args, work_dataset_path)
    patch_phonemizer(config, args, work_dataset_path)
    patch_collate(config, args, work_dataset_path)


TARGETS: Dict[str, TargetSpec] = {
    "preprocess.stage": TargetSpec(
        name="preprocess.stage",
        description="Full preprocess sequence: VAD chunking, crest-factor filter, loudness normalization.",
        modules=(
            "src.preprocess.preprocess",
            "src.preprocess.crest_factor_remover",
            "src.preprocess.preprocess_audio",
        ),
        mutator=patch_preprocess,
        uses_gpu=True,
    ),
    "preprocess.vad": TargetSpec(
        name="preprocess.vad",
        description="Smart-turn VAD chunking only.",
        modules=("src.preprocess.preprocess",),
        mutator=patch_preprocess,
        uses_gpu=True,
    ),
    "preprocess.crest_factor": TargetSpec(
        name="preprocess.crest_factor",
        description="Crest-factor filtering only.",
        modules=("src.preprocess.crest_factor_remover",),
        mutator=patch_preprocess,
        uses_gpu=False,
    ),
    "preprocess.audio_normalize": TargetSpec(
        name="preprocess.audio_normalize",
        description="Loudness normalization only.",
        modules=("src.preprocess.preprocess_audio",),
        mutator=patch_preprocess,
        uses_gpu=False,
    ),
    "separation.stage": TargetSpec(
        name="separation.stage",
        description="Full separation sequence: music_detect, NISQA, DistillMOS, diarization, silence_detect.",
        modules=(
            "src.separation.music_detect",
            "src.separation.nisqa_process",
            "src.separation.distillmos_process",
            "src.separation.diarization",
            "src.separation.silence_detect",
        ),
        mutator=patch_separation_stage,
        uses_gpu=True,
    ),
    "separation.music_detect": TargetSpec(
        name="separation.music_detect",
        description="Music detection model only.",
        modules=("src.separation.music_detect",),
        mutator=patch_separation_music_detect,
        uses_gpu=True,
    ),
    "separation.nisqa": TargetSpec(
        name="separation.nisqa",
        description="NISQA MOS model only.",
        modules=("src.separation.nisqa_process",),
        mutator=patch_separation_nisqa,
        uses_gpu=True,
    ),
    "separation.distillmos": TargetSpec(
        name="separation.distillmos",
        description="DistillMOS model only.",
        modules=("src.separation.distillmos_process",),
        mutator=patch_separation_distillmos,
        uses_gpu=True,
    ),
    "separation.diarization": TargetSpec(
        name="separation.diarization",
        description="Pyannote diarization only.",
        modules=("src.separation.diarization",),
        mutator=patch_separation_diarization,
        uses_gpu=True,
    ),
    "separation.silence_detect": TargetSpec(
        name="separation.silence_detect",
        description="Silero silence metrics only.",
        modules=("src.separation.silence_detect",),
        mutator=patch_separation_silence,
        uses_gpu=True,
    ),
    "transcription.stage": TargetSpec(
        name="transcription.stage",
        description="Full transcription stage using model_names from config.",
        modules=("src.transcription.transcription",),
        mutator=patch_transcription_stage,
        uses_gpu=True,
    ),
    "punctuation.stage": TargetSpec(
        name="punctuation.stage",
        description="Punctuation restoration stage.",
        modules=("src.punctuation.punctuation",),
        required_sidecars=("_rover.txt",),
        copied_sidecars=("_rover.txt",),
        mutator=patch_punctuation,
        uses_gpu=True,
    ),
    "accents.stage": TargetSpec(
        name="accents.stage",
        description="Accent restoration stage.",
        modules=("src.accents.accents",),
        required_sidecars=("_punct.txt",),
        copied_sidecars=("_punct.txt",),
        mutator=patch_accent,
        uses_gpu=True,
    ),
    "phonemizer.stage": TargetSpec(
        name="phonemizer.stage",
        description="Phonemizer stage.",
        modules=("src.phonemizer.phonemizer",),
        required_sidecars=("_rover.txt",),
        copied_sidecars=("_rover.txt",),
        mutator=patch_phonemizer,
        uses_gpu=True,
    ),
    "collate.stage": TargetSpec(
        name="collate.stage",
        description="Final collate into parquet.",
        modules=("src.collate",),
        copied_sidecars=COLLATE_SIDECARS,
        mutator=patch_collate,
        uses_gpu=False,
    ),
    "pipeline.base": TargetSpec(
        name="pipeline.base",
        description="Full base pipeline from preprocess to collate.",
        modules=(
            "src.preprocess.preprocess",
            "src.preprocess.crest_factor_remover",
            "src.preprocess.preprocess_audio",
            "src.separation.music_detect",
            "src.separation.distillmos_process",
            "src.transcription.transcription",
            "src.punctuation.punctuation",
            "src.accents.accents",
            "src.phonemizer.phonemizer",
            "src.collate",
        ),
        mutator=patch_pipeline,
        uses_gpu=True,
    ),
}

for transcription_model in TRANSCRIPTION_MODELS:
    TARGETS[f"transcription.{transcription_model}"] = TargetSpec(
        name=f"transcription.{transcription_model}",
        description=f"Transcription benchmark for {transcription_model}.",
        modules=("src.transcription.transcription",),
        mutator=patch_transcription_model(transcription_model),
        uses_gpu=True,
    )


def target_names() -> List[str]:
    return sorted(TARGETS.keys())


def min_duration_for_target(target: TargetSpec, config: Dict[str, Any]) -> Optional[float]:
    if not target.min_input_duration_from_config:
        return None
    current: Any = config
    for key in target.min_input_duration_from_config:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if current is None:
        return None
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def list_targets() -> None:
    for name in target_names():
        target = TARGETS[name]
        sidecars = ",".join(target.required_sidecars) if target.required_sidecars else "-"
        print(f"{name}\tGPU={int(target.uses_gpu)}\tinputs={sidecars}\t{target.description}")
