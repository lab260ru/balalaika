"""Per-node batch-size autotuner ("warmup") for balalaika model stages.

Probes every tunable model with growing batch sizes on THIS node, measures
throughput, respects free VRAM (other jobs may share the GPU), and writes a
node profile JSON that the pipeline can consume:

    source .dev_venv/bin/activate
    python -m benchmarking.warmup --config_path configs/config.yaml

Then either copy the recommended values into configs/config.yaml by hand
(see the printed summary / cache/node_profile.suggested.yaml), or set the
relevant ``batch_size: auto`` in the config — stages resolve ``auto`` through
:mod:`src.utils.node_profile` against ``cache/node_profile.json``.

The profile is node-specific: run this once per machine (it is exactly what
you re-run after moving to different hardware). On a GPU shared with other
jobs the numbers are depressed and marked ``"contended": true`` — re-run on
an idle GPU for a clean profile.

Probed models (auto-skipped with a reason when weights are unavailable):

* ``distillmos``      — torch ConvTransformerSQAModel, [B, T] @ 16 kHz
* ``antispoofing``    — Spectra-0 ONNX, fixed [B, 64600] @ 16 kHz
* ``denoising``       — MossFormer2_SE_48K ONNX, [B, 1, T] @ 48 kHz
* ``transcription.*`` — one probe per configured onnx-asr model
* ``punctuation``     — RUPunct token-classification pipeline (text batches)

Not probed (documented): sortformer (stateful streaming, batch fixed at 1 by
design), smart_turn (called per segment with batch 1), accents/phonemizer
(no batch API), music_detect (requires the optional musicdetection package —
probed only when importable).
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger

DEFAULT_PROFILE_PATH = REPO_ROOT / "cache" / "node_profile.json"
SAMPLE_RATE_16K = 16_000


class SkipProbe(RuntimeError):
    """Raised by a probe when its model cannot be loaded on this node."""


@dataclass
class SweepPoint:
    batch_size: int
    items_per_s: float
    audio_s_per_s: Optional[float]
    vram_used_mb: Optional[float]


@dataclass
class ProbeOutcome:
    key: str
    skipped: Optional[str] = None
    best_batch_size: Optional[int] = None
    curve: List[SweepPoint] = field(default_factory=list)
    error: Optional[str] = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cuda_free_total_mb(device_index: int) -> tuple[float, float]:
    import torch

    free, total = torch.cuda.mem_get_info(device_index)
    return free / 2**20, total / 2**20


def pick_device(requested: Optional[int]) -> int:
    """Default to the CUDA device with the most free VRAM."""
    import torch

    if requested is not None:
        return requested
    count = torch.cuda.device_count()
    if count == 0:
        raise SystemExit("No CUDA device available; warmup tunes GPU batch sizes.")
    best, best_free = 0, -1.0
    for i in range(count):
        free, _ = cuda_free_total_mb(i)
        if free > best_free:
            best, best_free = i, free
    return best


def batch_size_ladder(max_batch: int) -> List[int]:
    out = []
    bs = 1
    while bs <= max_batch:
        out.append(bs)
        bs *= 2
    return out


def sweep(
    key: str,
    *,
    device_index: int,
    make_batch: Callable[[int], Any],
    run_batch: Callable[[Any], None],
    audio_seconds_per_item: Optional[float],
    max_batch: int,
    min_probe_seconds: float = 2.0,
    vram_safety: float = 0.85,
    plateau_tolerance: float = 0.03,
) -> ProbeOutcome:
    """Measure throughput across the batch-size ladder for one model.

    Guards: per-item VRAM cost is estimated from the previous step; the next
    step is skipped when projected usage exceeds ``vram_safety`` of currently
    free VRAM (important when another job shares the GPU). OOM during a step
    ends the sweep instead of crashing.
    """
    import torch

    outcome = ProbeOutcome(key=key)
    best_throughput = 0.0
    declines = 0
    prev_per_item_mb: Optional[float] = None

    for bs in batch_size_ladder(max_batch):
        free_mb, _ = cuda_free_total_mb(device_index)
        if prev_per_item_mb is not None:
            projected = prev_per_item_mb * bs
            if projected > free_mb * vram_safety:
                logger.warning(
                    f"{key}: stopping at batch_size={bs} — projected "
                    f"{projected:.0f}MB exceeds {vram_safety:.0%} of free {free_mb:.0f}MB"
                )
                break
        try:
            batch = make_batch(bs)
            torch.cuda.synchronize(device_index)
            free_before, _ = cuda_free_total_mb(device_index)
            run_batch(batch)  # warmup iteration (excluded from timing)
            torch.cuda.synchronize(device_index)

            iters = 0
            started = time.perf_counter()
            while True:
                run_batch(batch)
                iters += 1
                torch.cuda.synchronize(device_index)
                elapsed = time.perf_counter() - started
                if elapsed >= min_probe_seconds and iters >= 3:
                    break
                if elapsed >= min_probe_seconds * 4:
                    break
            free_after, _ = cuda_free_total_mb(device_index)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"{key}: OOM at batch_size={bs}; ending sweep")
            torch.cuda.empty_cache()
            break
        except Exception as exc:  # ORT raises plain RuntimeError on OOM
            if "memory" in str(exc).lower() or "oom" in str(exc).lower():
                logger.warning(f"{key}: allocator failure at batch_size={bs}: {exc}")
                break
            raise

        items_per_s = (iters * bs) / elapsed
        vram_used = max(0.0, free_before - free_after)
        prev_per_item_mb = (vram_used / bs) if vram_used > 0 else prev_per_item_mb
        point = SweepPoint(
            batch_size=bs,
            items_per_s=items_per_s,
            audio_s_per_s=(items_per_s * audio_seconds_per_item)
            if audio_seconds_per_item
            else None,
            vram_used_mb=round(vram_used, 1),
        )
        outcome.curve.append(point)
        logger.info(
            f"{key}: bs={bs:<4d} {items_per_s:9.2f} items/s"
            + (f"  ({point.audio_s_per_s:8.1f} audio-s/s)" if point.audio_s_per_s else "")
            + (f"  vram~{vram_used:.0f}MB" if vram_used else "")
        )

        if items_per_s > best_throughput * (1 + plateau_tolerance):
            best_throughput = items_per_s
            declines = 0
        else:
            declines += 1
            if declines >= 2:
                logger.info(f"{key}: throughput plateaued; ending sweep")
                break

    if outcome.curve:
        # Best = highest throughput; ties within tolerance go to the SMALLER
        # batch (less VRAM, kinder to low-memory nodes).
        best = max(outcome.curve, key=lambda p: p.items_per_s)
        for p in outcome.curve:
            if p.items_per_s >= best.items_per_s * (1 - plateau_tolerance):
                outcome.best_batch_size = p.batch_size
                break
    else:
        outcome.error = "no successful batch"
    return outcome


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def probe_distillmos(cfg: Dict, device_index: int, args) -> ProbeOutcome:
    import torch

    try:
        import distillmos
    except Exception as exc:
        raise SkipProbe(f"distillmos package unavailable: {exc}")

    device = torch.device(f"cuda:{device_index}")
    model = distillmos.ConvTransformerSQAModel().to(device).eval()
    frames = int(SAMPLE_RATE_16K * args.probe_seconds)
    gen = torch.Generator().manual_seed(0)

    def make_batch(bs: int):
        return torch.randn(bs, frames, generator=gen) * 0.1

    @torch.inference_mode()
    def run_batch(batch):
        model(batch.to(device, non_blocking=False)).flatten().cpu()

    try:
        return sweep(
            "distillmos",
            device_index=device_index,
            make_batch=make_batch,
            run_batch=run_batch,
            audio_seconds_per_item=args.probe_seconds,
            max_batch=args.max_batch,
        )
    finally:
        del model
        torch.cuda.empty_cache()


def probe_antispoofing(cfg: Dict, device_index: int, args) -> ProbeOutcome:
    import numpy as np

    from src.separation.antispoofing import MODEL_NUM_SAMPLES, create_session

    sub = cfg.get("antispoofing", {}) or {}
    model_path = Path(sub.get("onnx_path", "./models/spectra_0.onnx"))
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    try:
        session, input_name, output_name = create_session(
            model_path, device_index, sub, args.config_path
        )
    except Exception as exc:
        raise SkipProbe(f"Spectra-0 session failed: {exc}")

    rng = np.random.default_rng(0)

    def make_batch(bs: int):
        return rng.standard_normal((bs, MODEL_NUM_SAMPLES), dtype=np.float32) * 0.1

    def run_batch(batch):
        session.run([output_name], {input_name: batch})

    try:
        return sweep(
            "antispoofing",
            device_index=device_index,
            make_batch=make_batch,
            run_batch=run_batch,
            audio_seconds_per_item=MODEL_NUM_SAMPLES / SAMPLE_RATE_16K,
            max_batch=args.max_batch,
        )
    finally:
        del session


def probe_denoising(cfg: Dict, device_index: int, args) -> ProbeOutcome:
    import numpy as np
    import onnxruntime as ort

    from src.denoising.denoising import (
        MODEL_SAMPLE_RATE,
        resolve_model_path,
    )
    from src.utils.gpu import get_onnx_providers

    model_path = resolve_model_path(
        cfg.get("onnx_path", "./models/MossFormer2_SE_48K_dynamic.onnx")
    )
    if not model_path.exists():
        raise SkipProbe(f"denoising ONNX not found at {model_path}")

    # CUDA EP for the sweep: TensorRT builds a new engine per (batch, length)
    # profile, which would turn a minutes-long warmup into hours. The relative
    # batch-size ranking carries over; rebuild TRT engines at pipeline runtime.
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = get_onnx_providers(device_index, use_tensorrt=False, config_path=args.config_path)
    try:
        session = ort.InferenceSession(str(model_path), sess_options, providers=providers)
    except Exception as exc:
        raise SkipProbe(f"denoising session failed: {exc}")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    frames = int(MODEL_SAMPLE_RATE * min(args.probe_seconds, 10.0))
    frames -= frames % 384  # MODEL_PAD_TO_MULTIPLE
    rng = np.random.default_rng(0)

    def make_batch(bs: int):
        # The stage feeds int16-scaled float32 (see datasets/denoising.py).
        return (rng.standard_normal((bs, 1, frames), dtype=np.float32) * 3000.0)

    def run_batch(batch):
        session.run([output_name], {input_name: batch})

    try:
        return sweep(
            "denoising",
            device_index=device_index,
            make_batch=make_batch,
            run_batch=run_batch,
            audio_seconds_per_item=frames / MODEL_SAMPLE_RATE,
            max_batch=min(args.max_batch, 32),
        )
    finally:
        del session


def probe_asr_model(model_name: str, cfg: Dict, device_index: int, args) -> ProbeOutcome:
    import torch

    try:
        import onnx_asr
    except Exception as exc:
        raise SkipProbe(f"onnx_asr unavailable: {exc}")

    from src.transcription.transcription import MODEL_MAP
    from src.utils.datasets.transcription import recognize_batch
    from src.utils.gpu import get_onnx_providers

    onnx_name = MODEL_MAP.get(model_name, model_name)
    providers = get_onnx_providers(
        device_index,
        use_tensorrt=False,  # see denoising note: per-shape TRT builds are not warmup material
        config_path=args.config_path,
    )
    try:
        model = onnx_asr.load_model(onnx_name, providers=providers)
    except Exception as exc:
        raise SkipProbe(f"onnx_asr load_model({onnx_name}) failed: {exc}")

    frames = int(SAMPLE_RATE_16K * args.probe_seconds)
    waveform_pool = _real_or_synthetic_audio(args, frames)

    def make_batch(bs: int):
        waves = torch.stack([waveform_pool[i % len(waveform_pool)] for i in range(bs)])
        lengths = torch.full((bs,), frames, dtype=torch.int64)
        return waves, lengths

    def run_batch(batch):
        waves, lengths = batch
        recognize_batch(model, waves, lengths)

    try:
        return sweep(
            f"transcription.{model_name}",
            device_index=device_index,
            make_batch=make_batch,
            run_batch=run_batch,
            audio_seconds_per_item=args.probe_seconds,
            max_batch=args.max_batch,
        )
    finally:
        del model


def _real_or_synthetic_audio(args, frames: int):
    """Up to 8 real 16 kHz mono clips from --audio-dir, else seeded noise.

    Real speech exercises the decoder realistically (noise emits few tokens,
    which makes autoregressive decoders look faster than they will be)."""
    import torch

    pool = []
    if args.audio_dir:
        import torchaudio

        for p in sorted(Path(args.audio_dir).rglob("*.wav"))[:8]:
            try:
                wave, sr = torchaudio.load_with_torchcodec(str(p))
                wave = wave.mean(dim=0) if wave.shape[0] > 1 else wave.squeeze(0)
                if sr != SAMPLE_RATE_16K:
                    wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE_16K)
                if wave.numel() >= frames:
                    pool.append(wave[:frames].contiguous())
                else:
                    reps = frames // max(1, wave.numel()) + 1
                    pool.append(wave.repeat(reps)[:frames].contiguous())
            except Exception:
                continue
    if not pool:
        gen = torch.Generator().manual_seed(0)
        pool = [torch.randn(frames, generator=gen) * 0.05 for _ in range(4)]
    return pool


def probe_punctuation(cfg: Dict, device_index: int, args) -> ProbeOutcome:
    try:
        from transformers import AutoTokenizer, pipeline
    except Exception as exc:
        raise SkipProbe(f"transformers unavailable: {exc}")

    # Mirror src/punctuation/punctuation.py:init_process exactly.
    model_name = cfg.get("model_name", "RUPunct/RUPunct_big")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            strip_accents=False,
            add_prefix_space=True,
        )
        nlp = pipeline(
            "ner",
            model=model_name,
            tokenizer=tokenizer,
            aggregation_strategy="first",
            device=device_index,
        )
    except Exception as exc:
        raise SkipProbe(f"RUPunct load failed: {exc}")

    text = (
        "привет меня зовут балалайка я обрабатываю русскую речь и расставляю "
        "знаки препинания в длинных предложениях которые модель должна разметить"
    )

    def make_batch(bs: int):
        return [text] * bs

    def run_batch(batch):
        nlp(batch, batch_size=len(batch))

    try:
        return sweep(
            "punctuation",
            device_index=device_index,
            make_batch=make_batch,
            run_batch=run_batch,
            audio_seconds_per_item=None,
            max_batch=min(args.max_batch, 128),
        )
    finally:
        del nlp, tokenizer


def probe_music_detect(cfg: Dict, device_index: int, args) -> ProbeOutcome:
    """Full-fidelity probe: reuses the stage's own loader + predict_proba.

    Synthesizes one batch worth of wav files per ladder step so the
    AutoFeatureExtractor / AudioCollate path is identical to the real stage.
    """
    try:
        import musicdetection  # noqa: F401
    except Exception as exc:
        raise SkipProbe(
            "musicdetection package not installed "
            f"(pip install git+https://github.com/NikiPshg/musicdetection): {exc}"
        )
    import tempfile

    import torch
    import torchaudio

    from src.separation.music_detect import create_loader, load_model

    sub = cfg.get("music_detect", {}) or {}
    weights = Path(sub.get("music_detect_model", "./models/music_detection.safetensors"))
    if not weights.is_absolute():
        weights = REPO_ROOT / weights
    if not weights.exists():
        raise SkipProbe(f"music detection weights not found at {weights}")
    base_model = sub.get("base_model", "microsoft/wavlm-base-plus")
    device = torch.device(f"cuda:{device_index}")
    model = load_model(str(weights), base_model, device)

    frames = int(SAMPLE_RATE_16K * args.probe_seconds)
    gen = torch.Generator().manual_seed(0)
    tmpdir = tempfile.TemporaryDirectory(prefix="warmup_music_")

    def make_batch(bs: int):
        paths = []
        for i in range(bs):
            p = Path(tmpdir.name) / f"clip_{bs}_{i}.wav"
            if not p.exists():
                wave = (torch.randn(1, frames, generator=gen) * 0.1).clamp(-1, 1)
                torchaudio.save_with_torchcodec(str(p), wave, SAMPLE_RATE_16K)
            paths.append(str(p))
        lengths = {p: float(args.probe_seconds) for p in paths}
        return create_loader(paths, base_model, bs, 0, lengths)

    @torch.inference_mode()
    def run_batch(loader):
        model.predict_proba(loader)

    try:
        return sweep(
            "music_detect",
            device_index=device_index,
            make_batch=make_batch,
            run_batch=run_batch,
            audio_seconds_per_item=args.probe_seconds,
            max_batch=args.max_batch,
        )
    finally:
        del model
        torch.cuda.empty_cache()
        tmpdir.cleanup()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def gpu_is_contended(device_index: int) -> bool:
    """True when other processes already hold a meaningful share of VRAM."""
    free, total = cuda_free_total_mb(device_index)
    return (total - free) > total * 0.2


def config_recommendations(results: Dict[str, ProbeOutcome]) -> Dict[str, Any]:
    rec: Dict[str, Any] = {}
    transcription_bs: List[int] = []
    for key, outcome in results.items():
        if outcome.best_batch_size is None:
            continue
        if key.startswith("transcription."):
            transcription_bs.append(outcome.best_batch_size)
        else:
            rec[key] = outcome.best_batch_size
    if transcription_bs:
        # one flat transcription.batch_size feeds every model -> take the min
        rec["transcription"] = min(transcription_bs)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config_path", type=str, default=str(REPO_ROOT / "configs" / "config.yaml"))
    ap.add_argument("--models", type=str, default="distillmos,antispoofing,denoising,transcription,punctuation,music_detect",
                    help="comma list: distillmos,antispoofing,denoising,transcription,punctuation,music_detect")
    ap.add_argument("--device", type=int, default=None, help="CUDA index (default: most free VRAM)")
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--probe-seconds", type=float, default=10.0,
                    help="synthetic clip duration for variable-length models")
    ap.add_argument("--audio-dir", type=str, default=None,
                    help="optional dir with real .wav files for ASR probes")
    ap.add_argument("--output", type=Path, default=DEFAULT_PROFILE_PATH)
    args = ap.parse_args()

    from src.utils.utils import load_config

    device_index = pick_device(args.device)
    free, total = cuda_free_total_mb(device_index)
    contended = gpu_is_contended(device_index)
    logger.info(
        f"Probing on cuda:{device_index} — {free:.0f}/{total:.0f} MB free"
        + (" (GPU SHARED with other jobs: numbers will be depressed)" if contended else "")
    )

    separation_cfg = load_config(args.config_path, "separation") or {}
    denoising_cfg = load_config(args.config_path, "denoising") or {}
    transcription_cfg = load_config(args.config_path, "transcription") or {}
    punctuation_cfg = load_config(args.config_path, "punctuation") or {}

    wanted = {m.strip() for m in args.models.split(",") if m.strip()}
    plan: List[tuple[str, Callable[[], ProbeOutcome]]] = []
    if "distillmos" in wanted:
        plan.append(("distillmos", lambda: probe_distillmos(separation_cfg, device_index, args)))
    if "antispoofing" in wanted:
        plan.append(("antispoofing", lambda: probe_antispoofing(separation_cfg, device_index, args)))
    if "denoising" in wanted:
        plan.append(("denoising", lambda: probe_denoising(denoising_cfg, device_index, args)))
    if "transcription" in wanted:
        for name in transcription_cfg.get("model_names", []) or []:
            plan.append(
                (f"transcription.{name}",
                 lambda name=name: probe_asr_model(name, transcription_cfg, device_index, args))
            )
    if "punctuation" in wanted:
        plan.append(("punctuation", lambda: probe_punctuation(punctuation_cfg, device_index, args)))
    if "music_detect" in wanted:
        plan.append(("music_detect", lambda: probe_music_detect(separation_cfg, device_index, args)))

    results: Dict[str, ProbeOutcome] = {}
    for key, fn in plan:
        logger.info(f"=== probing {key} ===")
        try:
            results[key] = fn()
        except SkipProbe as skip:
            logger.warning(f"{key}: SKIPPED — {skip}")
            results[key] = ProbeOutcome(key=key, skipped=str(skip))
        except Exception as exc:
            logger.error(f"{key}: probe failed: {exc}")
            results[key] = ProbeOutcome(key=key, error=str(exc))

    import torch

    profile = {
        "version": 1,
        "hostname": socket.gethostname(),
        "created_at_utc": utc_now(),
        "device": {
            "index": device_index,
            "name": torch.cuda.get_device_name(device_index),
            "vram_total_mb": round(total, 0),
            "vram_free_at_probe_mb": round(free, 0),
        },
        "contended": contended,
        "probe_seconds": args.probe_seconds,
        "models": {
            key: {
                "skipped": o.skipped,
                "error": o.error,
                "best_batch_size": o.best_batch_size,
                "curve": [vars(p) for p in o.curve],
            }
            for key, o in results.items()
        },
        "recommended_batch_sizes": config_recommendations(results),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    logger.success(f"node profile -> {args.output}")

    suggested = args.output.with_name("node_profile.suggested.yaml")
    rec = profile["recommended_batch_sizes"]
    lines = ["# Recommended batch sizes from benchmarking/warmup.py", f"# node: {profile['hostname']}, gpu: {profile['device']['name']}"]
    if contended:
        lines.append("# WARNING: probed on a GPU shared with other jobs; re-run when idle")
    mapping = {
        "distillmos": "separation.distillmos.batch_size",
        "antispoofing": "separation.antispoofing.batch_size",
        "denoising": "denoising.batch_size",
        "transcription": "transcription.batch_size",
        "music_detect": "separation.music_detect.bs",
        "punctuation": "punctuation.batch_size",
    }
    for key, bs in rec.items():
        lines.append(f"# {mapping.get(key, key)}: {bs}")
    suggested.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"profile": str(args.output), "recommended": rec}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
