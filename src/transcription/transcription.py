import argparse
import gc
import multiprocessing as mp
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import onnx_asr
import torch
from loguru import logger
from tqdm import tqdm

from src.utils.audio_durations import (
    duration_bucket_settings,
    duration_probe_workers,
    ensure_audio_durations,
)
from src.transcription.fast_rnnt import patch_model as _patch_fast_rnnt
from src.utils.datasets.transcription import (
    PersistentGroupTranscriptionLoader,
    PersistentTranscriptionLoader,
    create_group_transcription_dataloader,
    create_transcription_dataloader,
    recognize_batch,
)
from src.utils.gpu import get_onnx_providers, make_session_options
from src.utils.logging_setup import setup_logging
from src.utils.node_profile import resolve_batch_size
from src.utils.parallel import run_per_gpu_processes
from src.utils.csv_manager import discover_audio_paths
from src.utils.chunk_json import ChunkJsonCache, get_field, update_chunk_json
from src.utils.stage_status import last_line, write_stage_status
from src.utils.utils import load_config
from src.utils.work_shards import (
    claim_work_shard,
    load_work_shard_size,
    mark_work_shard_done,
    prepare_length_bucketed_work_shards,
    read_annotated_work_shard,
    read_work_shard,
)

MODEL_MAP = {
    'giga_rnnt': 'gigaam-v3-rnnt',
    'giga_ctc': 'gigaam-v3-ctc',
    'giga_ctc_lm': 'gigaam-v3-ctc',
    'tone': 't-tech/t-one',
    'vosk': 'alphacep/vosk-model-ru',
    'vosk_small': 'alphacep/vosk-model-small-ru',
    'parakeet_v2': 'nemo-parakeet-tdt-0.6b-v2',
    'parakeet_v3': 'nemo-parakeet-tdt-0.6b-v3',
    'canary': 'nemo-canary-1b-v2',
    'whisper_base': 'whisper-base',
    'whisper_turbo': 'onnx-community/whisper-large-v3-turbo',
    'gigaam-v3-e2e-ctc': 'gigaam-v3-e2e-ctc'
}

SUPPORTED_TIMESTAMPS = {'giga_ctc', 'giga_ctc_lm', 'tone', 'parakeet_v2', 'parakeet_v3', 'canary'}
TARGET_SAMPLE_RATE = 16_000

# Per-worker count of fast_rnnt patch skips (fast path -> stock decode). Each
# worker process gets its own module global; surfaced in the end-of-run summary
# log line so a mixed fast/stock run is greppable beyond the single warning.
_FAST_PATH_FALLBACKS = 0


def reset_fast_path_fallbacks() -> None:
    global _FAST_PATH_FALLBACKS
    _FAST_PATH_FALLBACKS = 0


def fast_path_fallback_count() -> int:
    return _FAST_PATH_FALLBACKS


def log_fast_path_fallbacks(cuda_id: int) -> None:
    logger.info(f"Worker {cuda_id}: fast-path fallbacks: {_FAST_PATH_FALLBACKS}")


def maybe_patch_fast_rnnt(model, config: dict, config_path: Optional[str] = None):
    """Install batched stateful RNN-T greedy decode on a loaded onnx-asr model.

    No-op for CTC / non-transducer / unrecognized topologies and when
    ``transcription.use_fast_rnnt`` is False. Plugs in transparently for
    BOTH ``share_decode`` modes (sequential ``run_worker`` and grouped
    ``run_group_worker``): it only replaces the per-utterance greedy decode
    loop — the stage-7 critical path for ``giga_rnnt`` (200+ sequential
    batch-1 ONNX calls/file) and the vosk Kaldi transducer — with a batched
    equivalent that is token- and timestamp-identical to stock at batch
    sizes 1/4/8 on 250 real wavs (report.md §9.1). Call after
    ``with_timestamps`` / ``with_vad``; those adapters share the same
    ``.asr`` object, so one patch covers the timestamp path too. Never
    raises: any unexpected model is returned unpatched (stock decode).
    """
    if not config.get('use_fast_rnnt', True):
        return model
    try:
        # config_path delivers runtime.threads_per_worker's intra-op cap to the
        # rebuilt decoder/joiner sessions (None falls back to the shell-exported
        # path / repo default inside fast_rnnt, so the cap still applies).
        return _patch_fast_rnnt(model, config_path=config_path)
    except Exception as exc:  # never let the fast path break a worker
        global _FAST_PATH_FALLBACKS
        _FAST_PATH_FALLBACKS += 1
        logger.warning(f"fast_rnnt patch skipped ({exc}); using stock decode")
        return model


def format_length_range(lengths: torch.Tensor, sample_rate: int) -> str:
    if lengths.numel() == 0:
        return "empty"
    seconds = lengths.to(dtype=torch.float32) / float(sample_rate)
    return f"min={seconds.min().item():.2f}s max={seconds.max().item():.2f}s"


def save_results(paths: List[str], texts: List[Optional[str]], timestamps: Optional[List[Optional[str]]], model_suffix: str):
    """Merge this model's transcript (and optional timestamps) into each chunk's
    ``<stem>.json`` under ``asr[model_suffix]`` / ``asr_ts[model_suffix]``.

    :func:`update_chunk_json` is an atomic read-modify-write, so a killed worker
    never publishes a partial JSON; a re-run re-reads the last complete version.
    """
    for i, (path_str, text) in enumerate(zip(paths, texts)):
        path = Path(path_str)

        if text is None:
            logger.debug(f"No transcript result for {path.name}; leaving JSON unchanged")
            continue

        updates: dict = {"asr": {model_suffix: text}}
        ts = timestamps[i] if timestamps and i < len(timestamps) else ''
        if ts:
            updates["asr_ts"] = {model_suffix: ts}
        try:
            update_chunk_json(path, updates)
        except Exception as e:
            logger.error(f"Write chunk JSON failed {path.name}: {e}")

def extract_text(result) -> str:
    """Extract plain text from onnx-asr result (str or TimestampedResult)."""
    if hasattr(result, 'text'):
        return result.text
    return str(result)


def format_timestamps(result) -> str:
    """Format TimestampedResult as word-level TSV: start\\tend\\tword per line.

    onnx-asr TimestampedResult has parallel arrays:
      .tokens     = ['с', 'п', 'а', 'с', 'и', 'б', 'о', ' ', ...]
      .timestamps = [0.39, 0.44, 0.51, 0.54, 0.57, 0.63, 0.66, 0.75, ...]
    We group characters into words and produce word-level timestamps.
    """
    tokens = getattr(result, 'tokens', None)
    timestamps = getattr(result, 'timestamps', None)

    if not tokens or not timestamps or len(tokens) != len(timestamps):
        return ''

    words = []
    current_word = ''
    word_start = None

    for token, ts in zip(tokens, timestamps):
        if token.strip() == '':
            if current_word and word_start is not None:
                words.append((word_start, ts, current_word))
                current_word = ''
                word_start = None
        else:
            if word_start is None:
                word_start = ts
            current_word += token

    if current_word and word_start is not None:
        words.append((word_start, timestamps[-1], current_word))

    return '\n'.join(f"{start:.3f}\t{end:.3f}\t{word}" for start, end, word in words)


def _process_batches(batch_iter, cuda_id: int, model_name: str, model, output_suffix: str,
                     do_timestamps: bool, target_sample_rate: int,
                     processed_counter=None, errors_counter=None, error_details=None):
    """Consume ``(paths, waveforms, lengths, load_errors)`` batches.

    Shared by the per-shard loader and the persistent loader: the batch
    sequence is identical either way, so recognition/save semantics are too.
    """
    for paths, waveforms, lengths, load_errors in batch_iter:
        for path_str, reason in load_errors:
            logger.error(f"Audio load failed {path_str}: {reason}")
            if errors_counter is not None:
                errors_counter.value += 1
            if error_details is not None:
                error_details.append({"file": path_str, "model": model_name, "reason": last_line(reason)})

        if not paths:
            continue

        try:
            results = recognize_batch(model, waveforms, lengths)
        except Exception as e:
            logger.error(
                f"Batch failed for {model_name}: files={len(paths)}, "
                f"lengths=({format_length_range(lengths, target_sample_rate)}): {e}. "
                "Falling back to single-file mode."
            )
            results = []
            for path_str, waveform, length in zip(paths, waveforms, lengths):
                try:
                    results.extend(recognize_batch(model, waveform[:length].unsqueeze(0).contiguous(), length.unsqueeze(0)))
                except Exception as e2:
                    seconds = float(length.item()) / float(target_sample_rate)
                    logger.error(f"File failed for {model_name}: seconds={seconds:.2f}, file={path_str}: {e2}")
                    results.append(None)
                    if errors_counter is not None:
                        errors_counter.value += 1
                    if error_details is not None:
                        error_details.append({"file": path_str, "model": model_name, "seconds": seconds, "reason": last_line(e2)})

        if not isinstance(results, list):
            results = [results]

        texts = [None if r is None else extract_text(r) for r in results]
        ts = [None if r is None else format_timestamps(r) for r in results] if do_timestamps else None

        save_results(paths, texts, ts, output_suffix)

        if processed_counter is not None:
            processed_counter.value += len(paths)


def _process_files(cuda_id: int, model_name: str, files: List[str], model, config: dict,
                   output_suffix: str, do_timestamps: bool, target_sample_rate: int,
                   processed_counter=None, errors_counter=None, error_details=None):
    batch_size = resolve_batch_size(
        f"transcription.{model_name}", config.get('batch_size'), 16
    )
    num_loader_workers = int(config.get('num_workers', 4))
    prefetch_factor = int(config.get('prefetch_factor', 2))

    dataloader = create_transcription_dataloader(
        files,
        sample_rate=target_sample_rate,
        batch_size=batch_size,
        num_workers=num_loader_workers,
        prefetch_factor=prefetch_factor,
    )

    _process_batches(
        tqdm(dataloader, desc=f"ASR-{cuda_id}", position=cuda_id),
        cuda_id, model_name, model, output_suffix, do_timestamps, target_sample_rate,
        processed_counter, errors_counter, error_details,
    )


def run_worker(cuda_id: int, world_size: int, model_name: str,
               work_dir: str, config: dict, config_path: Optional[str] = None,
               processed_counter=None, errors_counter=None, error_details=None):
    """Inference worker: loads onnx-asr model on a single GPU and claims file shards."""
    torch.cuda.set_device(cuda_id)
    reset_fast_path_fallbacks()

    batch_size = resolve_batch_size(
        f"transcription.{model_name}", config.get('batch_size'), 16
    )
    use_trt = config.get('use_tensorrt', False)
    quantization = config.get('quantization')

    onnx_name = MODEL_MAP.get(model_name, model_name)
    output_suffix = 'vosk' if 'vosk' in model_name else model_name
    do_timestamps = config.get('with_timestamps', False) and model_name in SUPPORTED_TIMESTAMPS

    local_path = config.get('vosk_path') if 'vosk' in model_name else config.get('model_path')

    logger.info(
        f"Worker {cuda_id}/{world_size}: {onnx_name} on cuda:{cuda_id}, "
        f"claiming shards, batch={batch_size}, tensorrt={use_trt}"
    )

    try:
        providers = get_onnx_providers(cuda_id, use_tensorrt=use_trt, config_path=config_path)
        logger.info(f"ONNX providers for {model_name} on cuda:{cuda_id}: {providers}")
        load_args = [onnx_name] + ([local_path] if local_path else [])
        load_kwargs = {"providers": providers}
        # No-op unless runtime.threads_per_worker is set (default keeps ORT's
        # physical-core intra-op pool, so single-worker latency is unchanged).
        load_kwargs["sess_options"] = make_session_options(config_path=config_path)
        if quantization:
            load_kwargs["quantization"] = quantization

        model = onnx_asr.load_model(*load_args, **load_kwargs)

        if do_timestamps:
            model = model.with_timestamps()

        if config.get('use_vad', False):
            vad_params = config.get('vad_params', {})
            vad = onnx_asr.load_vad("silero", **vad_params)
            model = model.with_vad(vad)

        model = maybe_patch_fast_rnnt(model, config, config_path=config_path)

        target_sample_rate = int(model.asr._get_sample_rate()) if hasattr(model, "asr") else TARGET_SAMPLE_RATE

        num_loader_workers = int(config.get('num_workers', 4))
        prefetch_factor = int(config.get('prefetch_factor', 2))
        # Persistent loader keeps DataLoader workers alive across shards instead
        # of respawning them per claimed shard. Only meaningful with worker
        # processes; with num_workers==0 the per-shard loader has no spawn cost.
        use_persistent = bool(config.get('persistent_loaders', True)) and num_loader_workers > 0

        claimed = 0
        loader = None
        try:
            if use_persistent:
                loader = PersistentTranscriptionLoader(
                    sample_rate=target_sample_rate,
                    batch_size=batch_size,
                    num_workers=num_loader_workers,
                    prefetch_factor=prefetch_factor,
                ).__enter__()

            while True:
                shard_path = claim_work_shard(work_dir, cuda_id)
                if shard_path is None:
                    break
                files = read_work_shard(shard_path)
                claimed += 1
                logger.info(f"Worker {cuda_id}: processing {len(files)} files from {shard_path.name}")
                progress_desc = f"ASR {model_name} GPU {cuda_id} shard {claimed}"
                if loader is not None:
                    _process_batches(
                        tqdm(loader.iter_shard(files), desc=progress_desc, position=cuda_id, unit="batch"),
                        cuda_id, model_name, model, output_suffix, do_timestamps,
                        target_sample_rate, processed_counter, errors_counter, error_details,
                    )
                else:
                    _process_files(
                        cuda_id,
                        model_name,
                        files,
                        model,
                        config,
                        output_suffix,
                        do_timestamps,
                        target_sample_rate,
                        processed_counter,
                        errors_counter,
                        error_details,
                    )
                mark_work_shard_done(shard_path)
        finally:
            if loader is not None:
                loader.__exit__(None, None, None)

        logger.info(f"Worker {cuda_id} finished {claimed} shard(s) for {model_name}.")
        log_fast_path_fallbacks(cuda_id)

    except Exception as e:
        logger.exception(f"Worker {cuda_id} fatal error ({model_name}): {e}")
        if errors_counter is not None:
            errors_counter.value += 1
        if error_details is not None:
            error_details.append({"worker": cuda_id, "model": model_name, "reason": last_line(e)})

@dataclass
class _GroupModelSpec:
    """One loaded ASR model inside a shared-decode group worker."""

    name: str
    model: object
    suffix: str
    do_timestamps: bool
    batch_size: int
    sample_rate: int


def _load_group_model(
    model_name: str, config: dict, providers, config_path: Optional[str] = None
) -> _GroupModelSpec:
    onnx_name = MODEL_MAP.get(model_name, model_name)
    local_path = config.get('vosk_path') if 'vosk' in model_name else config.get('model_path')
    load_args = [onnx_name] + ([local_path] if local_path else [])
    load_kwargs = {"providers": providers}
    # No-op unless runtime.threads_per_worker is set (see make_session_options).
    load_kwargs["sess_options"] = make_session_options(config_path=config_path)
    if config.get('quantization'):
        load_kwargs["quantization"] = config.get('quantization')

    model = onnx_asr.load_model(*load_args, **load_kwargs)

    do_timestamps = config.get('with_timestamps', False) and model_name in SUPPORTED_TIMESTAMPS
    if do_timestamps:
        model = model.with_timestamps()
    if config.get('use_vad', False):
        vad = onnx_asr.load_vad("silero", **config.get('vad_params', {}))
        model = model.with_vad(vad)

    model = maybe_patch_fast_rnnt(model, config, config_path=config_path)

    sample_rate = int(model.asr._get_sample_rate()) if hasattr(model, "asr") else TARGET_SAMPLE_RATE
    return _GroupModelSpec(
        name=model_name,
        model=model,
        suffix='vosk' if 'vosk' in model_name else model_name,
        do_timestamps=do_timestamps,
        batch_size=resolve_batch_size(
            f"transcription.{model_name}", config.get('batch_size'), 16
        ),
        sample_rate=sample_rate,
    )


def _recognize_chunk(spec: _GroupModelSpec, paths: List[str], waveforms, lengths,
                     sample_rate: int, errors_counter=None, error_details=None):
    """Run one model over one sub-chunk with the same per-file fallback as
    the sequential path."""
    try:
        return recognize_batch(spec.model, waveforms, lengths)
    except Exception as e:
        logger.error(
            f"Batch failed for {spec.name}: files={len(paths)}, "
            f"lengths=({format_length_range(lengths, sample_rate)}): {e}. "
            "Falling back to single-file mode."
        )
        results = []
        for path_str, waveform, length in zip(paths, waveforms, lengths):
            try:
                results.extend(recognize_batch(
                    spec.model, waveform[:length].unsqueeze(0).contiguous(), length.unsqueeze(0)
                ))
            except Exception as e2:
                seconds = float(length.item()) / float(sample_rate)
                logger.error(f"File failed for {spec.name}: seconds={seconds:.2f}, file={path_str}: {e2}")
                results.append(None)
                if errors_counter is not None:
                    errors_counter.value += 1
                if error_details is not None:
                    error_details.append({"file": path_str, "model": spec.name, "seconds": seconds, "reason": last_line(e2)})
        return results


def _group_shard_inputs(specs: List[_GroupModelSpec], items: List[tuple]):
    """Split annotated shard items into (files, needed-map) for a group shard."""
    all_names = [spec.name for spec in specs]
    files: List[str] = []
    needed: Dict[str, Set[str]] = {}
    for path, note in items:
        files.append(path)
        needed[path] = set(note.split(',')) if note else set(all_names)
    return files, needed


def _process_group_batches(batch_iter, specs: List[_GroupModelSpec],
                           needed: Dict[str, Set[str]], all_names: List[str],
                           processed_counter=None, errors_counter=None, error_details=None):
    """Consume grouped ``(paths, padded_by_rate, load_errors)`` batches.

    Shared by the per-shard and persistent group loaders: the macro-batch
    sequence is identical, so sub-batching and save semantics are too.
    """
    for paths, padded_by_rate, load_errors in batch_iter:
        for path_str, reason in load_errors:
            # Mirror the sequential flow's accounting: each model that still
            # needed this file would have failed to load it once.
            for name in needed.get(path_str, set(all_names)):
                logger.error(f"Audio load failed {path_str}: {reason}")
                if errors_counter is not None:
                    errors_counter.value += 1
                if error_details is not None:
                    error_details.append({"file": path_str, "model": name, "reason": last_line(reason)})

        if not paths:
            continue

        for spec in specs:
            indices = [i for i, p in enumerate(paths) if spec.name in needed[p]]
            if not indices:
                continue
            padded, lengths = padded_by_rate[spec.sample_rate]
            for start in range(0, len(indices), spec.batch_size):
                chunk = indices[start:start + spec.batch_size]
                chunk_paths = [paths[i] for i in chunk]
                chunk_lengths = lengths[chunk]
                max_len = int(chunk_lengths.max().item())
                chunk_waveforms = padded[chunk][:, :max_len].contiguous()

                results = _recognize_chunk(
                    spec, chunk_paths, chunk_waveforms, chunk_lengths,
                    spec.sample_rate, errors_counter, error_details,
                )
                if not isinstance(results, list):
                    results = [results]
                texts = [None if r is None else extract_text(r) for r in results]
                ts = [None if r is None else format_timestamps(r) for r in results] if spec.do_timestamps else None
                save_results(chunk_paths, texts, ts, spec.suffix)

                if processed_counter is not None:
                    processed_counter.value += len(chunk_paths)


def _process_group_files(cuda_id: int, specs: List[_GroupModelSpec],
                         items: List[tuple], config: dict,
                         processed_counter=None, errors_counter=None, error_details=None):
    """Per-shard group loader path (non-persistent fallback)."""
    all_names = [spec.name for spec in specs]
    files, needed = _group_shard_inputs(specs, items)

    macro_batch = max(spec.batch_size for spec in specs)
    num_loader_workers = int(config.get('num_workers', 4))
    prefetch_factor = int(config.get('prefetch_factor', 2))

    dataloader = create_group_transcription_dataloader(
        files,
        sample_rates=[spec.sample_rate for spec in specs],
        batch_size=macro_batch,
        num_workers=num_loader_workers,
        prefetch_factor=prefetch_factor,
    )

    _process_group_batches(
        tqdm(dataloader, desc=f"ASR-group-{cuda_id}", position=cuda_id),
        specs, needed, all_names,
        processed_counter, errors_counter, error_details,
    )


def run_group_worker(cuda_id: int, world_size: int, group_models: List[str],
                     work_dir: str, config: dict, config_path: Optional[str] = None,
                     processed_counter=None, errors_counter=None, error_details=None):
    """Shared-decode inference worker: loads ALL group models on one GPU and
    claims annotated file shards (path TAB comma-joined-pending-models)."""
    torch.cuda.set_device(cuda_id)
    reset_fast_path_fallbacks()

    try:
        providers = get_onnx_providers(
            cuda_id, use_tensorrt=config.get('use_tensorrt', False), config_path=config_path
        )
        logger.info(f"ONNX providers for group {group_models} on cuda:{cuda_id}: {providers}")
        specs = [
            _load_group_model(name, config, providers, config_path=config_path)
            for name in group_models
        ]
        for spec in specs:
            logger.info(
                f"Worker {cuda_id}/{world_size}: {spec.name} loaded "
                f"(batch={spec.batch_size}, rate={spec.sample_rate}, timestamps={spec.do_timestamps})"
            )

        all_names = [spec.name for spec in specs]
        macro_batch = max(spec.batch_size for spec in specs)
        num_loader_workers = int(config.get('num_workers', 4))
        prefetch_factor = int(config.get('prefetch_factor', 2))
        use_persistent = bool(config.get('persistent_loaders', True)) and num_loader_workers > 0

        claimed = 0
        loader = None
        try:
            if use_persistent:
                loader = PersistentGroupTranscriptionLoader(
                    sample_rates=[spec.sample_rate for spec in specs],
                    batch_size=macro_batch,
                    num_workers=num_loader_workers,
                    prefetch_factor=prefetch_factor,
                ).__enter__()

            while True:
                shard_path = claim_work_shard(work_dir, cuda_id)
                if shard_path is None:
                    break
                items = read_annotated_work_shard(shard_path)
                claimed += 1
                logger.info(f"Worker {cuda_id}: group-processing {len(items)} files from {shard_path.name}")
                progress_desc = f"ASR group GPU {cuda_id} shard {claimed}"
                if loader is not None:
                    files, needed = _group_shard_inputs(specs, items)
                    _process_group_batches(
                        tqdm(loader.iter_shard(files), desc=progress_desc, position=cuda_id, unit="batch"),
                        specs, needed, all_names,
                        processed_counter, errors_counter, error_details,
                    )
                else:
                    _process_group_files(
                        cuda_id, specs, items, config,
                        processed_counter, errors_counter, error_details,
                    )
                mark_work_shard_done(shard_path)
        finally:
            if loader is not None:
                loader.__exit__(None, None, None)

        logger.info(f"Worker {cuda_id} finished {claimed} shard(s) for group {group_models}.")
        log_fast_path_fallbacks(cuda_id)

    except Exception as e:
        logger.exception(f"Worker {cuda_id} fatal error (group {group_models}): {e}")
        if errors_counter is not None:
            errors_counter.value += 1
        if error_details is not None:
            error_details.append({"worker": cuda_id, "model": ",".join(group_models), "reason": last_line(e)})


def normalize_consensus_text(text: str) -> str:
    """Normalize model text for consensus without changing saved sidecars."""
    without_punctuation = (
        " " if unicodedata.category(char).startswith("P") else char
        for char in text.casefold()
    )
    return " ".join("".join(without_punctuation).split())


def check_consensus(audio_path: Path, model_names: List[str], consensus_num: int,
                    cache: Optional[ChunkJsonCache] = None) -> bool:
    texts = []
    data = cache.get(audio_path) if cache is not None else None
    if data is None:
        from src.utils.chunk_json import read_chunk_json, chunk_json_path
        data = read_chunk_json(chunk_json_path(audio_path))
    for mn in model_names:
        suffix = 'vosk' if 'vosk' in mn else mn
        t = get_field(data, f"asr.{suffix}")
        if t:
            normalized = normalize_consensus_text(t)
            if normalized:
                texts.append(normalized)
    if len(texts) < consensus_num:
        return False
    return max(Counter(texts).values()) >= consensus_num


def get_valid_paths(src_path: str, output_suffix: str,
                    processed: List[str], consensus_num: int,
                    retry_empty_outputs: bool = False,
                    config_path: Optional[str] = None,
                    cache: Optional[ChunkJsonCache] = None) -> List[str]:
    """Audio files still needing ``output_suffix`` transcription.

    Completeness is read from each chunk's ``<stem>.json`` via one
    ``ChunkJsonCache`` (one scandir per directory + one small JSON read per
    existing file). Pass a shared ``cache`` to reuse directory listings and
    parsed JSONs across several model sweeps (the grouped shared-decode pass
    does this); otherwise a fresh per-call cache is built. The cache lives in
    this process only — it must not cross the spawn boundary into workers.
    """
    all_paths = [Path(p) for p in discover_audio_paths(src_path, config_path=config_path)]

    if not all_paths:
        return []

    if cache is None:
        cache = ChunkJsonCache()

    field = f"asr.{output_suffix}"
    valid = []
    retry_empty_count = 0
    scan_desc = f"Pending scan [{output_suffix}]"
    for p in tqdm(all_paths, desc=scan_desc, unit="file", mininterval=0.5):
        if cache.field_complete(p, field, retry_empty=retry_empty_outputs):
            continue
        if retry_empty_outputs and get_field(cache.get(p), field) == "":
            retry_empty_count += 1
        valid.append(p)

    if retry_empty_count:
        logger.info(f"Retrying {retry_empty_count} empty {output_suffix} sidecars")

    if consensus_num > 0 and len(processed) >= consensus_num:
        skipped = 0
        filtered = []
        consensus_desc = f"Consensus scan [{output_suffix}]"
        for p in tqdm(valid, desc=consensus_desc, unit="file", mininterval=0.5):
            if check_consensus(p, processed, consensus_num, cache):
                skipped += 1
            else:
                filtered.append(p)
        if skipped:
            logger.info(f"Consensus reached for {skipped} files, skipping")
        valid = filtered

    return [str(p) for p in valid]


def main(args):
    setup_logging("transcription", log_dir=args.log_dir)
    config = load_config(args.config_path, 'transcription')
    model_names = config.get('model_names', ['giga_rnnt'])
    src_path = config.get('podcasts_path', '.')
    consensus_num = config.get('consensus_num', 0)
    retry_empty_outputs = bool(config.get('retry_empty_outputs', False))

    processed = mp.Value('i', 0)
    errors = mp.Value('i', 0)
    error_details_list = mp.Manager().list()

    num_gpus = torch.cuda.device_count()

    logger.info(f"{num_gpus} GPU(s) detected. Starting transcription pipeline.")
    if consensus_num > 0:
        logger.info(f"Consensus mode: {consensus_num} models must agree")
    if retry_empty_outputs:
        logger.info("Retry-empty mode enabled: zero-byte transcript sidecars will be reprocessed")

    share_decode = bool(config.get('share_decode', True))
    # The first consensus_num models always transcribe every pending file (the
    # consensus filter in get_valid_paths only engages once that many earlier
    # models exist), so they can share one decode per file without changing
    # which files any model processes. The remaining models keep the exact
    # sequential flow because each one's pending set depends on the previous
    # model's outputs (consensus skipping).
    always_run = model_names[:consensus_num] if consensus_num > 0 else list(model_names)
    grouped_models = always_run if (share_decode and len(always_run) > 1) else []

    shard_size = load_work_shard_size(args.config_path)
    duration_workers = duration_probe_workers(config)
    bucket_seconds, max_bucket_duration = duration_bucket_settings(
        args.config_path,
        config,
    )
    # Intra-bucket shard order. "duration" (default) reproduces the old flow
    # bit-for-bit; "path" reads the disk in directory order — the big win on
    # HDD datasets — at the cost of changed ASR batch composition (measured:
    # ~15% of knife-edge bench transcripts shift by ~1 char, ~1.2% of chars).
    shard_order = str(config.get('shard_order', 'duration'))

    if grouped_models:
        logger.info(f"=== shared-decode group: {grouped_models} ===")
        needed: Dict[str, List[str]] = {}
        # One cache shared across the per-model sweeps: every grouped model
        # walks the same audio tree, so the directory listings and parsed chunk
        # JSONs are read once instead of once per model.
        group_cache = ChunkJsonCache()
        for model_name in tqdm(
            grouped_models,
            desc="Preparing shared-decode models",
            unit="model",
        ):
            output_suffix = 'vosk' if 'vosk' in model_name else model_name
            for p in get_valid_paths(src_path, output_suffix, [], consensus_num, retry_empty_outputs, args.config_path, cache=group_cache):
                needed.setdefault(p, []).append(model_name)

        # The scan-time directory cache holds every filename in the dataset
        # tree (audio + all sidecars) — multiple GB. It is only used by
        # get_valid_paths above; drop it before the multi-hour processing phase
        # so the parent does not carry it as dead weight while the per-GPU
        # workers (separate processes) run.
        del group_cache

        if not needed:
            logger.info(f"No files to process for group {grouped_models}")
        else:
            union_paths = list(needed.keys())
            total_group_results = sum(len(models) for models in needed.values())
            durations = ensure_audio_durations(
                src_path,
                union_paths,
                num_workers=duration_workers,
            )
            annotations = {p: ",".join(models) for p, models in needed.items()}
            group_tag = "_".join('vosk' if 'vosk' in m else m for m in grouped_models)
            work_plan = prepare_length_bucketed_work_shards(
                src_path,
                f"transcription_group_{group_tag}",
                union_paths,
                durations,
                shard_size=shard_size,
                bucket_seconds=bucket_seconds,
                max_duration=max_bucket_duration,
                annotations=annotations,
                order=shard_order,
            )
            del union_paths
            del durations
            del needed
            # annotations are now persisted in the shard files on disk; the
            # spawned workers read them back via read_annotated_work_shard and
            # never receive this dict (≈one entry per pending file). Free it and
            # return the freed pages to the OS before launching the workers, so
            # the parent's resident set drops from ~13 GB to ~1-2 GB for the
            # whole processing phase (critical headroom on multi-GPU runs).
            del annotations
            gc.collect()

            logger.info(
                f"{work_plan.total_items} files to process for group {grouped_models} "
                f"in {work_plan.shard_count} shard(s)."
            )

            worker_errors, worker_error_details = run_per_gpu_processes(
                run_group_worker,
                num_gpus=num_gpus,
                args=(grouped_models, str(work_plan.work_dir), config, args.config_path, processed, errors, error_details_list),
                progress_counter=processed,
                progress_total=total_group_results,
                progress_desc=f"Transcribing group ({len(grouped_models)} models)",
            )
            if worker_errors:
                errors.value += worker_errors
                for detail in worker_error_details:
                    error_details_list.append({"model": ",".join(grouped_models), **detail})

    for idx, model_name in enumerate(model_names):
        if model_name in grouped_models:
            continue
        logger.info(f"=== [{idx + 1}/{len(model_names)}] {model_name} ===")

        output_suffix = 'vosk' if 'vosk' in model_name else model_name
        processed_names = model_names[:idx] if consensus_num > 0 else []
        paths = get_valid_paths(src_path, output_suffix, processed_names, consensus_num, retry_empty_outputs, args.config_path)

        if not paths:
            logger.info(f"No files to process for {model_name}")
            continue

        durations = ensure_audio_durations(
            src_path,
            paths,
            num_workers=duration_workers,
        )
        work_plan = prepare_length_bucketed_work_shards(
            src_path,
            f"transcription_{output_suffix}",
            paths,
            durations,
            shard_size=shard_size,
            bucket_seconds=bucket_seconds,
            max_duration=max_bucket_duration,
            order=shard_order,
        )
        del paths
        del durations

        logger.info(
            f"{work_plan.total_items} files to process for {model_name} "
            f"in {work_plan.shard_count} shard(s)."
        )

        worker_errors, worker_error_details = run_per_gpu_processes(
            run_worker,
            num_gpus=num_gpus,
            args=(model_name, str(work_plan.work_dir), config, args.config_path, processed, errors, error_details_list),
            progress_counter=processed,
            progress_total=work_plan.total_items,
            progress_desc=f"Transcribing [{model_name}]",
        )
        if worker_errors:
            errors.value += worker_errors
            for detail in worker_error_details:
                error_details_list.append({"model": model_name, **detail})

    if config.get('use_rover', False):
        logger.info("ROVER aggregation...")
        try:
            from src.transcription.rover import ROVERWrapper
            ROVERWrapper(
                podcasts_path=src_path,
                model_names=model_names,
                config_path=args.config_path,
                shard_size=config.get('rover_shard_size'),
                workers=config.get('rover_workers', 1),
                retry_empty_outputs=retry_empty_outputs,
                use_fast_rover=bool(config.get('use_fast_rover', True)),
            ).aggregate_and_save()
            logger.info("ROVER done.")
        except ImportError:
            logger.warning("ROVER module not available, skipping")
        except Exception as e:
            logger.error(f"ROVER failed: {e}")

    logger.info("Transcription pipeline complete!")

    write_stage_status(
        stage=8,
        stage_name="transcription",
        log_dir=args.log_dir or "./logs",
        processed=processed.value,
        skipped=0,
        errors=errors.value,
        error_details=list(error_details_list),
    )


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="ASR Transcription (onnx-asr)")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    main(parser.parse_args())
