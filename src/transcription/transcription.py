import argparse
import multiprocessing as mp
from collections import Counter
from pathlib import Path
from typing import List, Optional

import onnx_asr
import torch
from loguru import logger
from tqdm import tqdm

from src.utils.datasets.transcription import create_transcription_dataloader, recognize_batch
from src.utils.gpu import get_onnx_providers
from src.utils.logging_setup import setup_logging
from src.utils.parallel import run_per_gpu_processes
from src.utils.csv_manager import discover_audio_paths
from src.utils.sidecars import text_sidecar_complete
from src.utils.stage_status import write_stage_status
from src.utils.utils import load_config, read_file_content

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
}

SUPPORTED_TIMESTAMPS = {'giga_ctc', 'giga_ctc_lm', 'tone', 'parakeet_v2', 'parakeet_v3', 'canary'}
TARGET_SAMPLE_RATE = 16_000


def format_length_range(lengths: torch.Tensor, sample_rate: int) -> str:
    if lengths.numel() == 0:
        return "empty"
    seconds = lengths.to(dtype=torch.float32) / float(sample_rate)
    return f"min={seconds.min().item():.2f}s max={seconds.max().item():.2f}s"


def save_results(paths: List[str], texts: List[Optional[str]], timestamps: Optional[List[Optional[str]]], model_suffix: str):
    for i, (path_str, text) in enumerate(zip(paths, texts)):
        path = Path(path_str)

        if text is None:
            logger.debug(f"No transcript result for {path.name}; leaving sidecar unchanged")
            continue

        txt_path = path.with_name(f"{path.stem}_{model_suffix}.txt")
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            logger.error(f"Write TXT failed {path.name}: {e}")

        ts = timestamps[i] if timestamps and i < len(timestamps) else ''
        if ts:
            tst_path = path.with_name(f"{path.stem}_{model_suffix}.tst")
            try:
                with open(tst_path, "w", encoding="utf-8") as f:
                    f.write(ts)
            except Exception as e:
                logger.error(f"Write TST failed {path.name}: {e}")

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


def run_worker(cuda_id: int, world_size: int, model_name: str,
               all_files: List[str], config: dict, config_path: Optional[str] = None,
               processed_counter=None, errors_counter=None, error_details=None):
    """Inference worker: loads onnx-asr model on a single GPU and processes its shard."""
    my_files = all_files[cuda_id::world_size]
    if not my_files:
        return
    torch.cuda.set_device(cuda_id)

    batch_size = config.get('batch_size', 16)
    num_loader_workers = int(config.get('num_workers', 4))
    prefetch_factor = int(config.get('prefetch_factor', 2))
    use_trt = config.get('use_tensorrt', False)
    quantization = config.get('quantization')

    onnx_name = MODEL_MAP.get(model_name, model_name)
    output_suffix = 'vosk' if 'vosk' in model_name else model_name
    do_timestamps = config.get('with_timestamps', False) and model_name in SUPPORTED_TIMESTAMPS

    local_path = config.get('vosk_path') if 'vosk' in model_name else config.get('model_path')

    logger.info(
        f"Worker {cuda_id}/{world_size}: {onnx_name} on cuda:{cuda_id}, "
        f"{len(my_files)} files, batch={batch_size}, tensorrt={use_trt}"
    )

    try:
        providers = get_onnx_providers(cuda_id, use_tensorrt=use_trt, config_path=config_path)
        logger.info(f"ONNX providers for {model_name} on cuda:{cuda_id}: {providers}")
        load_args = [onnx_name] + ([local_path] if local_path else [])
        load_kwargs = {"providers": providers}
        if quantization:
            load_kwargs["quantization"] = quantization

        model = onnx_asr.load_model(*load_args, **load_kwargs)

        if do_timestamps:
            model = model.with_timestamps()

        if config.get('use_vad', False):
            vad_params = config.get('vad_params', {})
            vad = onnx_asr.load_vad("silero", **vad_params)
            model = model.with_vad(vad)

        target_sample_rate = int(model.asr._get_sample_rate()) if hasattr(model, "asr") else TARGET_SAMPLE_RATE
        dataloader = create_transcription_dataloader(
            my_files,
            sample_rate=target_sample_rate,
            batch_size=batch_size,
            num_workers=num_loader_workers,
            prefetch_factor=prefetch_factor,
        )

        for paths, waveforms, lengths, load_errors in tqdm(dataloader, desc=f"ASR-{cuda_id}", position=cuda_id):
            for path_str, reason in load_errors:
                logger.error(f"Audio load failed {path_str}: {reason}")
                if errors_counter is not None:
                    errors_counter.value += 1
                if error_details is not None:
                    error_details.append({"file": path_str, "model": model_name, "reason": reason})

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
                            error_details.append({"file": path_str, "model": model_name, "seconds": seconds, "reason": str(e2)})

            if not isinstance(results, list):
                results = [results]

            texts = [None if r is None else extract_text(r) for r in results]
            ts = [None if r is None else format_timestamps(r) for r in results] if do_timestamps else None

            save_results(paths, texts, ts, output_suffix)

            if processed_counter is not None:
                processed_counter.value += len(paths)

    except Exception as e:
        logger.exception(f"Worker {cuda_id} fatal error ({model_name}): {e}")
        if errors_counter is not None:
            errors_counter.value += 1
        if error_details is not None:
            error_details.append({"worker": cuda_id, "model": model_name, "reason": str(e)})


def check_consensus(audio_path: Path, model_names: List[str], consensus_num: int) -> bool:
    texts = []
    for mn in model_names:
        suffix = 'vosk' if 'vosk' in mn else mn
        tp = audio_path.with_name(f"{audio_path.stem}_{suffix}.txt")
        if text_sidecar_complete(tp):
            try:
                t = read_file_content(tp)
                if t:
                    texts.append(t.lower().strip())
            except Exception:
                pass
    if len(texts) < consensus_num:
        return False
    return max(Counter(texts).values()) >= consensus_num


def get_valid_paths(src_path: str, output_suffix: str,
                    processed: List[str], consensus_num: int,
                    retry_empty_outputs: bool = False,
                    config_path: Optional[str] = None) -> List[str]:
    all_paths = [Path(p) for p in discover_audio_paths(src_path, config_path=config_path)]
    if not all_paths:
        return []

    valid = []
    retry_empty_count = 0
    for p in all_paths:
        sidecar = p.with_name(f"{p.stem}_{output_suffix}.txt")
        if text_sidecar_complete(sidecar, retry_empty=retry_empty_outputs):
            continue
        if retry_empty_outputs:
            try:
                if sidecar.exists() and sidecar.stat().st_size == 0:
                    retry_empty_count += 1
            except OSError:
                pass
        valid.append(p)

    if retry_empty_count:
        logger.info(f"Retrying {retry_empty_count} empty {output_suffix} sidecars")

    if consensus_num > 0 and len(processed) >= consensus_num:
        skipped = 0
        filtered = []
        for p in valid:
            if check_consensus(p, processed, consensus_num):
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

    for idx, model_name in enumerate(model_names):
        logger.info(f"=== [{idx + 1}/{len(model_names)}] {model_name} ===")

        output_suffix = 'vosk' if 'vosk' in model_name else model_name
        processed_names = model_names[:idx] if consensus_num > 0 else []
        paths = get_valid_paths(src_path, output_suffix, processed_names, consensus_num, retry_empty_outputs, args.config_path)

        if not paths:
            logger.info(f"No files to process for {model_name}")
            continue

        logger.info(f"{len(paths)} files to process")

        worker_errors, worker_error_details = run_per_gpu_processes(
            run_worker,
            num_gpus=num_gpus,
            args=(model_name, paths, config, args.config_path, processed, errors, error_details_list),
        )
        if worker_errors:
            errors.value += worker_errors
            for detail in worker_error_details:
                error_details_list.append({"model": model_name, **detail})

    if config.get('use_rover', False):
        logger.info("ROVER aggregation...")
        try:
            from src.transcription.rover import ROVERWrapper
            ROVERWrapper(podcasts_path=src_path, model_names=model_names, config_path=args.config_path).aggregate_and_save()
            logger.info("ROVER done.")
        except ImportError:
            logger.warning("ROVER module not available, skipping")
        except Exception as e:
            logger.error(f"ROVER failed: {e}")

    logger.info("Transcription pipeline complete!")

    write_stage_status(
        stage=6,
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
