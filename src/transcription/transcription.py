import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional
from collections import Counter
from loguru import logger
from tqdm import tqdm

import onnx_asr

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

from src.utils.utils import get_audio_paths, load_config, read_file_content

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


def get_gpu_count() -> int:
    try:
        import onnxruntime as ort
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            return 0
    except ImportError:
        return 0
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return len([line for line in result.stdout.strip().split('\n') if line.strip()])
    except Exception:
        pass
    return 1


def get_providers(cuda_id: int, use_tensorrt: bool = False) -> list:
    if use_tensorrt:
        return [
            ("TensorrtExecutionProvider", {
                "device_id": cuda_id,
                "trt_max_workspace_size": 6 * 1024**3,
                "trt_fp16_enable": True,
            }),
            ("CUDAExecutionProvider", {"device_id": cuda_id}),
        ]
    return [("CUDAExecutionProvider", {"device_id": cuda_id})]


def save_results(paths: List[str], texts: List[str], timestamps: Optional[List[str]], model_suffix: str):
    for i, (path_str, text) in enumerate(zip(paths, texts)):
        path = Path(path_str)

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


def load_batch(file_paths: List[str]):
    """Read batch — pass WAV paths directly to onnx-asr; non-WAV read via soundfile."""
    all_wav = all(Path(f).suffix.lower() == '.wav' for f in file_paths)
    if all_wav or not HAS_SOUNDFILE:
        return file_paths, None

    waveforms, sr = [], None
    for f in file_paths:
        wf, file_sr = sf.read(f, dtype='float32')
        if wf.ndim > 1:
            wf = wf.mean(axis=1)
        waveforms.append(wf)
        sr = file_sr
    return waveforms, sr


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
               all_files: List[str], config: dict):
    """Inference worker: loads onnx-asr model on a single GPU and processes its shard."""
    my_files = all_files[cuda_id::world_size]
    if not my_files:
        return

    model_cfg = config.get('giga', {}) if 'giga' in model_name else config.get(model_name, {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    batch_size = model_cfg.get('batch_size', 16)
    use_trt = config.get('use_tensorrt', False)
    quantization = model_cfg.get('quantization')

    onnx_name = MODEL_MAP.get(model_name, model_name)
    output_suffix = 'vosk' if 'vosk' in model_name else model_name
    do_timestamps = config.get('with_timestamps', False) and model_name in SUPPORTED_TIMESTAMPS

    local_path = model_cfg.get('vosk_path') if 'vosk' in model_name else model_cfg.get('model_path')

    logger.info(f"Worker {cuda_id}/{world_size}: {onnx_name} on cuda:{cuda_id}, {len(my_files)} files, batch={batch_size}")

    try:
        providers = get_providers(cuda_id, use_trt)
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

        for i in tqdm(range(0, len(my_files), batch_size), desc=f"ASR-{cuda_id}", position=cuda_id):
            batch = my_files[i:i + batch_size]

            try:
                data, sr = load_batch(batch)
                kw = {"sample_rate": sr} if sr else {}
                results = model.recognize(data, **kw)
            except Exception as e:
                logger.error(f"Batch failed: {e}. Falling back to single-file mode.")
                results = []
                for f in batch:
                    try:
                        results.append(model.recognize(f))
                    except Exception as e2:
                        logger.error(f"File failed {f}: {e2}")
                        results.append("")

            if not isinstance(results, list):
                results = [results]

            texts = [extract_text(r) for r in results]
            ts = [format_timestamps(r) for r in results] if do_timestamps else None

            save_results(batch, texts, ts, output_suffix)

    except Exception as e:
        logger.exception(f"Worker {cuda_id} fatal error ({model_name}): {e}")


def check_consensus(audio_path: Path, model_names: List[str], consensus_num: int) -> bool:
    texts = []
    for mn in model_names:
        suffix = 'vosk' if 'vosk' in mn else mn
        tp = audio_path.with_name(f"{audio_path.stem}_{suffix}.txt")
        if tp.exists():
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
                    processed: List[str], consensus_num: int) -> List[str]:
    all_paths = get_audio_paths(src_path)
    if not all_paths:
        return []

    valid = [p for p in all_paths if not p.with_name(f"{p.stem}_{output_suffix}.txt").exists()]

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
    config = load_config(args.config_path, 'transcription')
    model_names = config.get('model_names', ['giga_rnnt'])
    src_path = config.get('podcasts_path', '.')
    consensus_num = config.get('consensus_num', 0)

    num_gpus = get_gpu_count()
    if num_gpus == 0:
        logger.error("No CUDA GPUs detected. GPU required for transcription.")
        return

    logger.info(f"{num_gpus} GPU(s) detected. Starting transcription pipeline.")
    if consensus_num > 0:
        logger.info(f"Consensus mode: {consensus_num} models must agree")

    for idx, model_name in enumerate(model_names):
        logger.info(f"=== [{idx + 1}/{len(model_names)}] {model_name} ===")

        output_suffix = 'vosk' if 'vosk' in model_name else model_name
        processed = model_names[:idx] if consensus_num > 0 else []
        paths = get_valid_paths(src_path, output_suffix, processed, consensus_num)

        if not paths:
            logger.info(f"No files to process for {model_name}")
            continue

        logger.info(f"{len(paths)} files to process")

        if num_gpus == 1:
            run_worker(0, 1, model_name, paths, config)
        else:
            procs = []
            for gid in range(num_gpus):
                p = mp.Process(
                    target=run_worker,
                    args=(gid, num_gpus, model_name, paths, config)
                )
                p.start()
                procs.append(p)

            for p in procs:
                p.join()

            failed = [p.exitcode for p in procs if p.exitcode != 0]
            if failed:
                logger.error(f"Workers failed with exit codes: {failed}")

    if config.get('use_rover', False):
        logger.info("ROVER aggregation...")
        try:
            from src.transcription.rover import ROVERWrapper
            ROVERWrapper(podcasts_path=src_path, model_names=model_names).aggregate_and_save()
            logger.info("ROVER done.")
        except ImportError:
            logger.warning("ROVER module not available, skipping")
        except Exception as e:
            logger.error(f"ROVER failed: {e}")

    logger.info("Transcription pipeline complete!")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="ASR Transcription (onnx-asr)")
    parser.add_argument("--config_path", type=str, required=True)
    main(parser.parse_args())
