import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import List, Tuple

import gigaam
import pyctcdecode
import torch
import torchaudio
from loguru import logger
from tqdm import tqdm

from src.utils import get_audio_paths, load_config

# Global variables for each worker process
model = None
decoder = None
# Frame size in milliseconds for the GigaAM model, crucial for timestamp calculation
GIGA_AM_FRAME_SIZE_MS = 40


def init_process(
    model_name: str,
    device_str: str,
    lm_path: str,
    with_timestamps: bool,
):
    """
    Initializes the model and, if needed, the CTC decoder for each worker process.
    """
    global model, decoder
    logger.info(f"Initializing worker on {device_str}...")
    if not (with_timestamps and 'ctc' in model_name):
        logger.info("Timestamp generation requested CTC model. Decoder will not be initialized.")
        with_timestamps=False

    model = gigaam.load_model(model_name, device=device_str)

    if with_timestamps:
        if not lm_path:
            logger.warning("Timestamp generation requested without an LM path. Decoder will not be initialized.")
            decoder = None
            return

        logger.info(f"Building CTC decoder with LM: {lm_path}")
        try:
            vocab = model.decoding.tokenizer.vocab
            decoder = pyctcdecode.build_ctcdecoder(
                vocab,
                lm_path,
                alpha=0.5, 
                beta=1.0,
            )
        except Exception as e:
            logger.error(f"Failed to build CTC decoder: {e}")
            decoder = None


def to_simple_timestamps(word_timestamps: List[Tuple[str, Tuple[int, int]]]) -> str:
    output_lines = []
    sec_per_frame = GIGA_AM_FRAME_SIZE_MS / 1000.0
    for word, (start_frame, end_frame) in word_timestamps:
        start_time = start_frame * sec_per_frame
        end_time = end_frame * sec_per_frame
        output_lines.append(f"{word} {start_time:.3f} {end_time:.3f}")
    return "\n".join(output_lines)


def make_txt_and_tst(path: Path, with_timestamps: bool):
    """
    Transcribes an audio file. If with_timestamps is True and the decoder is available,
    it generates both a .txt (transcription) and a .tstt (timestamps) file.
    Otherwise, it only generates the .txt file.
    """
    txt_path = path.with_name(f"{path.stem}_giga.txt")
    tst_path = path.with_name(f"{path.stem}_giga.tst")
    
    if os.path.exists(txt_path):
        return

    # Timestamp-enabled path using the CTC decoder
    if not (with_timestamps and decoder):
        text = model.transcribe(str(path))
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
            return
    try:
        wav, sr = torchaudio.load(path)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0).unsqueeze(0) 
        
        wav = torchaudio.functional.resample(wav, sr, 16000)
        length = torch.full([1], wav.shape[-1])
        
        encoded, _ = model.forward(wav.to(model._device), length.to(model._device))
        logitst = model.head(encoded).squeeze(0).detach().cpu().numpy()
        
        # Use decode_beams to get timestamps
        beams = decoder.decode_beams(logitst, beam_width=100)
        
        # The top beam result with timestamps is typically at index 0
        best_beam = beams[0]
        word_timestamps = best_beam[2]

        # 1. Save the plain text transcription to _giga.txt
        plain_text = best_beam[0]
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(plain_text)
        
        # 2. Save the timestamps to _giga.tst
        tst_content = to_simple_timestamps(word_timestamps)
        with open(tst_path, "w", encoding="utf-8") as f:
            f.write(tst_content)

    except Exception as e:
        logger.error(f"Error processing {path} with timestamps: {e}")
        # Fallback to simple transcription if timestamping fails
        text = model.transcribe(str(path))
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)


def get_valid_audio_paths(src_path: str) -> List[Path]:
    """
    Getst all audio paths and filters out those that have already been transcribed.
    """
    all_audio_paths = get_audio_paths(src_path)
    valid_paths = []
    for audio_path in all_audio_paths:
        giga_path = audio_path.with_name(audio_path.stem + "_giga.txt")
        if not giga_path.exists():
            valid_paths.append(audio_path)
    return valid_paths


def main(args):
    config = load_config(args.config_path, 'transcription')

    model_name = args.model_name if args.model_name else config.get('model_name', 'rnnt')
    num_workers_per_gpu = args.num_workers if args.num_workers else config.get('num_workers', 4)
    src_path = args.podcasts_path if args.podcasts_path else config.get('podcasts_path', '../../../balalaika')
    lm_path = args.lm_path if args.lm_path else config.get('lm_path', 'ru.lm.bin')
    with_timestamps = args.with_timestamps if args.with_timestamps else config.get('with_timestamps', 'False')

    if with_timestamps and not lm_path:
        raise ValueError("Language model path (--lm_path) is required when using --with_timestamps.")

    all_audio_paths = get_valid_audio_paths(src_path)
    logger.info(f"Found {len(all_audio_paths)} audio files to process.")

    available_gpu_ids = list(range(torch.cuda.device_count()))
    num_gpus = len(available_gpu_ids)
    
    if num_gpus == 0:
        logger.error("No GPUs available. Exiting.")
        return

    logger.info(
        f"""
        Starting transcription with parameters:
        Source Path: {src_path}
        Model Name: {model_name}
        Timestamps Enabled: {with_timestamps}
        Language Model Path: {lm_path}
        Number of GPUs: {num_gpus} (IDs: {available_gpu_ids})
        Workers per GPU: {num_workers_per_gpu}
        Total Worker Processes: {num_gpus * num_workers_per_gpu}
        """
    )

    files_for_each_gpu = [[] for _ in range(num_gpus)]
    for i, path in enumerate(all_audio_paths):
        gpu_assignment_index = i % num_gpus
        files_for_each_gpu[gpu_assignment_index].append(path)

    all_futures = []
    executors = []

    task_fn = partial(make_txt_and_tst, with_timestamps=with_timestamps)

    for i, gpu_id in enumerate(available_gpu_ids):
        device_str = f'cuda:{gpu_id}'
        files_for_this_gpu = files_for_each_gpu[i]

        if not files_for_this_gpu:
            continue

        logger.info(f"Creating ProcessPoolExecutor for {device_str} with {num_workers_per_gpu} workers for {len(files_for_this_gpu)} files.")
        
        executor = ProcessPoolExecutor(
            max_workers=num_workers_per_gpu,
            initializer=init_process,
            initargs=(model_name, device_str, lm_path, with_timestamps),
        )
        executors.append(executor)

        for path in files_for_this_gpu:
            future = executor.submit(task_fn, path)
            all_futures.append(future)

    for future in tqdm(as_completed(all_futures), total=len(all_futures), desc="Overall Transcription Progress"):
        try:
            future.result()
        except Exception as e:
            logger.error(f"A task processing encountered an error: {e}")

    for executor in executors:
        executor.shutdown()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    torchaudio.set_audio_backend('soundfile')

    parser = argparse.ArgumentParser(
        description="Transcribe audio files in parallel using multiple GPUs."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--podcasts_path",
        type=str,
        help="Path to the directory containing audio files (e.g., MP3s)."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of worker processes per GPU for parallel processing."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model to use for transcription (e.g., 'rnnt', 'ctc')."
    )
    parser.add_argument(
        "--lm_path",
        type=str,
        help="Path to the language model binary file (e.g., 'ru.lm.bin') required for timestamps."
    )
    parser.add_argument(
        '--with_timestamps',
        type=bool,
        help="Enable to generate tst files with word timestamps."
    )

    args = parser.parse_args()
    main(args)