import argparse
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import torch
from loguru import logger
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer

from src.utils import load_config, get_audio_paths, process_token, read_file_content

torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

model = None 

def init_process(
    model_name: str,
    device: str
    ) -> None:
    global model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        strip_accents=False,
        add_prefix_space=True
    )

    model = pipeline(
        "ner",
        model=model_name,
        tokenizer=tokenizer,
        aggregation_strategy="first",
        device=device
    )

def make_punct_txt(
    path: Path
    ):

    src_text = read_file_content(path)

    punct_path = path.with_name(path.name.replace("_rover.txt", "_punct.txt"))

    if str(path).endswith('_punct.txt') or str(path).endswith('_accent.txt') or os.path.exists(punct_path):
        return
    
    preds = model(src_text)

    tokens = [
        process_token(item["word"].strip(), item["entity_group"])
        for item in preds
    ]
    
    output = " ".join(tokens).strip()
    
    with open(punct_path, "w", encoding="utf-8") as f:
        f.write(output)

def get_valid_txt_paths(src_path: str) -> List[str]:
    all_audio_paths = get_audio_paths(src_path)
    
    valid_paths = []
    for audio_path in all_audio_paths:
        giga_path = audio_path.with_name(audio_path.stem + "_rover.txt")
        punct_path = audio_path.with_name(audio_path.stem + "_punct.txt")
        
        if os.path.exists(giga_path) and not os.path.exists(punct_path):
            valid_paths.append(giga_path)
    
    return valid_paths


def main(args):
    config = load_config(args.config_path, 'punctuation')
    num_workers_per_gpu = config.get('num_workers', 4)
    model_name = config.get('model_name', 'RUPunct/RUPunct_big')
    podcasts_path = config.get('podcasts_path', '../../../balalaika')

    all_text_files = get_valid_txt_paths(podcasts_path)

    available_gpu_ids = list(range(torch.cuda.device_count()))
    num_gpus = len(available_gpu_ids)

    if num_gpus == 0:
        logger.error("No GPUs available. Exiting.")
        return

    logger.info(
        f"""
        Starting punctuation restoration with parameters:
        Source Path: {podcasts_path}
        Model Name: {model_name}
        Number of GPUs: {num_gpus} (IDs: {available_gpu_ids})
        Workers per GPU: {num_workers_per_gpu}
        Total Worker Processes: {num_gpus * num_workers_per_gpu}
        """
    )

    files_for_each_gpu = [[] for _ in range(num_gpus)]
    for i, path in enumerate(all_text_files):
        gpu_assignment_index = i % num_gpus
        files_for_each_gpu[gpu_assignment_index].append(path)

    all_futures = []
    executors = []

    for i, gpu_id in enumerate(available_gpu_ids):
        device_str = f'cuda:{gpu_id}'
        files_for_this_gpu = files_for_each_gpu[i]

        if not files_for_this_gpu:
            continue

        logger.info(f"Creating ProcessPoolExecutor for {device_str} with {num_workers_per_gpu} workers for {len(files_for_this_gpu)} files.")
        
        executor = ProcessPoolExecutor(
            max_workers=num_workers_per_gpu,
            initializer=init_process,
            initargs=(model_name, device_str)
        )
        executors.append(executor)

        for path in files_for_this_gpu:
            future = executor.submit(make_punct_txt, path)
            all_futures.append(future)

    logger.info(f"Submitted all {len(all_futures)} tasks across {len(executors)} GPU(s). Waiting for completion...")

    for future in tqdm(as_completed(all_futures), total=len(all_futures), desc="Overall Punctuation Progress"):
        try:
            future.result()
        except Exception as e:
            logger.error(f"A task processing encountered an error (already logged by worker): {e}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Punctuation restoration script using multiple GPUs.")
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the configuration file"
        )

    args = parser.parse_args()
    main(args)