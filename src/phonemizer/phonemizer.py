import argparse
from pathlib import Path
from typing import Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import torch
from loguru import logger
from tryiparu import G2PModel
from tqdm import tqdm

from src.utils import get_txt_paths, load_config, read_file_content

g2p_model: Any = None

def init_process(device_str: str):
    global g2p_model
    g2p_model = G2PModel(
        load_dataset=True,
        device=device_str
    )

def process_text(text_path: Path):
    output_path = text_path.with_name(f"{text_path.stem}_phonemes.txt")
    
    if output_path.exists():
        return

    text = read_file_content(text_path)   
    phonemes = g2p_model(text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(" ".join(phonemes))

def get_valid_text_paths(src_path: str) -> List[Path]:
    all_paths = get_txt_paths(src_path, '_giga.txt')
    valid_paths = []
    
    for giga_path in all_paths:
        giga_path = Path(giga_path)
        phonemes_path = giga_path.with_name(f"{giga_path.stem}_phonemes.txt")
        
        if not phonemes_path.exists():
            valid_paths.append(giga_path.absolute())

    return valid_paths

def main(args):
    config = load_config(args.config_path, 'phonemizer')
    num_workers = args.num_workers if args.num_workers else config.get('num_workers', 4)
    src_path_str = args.podcasts_path if args.podcasts_path else config.get('podcasts_path', '../../../podcasts')
    
    all_text_paths = get_valid_text_paths(src_path_str)
    logger.info(f"Found {len(all_text_paths)} text files to process")
    
    available_gpu_ids = list(range(torch.cuda.device_count()))
    num_gpus = len(available_gpu_ids)
    
    if num_gpus == 0:
        logger.error("No GPUs available. Exiting.")
        return
    
    logger.info(
        f"""
        Starting phoneme conversion with parameters:
        Source Path: {src_path_str}
        Number of GPUs: {num_gpus} (IDs: {available_gpu_ids})
        Workers per GPU: {num_workers}
        Total Worker Processes: {num_gpus * num_workers}
        """
    )
    
    files_for_each_gpu = [[] for _ in range(num_gpus)]
    for i, path in enumerate(all_text_paths):
        gpu_assignment_index = i % num_gpus
        files_for_each_gpu[gpu_assignment_index].append(path)

    all_futures = []
    executors = []
    
    for i, gpu_id in enumerate(available_gpu_ids):
        device_str = f'cuda:{gpu_id}'
        files_for_this_gpu = files_for_each_gpu[i]
        
        if not files_for_this_gpu:
            continue
            
        logger.info(f"Creating ProcessPoolExecutor for {device_str} with {num_workers} workers for {len(files_for_this_gpu)} files.")
        
        executor = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_process,
            initargs=(device_str,)
        )
        executors.append(executor)
        
        for path in files_for_this_gpu:
            future = executor.submit(process_text, path)
            all_futures.append(future)
    
    for future in tqdm(as_completed(all_futures), total=len(all_futures), desc="Overall Progress"):
        try:
            future.result()
        except Exception as e:
            logger.error(f"A task processing encountered an error: {e}")
            break

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    
    parser = argparse.ArgumentParser(
        description="Parallel Text-to-Phoneme Conversion with Multi-GPU Support"
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

    args = parser.parse_args()
    main(args)