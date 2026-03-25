import argparse
import os
import re
import torch
import torch.multiprocessing as mp
import pandas as pd
import torchaudio
from pathlib import Path
from typing import List, Dict
from loguru import logger
from tqdm import tqdm

from src.utils.utils import get_audio_paths, load_config

torch.backends.cuda.matmul.allow_tf32 = True

def save_chunk(results: List[Dict], output_path: Path):
    """Saves a chunk of results to CSV, appending if exists."""
    if not results:
        return
    df = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = not output_path.exists()
    df.to_csv(output_path, mode='a', header=header, index=False)

def run_inference_worker(rank: int, world_size: int, file_paths: List[str], config: dict, final_output_path: Path):
    """
    Worker function running on a dedicated GPU for DistillMOS.
    """
    my_files = file_paths[rank::world_size]
    if not my_files:
        logger.info(f"Worker {rank}: No files to process.")
        return

    device = torch.device(f"cuda:{rank}")
    worker_output_path = final_output_path.with_name(f"distillmos_part_{rank}.csv")

    logger.info(f"[cuda:{rank}] Loading DistillMOS model...")
    try:
        import distillmos
        sqa_model = distillmos.ConvTransformerSQAModel()
        sqa_model.to(device)
        sqa_model.eval()
    except Exception as e:
        logger.error(f"Failed to load distillmos model on worker {rank}: {e}")
        return

    results_buffer = []
    save_every = 500

    logger.info(f"[cuda:{rank}] Starting inference for {len(my_files)} files.")

    for path_str in tqdm(my_files, desc=f"DistillMOS-{rank}", position=rank):
        try:
            x, sr = torchaudio.load(path_str)

            if x.shape[0] > 1:
                x = x[0, None, :]
            
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000).to(device)
                x = resampler(x.to(device))
            else:
                x = x.to(device)

            with torch.no_grad():
                mos = sqa_model(x)
                mos_val = mos[0].item()

            results_buffer.append({
                'filepath': path_str,
                'DistillMOS': mos_val
            })

        except Exception as e:
            logger.warning(f"Error processing {path_str}: {e}")
            continue

        if len(results_buffer) >= save_every:
            save_chunk(results_buffer, worker_output_path)
            results_buffer = []

    save_chunk(results_buffer, worker_output_path)
    logger.success(f"[cuda:{rank}] Finished.")

def combine_results(final_output_path: Path, num_parts: int):
    """Merges partial CSVs into the main balalaika.csv securely."""
    logger.info("Combining DistillMOS results...")
    dfs = []
    for i in range(num_parts):
        part_path = final_output_path.with_name(f"distillmos_part_{i}.csv")
        if part_path.exists():
            try:
                dfs.append(pd.read_csv(part_path))
                os.remove(part_path)
            except Exception as e:
                logger.error(f"Error reading {part_path}: {e}")

    if not dfs:
        logger.warning("No DistillMOS results found to merge.")
        return

    new_df = pd.concat(dfs, ignore_index=True)

    if final_output_path.exists():
        logger.info(f"Safely merging with existing CSV: {final_output_path}")
        main_df = pd.read_csv(final_output_path)
        
        main_df.set_index('filepath', inplace=True)
        new_df.set_index('filepath', inplace=True)
        
        main_df = main_df.combine_first(new_df).reset_index()
    else:
        main_df = new_df

    if 'is_single_speaker' in main_df.columns:
        main_df.drop(columns=['is_single_speaker'], inplace=True)

    base_cols = ['filepath', 'speaker_id', 'start', 'end', 'total_duration', 
                 'playlist_id', 'podcast_id', 'silence_percent', 'max_silence_duration', 'DistillMOS']
    final_cols = [c for c in base_cols if c in main_df.columns] + [c for c in main_df.columns if c not in base_cols]
    
    main_df[final_cols].to_csv(final_output_path, index=False)
    logger.success(f"Combined successfully. Total rows: {len(main_df)}")

def get_unprocessed_paths(podcasts_path: Path, result_csv_path: Path) -> List[str]:
    """Finds all audio files that haven't been processed by DistillMOS yet."""
    all_audio_paths = [str(Path(p).resolve()) for p in get_audio_paths(str(podcasts_path))]

    if not result_csv_path.exists():
        return all_audio_paths

    try:
        df = pd.read_csv(result_csv_path)
        if 'DistillMOS' not in df.columns:
            return all_audio_paths

        processed = set(
            df.dropna(subset=['DistillMOS'])['filepath']
            .apply(lambda p: str(Path(p).resolve()))
            .tolist()
        )

        return [p for p in all_audio_paths if p not in processed]
    except Exception as e:
        logger.warning(f"Could not read CSV to filter paths: {e}. Processing all chunks.")
        return all_audio_paths

def main():
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config_path, 'separation') 
    podcasts_path = Path(config.get('podcasts_path', '.'))
    final_output_path = podcasts_path / 'balalaika.csv'

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        logger.error("No GPU detected.")
        return
    
    unprocessed = get_unprocessed_paths(podcasts_path, final_output_path)
    if not unprocessed:
        logger.success("All small audio files already have a DistillMOS score. Exiting.")
        return

    logger.info(f"Processing {len(unprocessed)} files on {available_gpus} GPUs.")

    try:
        mp.spawn(
            run_inference_worker,
            args=(available_gpus, unprocessed, config, final_output_path),
            nprocs=available_gpus,
            join=True
        )
    except Exception as e:
        logger.critical(f"Multiprocessing failed: {e}")
    
    combine_results(final_output_path, available_gpus)

if __name__ == "__main__":
    main()