import argparse
import os
import torch
import torch.multiprocessing as mp
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Dict
from loguru import logger
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.utils import get_audio_paths, load_config

from nisqab.core.model_torch import model_init
from nisqab.utils.dataset import NISQADataset, collate_fn
from nisqab.utils.audio_cache import create_audio_length_cache
from nisqab.utils.audio_sampler import LengthBasedBatchSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)

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
    Worker function running on a dedicated GPU.
    """
    my_files = file_paths[rank::world_size]
    if not my_files:
        logger.info(f"Worker {rank}: No files to process.")
        return

    device_str = f"cuda:{rank}"
    device = torch.device(device_str)
    
    # Читаем конфиг
    bs = config.get('bs', 32)
    num_workers = config.get('num_workers_nisqa', 4) 
    
    cache_dir = Path(config.get('cache_path', './cache')) / f'nisqa_temp_worker_{rank}'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / 'audio_lengths.json'

    worker_output_path = final_output_path.with_name(f"{final_output_path.stem}_part_{rank}.csv")

    logger.info(f"[{device_str}] Starting worker. Processing {len(my_files)} files.")

    try:
        nisqa_config_path = Path(config.get('nisqa_config_path', 'configs/nisqa_b.yaml'))
        with open(nisqa_config_path, "r") as ymlfile:
            args_yaml = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
        model = model_init(args_yaml)
        model = model.to(device)
        model.eval()
        

        audio_lengths = create_audio_length_cache(
            file_paths=my_files,
            cache_file=str(cache_file),
            num_workers=min(num_workers, cpu_count() // world_size),
            force_rebuild=False
        )

        dataset = NISQADataset(
            file_paths=my_files,
            audio_lengths=audio_lengths,
            seg_length=15, seg_hop=1, max_length=2000,
            n_fft=4096, hop_length=0.01, win_length=0.02, n_mels=48, fmax=20000
        )
        
        sampler = LengthBasedBatchSampler(
            file_paths=my_files,
            audio_lengths=audio_lengths,
            batch_size=bs,
            drop_last=False,
            shuffle=True 
        )

        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        model = model.to(dtype)
        
        results_buffer = []
        save_every = 1000 
        
        with torch.inference_mode(): 
            for batch in tqdm(dataloader, desc=f"NISQA-{rank}", position=rank):
                x_batch = batch['x_spec_seg'].to(device, dtype=dtype)
                n_wins_batch = batch['n_wins'].to(device)
                
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    outputs = model(x_batch, n_wins_batch)
                
                outputs_cpu = outputs.float().cpu() 
                
                for i in range(len(batch['file_paths'])):
                    MOS, NOI, DISC, COL, LOUD = outputs_cpu[i].tolist()
                    results_buffer.append({
                        'filepath': batch['file_paths'][i],
                        'MOS': MOS, 'NOI': NOI, 'DISC': DISC, 'COL': COL, 'LOUD': LOUD
                    })

                if len(results_buffer) >= save_every:
                    save_chunk(results_buffer, worker_output_path)
                    results_buffer = []

        save_chunk(results_buffer, worker_output_path)
        logger.success(f"[{device_str}] Finished. Saved to {worker_output_path}")

    except Exception as e:
        logger.exception(f"Worker {rank} failed: {e}")


def combine_results(final_output_path: Path, num_parts: int):
    """Merges NISQA partial CSVs into the main balalaika.csv, filling MOS columns."""
    logger.info("Combining NISQA partial results...")

    nisqa_dfs = []
    for i in range(num_parts):
        part_path = final_output_path.with_name(f"{final_output_path.stem}_part_{i}.csv")
        if part_path.exists():
            try:
                nisqa_dfs.append(pd.read_csv(part_path))
                os.remove(part_path)
            except Exception as e:
                logger.error(f"Error reading {part_path}: {e}")

    if not nisqa_dfs:
        logger.warning("No NISQA results found to combine.")
        return

    nisqa_df = pd.concat(nisqa_dfs, ignore_index=True).drop_duplicates(subset=['filepath'])
    nisqa_df = nisqa_df.set_index('filepath')

    if final_output_path.exists():
        main_df = pd.read_csv(final_output_path).set_index('filepath')
        main_df = main_df.combine_first(nisqa_df)
        main_df = main_df.reset_index()
    else:
        main_df = nisqa_df.reset_index()

    main_df.to_csv(final_output_path, index=False)
    mos_filled = main_df['MOS'].notna().sum() if 'MOS' in main_df.columns else 0
    logger.success(f"Combined results saved to {final_output_path}. Total: {len(main_df)}, with MOS: {mos_filled}")


def get_unprocessed_audio_paths(podcasts_path: Path, result_csv_path: Path) -> List[str]:
    """Get list of audio paths that don't have MOS scores in the result CSV yet."""
    all_audio_paths = get_audio_paths(str(podcasts_path))
    all_paths_str = [str(p.resolve()) for p in all_audio_paths]

    if not result_csv_path.exists():
        return all_paths_str

    logger.info(f"Filtering existing results from: {result_csv_path}")
    try:
        df = pd.read_csv(result_csv_path)
        if 'filepath' not in df.columns or 'MOS' not in df.columns:
            return all_paths_str

        processed = set(
            df.loc[df['MOS'].notna(), 'filepath'].astype(str).tolist()
        )

        unprocessed = [p for p in all_paths_str if p not in processed]
        logger.info(f"Already processed (have MOS): {len(processed)}, remaining: {len(unprocessed)}")
        return unprocessed
    except Exception as e:
        logger.warning(f"Could not read existing CSV ({e}), processing all files.")
        return all_paths_str


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
        logger.error("No GPU detected. NISQA is too slow on CPU for large datasets.")
        return
    
    logger.info(f"Detected {available_gpus} GPUs. Preparing pipeline...")

    unprocessed_files = get_unprocessed_audio_paths(podcasts_path, final_output_path)
    
    if not unprocessed_files:
        logger.success("All files processed. Exiting.")
        return
    
    logger.info(f"Total files to process: {len(unprocessed_files)}")

    try:
        mp.spawn(
            run_inference_worker,
            args=(available_gpus, unprocessed_files, config, final_output_path),
            nprocs=available_gpus,
            join=True
        )
    except Exception as e:
        logger.critical(f"Multiprocessing failed: {e}")
    
    combine_results(final_output_path, available_gpus)


if __name__ == "__main__":
    main()