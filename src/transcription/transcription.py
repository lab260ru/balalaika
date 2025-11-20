import argparse
import torch
import torch.multiprocessing as mp
from pathlib import Path
from typing import List
from loguru import logger
from tqdm import tqdm

from src.transcription.transcripton_base import (
    GigaAMWrapper, 
    ToneWrapper, 
    VOSKCUDAWrapper, 
    ROVERWrapper
)
from src.transcription.transcripton_dataset import (
    GigaAudioDataset, collate_giga, LengthGroupedSampler,
    ToneAudioDataset, collate_tone,
    VoskAudioDataset, collate_vosk
)
from src.utils import get_audio_paths, load_config

torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

SUPPORTED_TIME_STAMPS = ['giga_ctc_lm', 'tone']

def save_results(paths: List[str], texts: List[str], timestamps: List[str], model_suffix: str):
    """Helper to save text and timestamp files."""
    for path_str, text, ts_content in zip(paths, texts, timestamps):
        path = Path(path_str)
        
        txt_path = path.with_name(f"{path.stem}_{model_suffix}.txt")
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            logger.error(f"Failed to write TXT for {path.name}: {e}")

        if ts_content:
            tst_path = path.with_name(f"{path.stem}_{model_suffix}.tst")
            try:
                with open(tst_path, "w", encoding="utf-8") as f:
                    f.write(ts_content)
            except Exception as e:
                logger.error(f"Failed to write TST for {path.name}: {e}")


def run_inference_on_device(cuda_id: int, world_size: int, model_name: str, all_file_paths: List[Path], config: dict):
    if not all_file_paths:
        return

    my_files = all_file_paths[cuda_id::world_size]
    
    if not my_files:
        return

    device_str = f"cuda:{cuda_id}"
    logger.info(f"Worker {cuda_id}/{world_size} started for '{model_name}' on {device_str}. Processing {len(my_files)} files.")

    model_config = config.get('giga') if 'giga' in model_name else config.get(model_name, {})
    batch_size = model_config.get('batch_size', 8)
    num_workers = model_config.get('num_workers', 4) 

    lengths_cache_path = config.get('audio_lengths_cache', './cache/audio_lengths_cache.json')

    model_name_for_output = 'vosk' if 'vosk' in model_name else model_name
    
    with_timestamps = config.get('with_timestamps', False)
    timestamps_supported = any(x in model_name for x in SUPPORTED_TIME_STAMPS)
    process_timestamps = with_timestamps and timestamps_supported
    
    my_files_str = [str(p) for p in my_files]

    try:
        if 'giga' in model_name:
            model = GigaAMWrapper(model_id=model_name, device=device_str, **model_config)
            
            dataset = GigaAudioDataset(my_files_str, target_sr=16000)
            sampler = LengthGroupedSampler(dataset, lengths_cache_path, batch_size)
            
            loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                sampler=sampler, 
                num_workers=num_workers,
                collate_fn=collate_giga,
                pin_memory=True,
                persistent_workers=True if num_workers > 0 else False
            )
            
            for batch_wavs, batch_lengths, batch_paths in tqdm(loader, desc=f"Giga-{cuda_id}", position=cuda_id):
                if batch_wavs is None: continue
                
                if process_timestamps:
                    texts, tstamps = model.transcribe_tensors_with_timestamps(batch_wavs, batch_lengths)
                else:
                    texts = model.transcribe_tensors(batch_wavs, batch_lengths)
                    tstamps = [''] * len(texts)
                
                save_results(batch_paths, texts, tstamps, model_name_for_output)

        elif 'tone' in model_name:
            model = ToneWrapper(model_id=model_name, device=device_str, **model_config)
            
            dataset = ToneAudioDataset(my_files_str, target_sr=8000)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_tone,
                pin_memory=False
            )
            
            for batch_audios, batch_paths in tqdm(loader, desc=f"Tone-{cuda_id}", position=cuda_id):
                if not batch_audios: continue
                
                if process_timestamps:
                    texts, tstamps = model.transcribe_audio_data_with_timestamps(batch_audios)
                else:
                    texts = model.transcribe_audio_data(batch_audios)
                    tstamps = [''] * len(texts)
                    
                save_results(batch_paths, texts, tstamps, model_name_for_output)

        elif 'vosk' in model_name:
            model = VOSKCUDAWrapper(model_id=model_config.get('vosk_path'), device=device_str, **model_config)
            
            dataset = VoskAudioDataset(my_files_str, target_sr=16000)
            sampler = LengthGroupedSampler(dataset, lengths_cache_path, batch_size)
            
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate_vosk,
                pin_memory=True
            )
            
            for batch_waveforms, batch_paths in tqdm(loader, desc=f"Vosk-{cuda_id}", position=cuda_id):
                if not batch_waveforms: continue
                
                texts = model.transcribe_batch_data(batch_waveforms)
                tstamps = [''] * len(texts)
                
                save_results(batch_paths, texts, tstamps, model_name_for_output)
        
        else:
            logger.error(f"Unknown model type: {model_name}")

    except Exception as e:
        logger.exception(f"Critical error in worker {cuda_id} for model {model_name}: {e}")


def get_valid_audio_paths(src_path: str, model_name_for_output: str) -> List[Path]:
    all_audio_paths = get_audio_paths(src_path)
    if not all_audio_paths:
        return []

    valid_paths = [
        p for p in all_audio_paths 
        if not p.with_name(f"{p.stem}_{model_name_for_output}.txt").exists()
    ]
    return valid_paths


def main(args):
    config = load_config(args.config_path, 'transcription')
    model_names = config.get('model_names', ['giga_rnnt'])
    src_path = config.get('podcasts_path', '.')

    available_gpu_ids = list(range(torch.cuda.device_count()))
    if not available_gpu_ids:
        logger.error("No CUDA GPUs detected. This optimized script requires GPU.")
        return
    
    num_gpus = len(available_gpu_ids)
    logger.info(f"Detected {num_gpus} GPUs. Starting processing pipeline.")

    for model_name in model_names:
        logger.info(f"=== Processing model: {model_name} ===")
        
        model_name_for_output = 'vosk' if 'vosk' in model_name else model_name
        
        all_paths = get_valid_audio_paths(src_path, model_name_for_output)
        
        if not all_paths:
            logger.info(f"No new files for {model_name}. Skipping.")
            continue
            
        logger.info(f"Total files to process: {len(all_paths)}")
        
        try:
            mp.spawn(
                run_inference_on_device,
                args=(num_gpus, model_name, all_paths, config),
                nprocs=num_gpus,
                join=True
            )
        except Exception as e:
            logger.error(f"Multiprocessing error: {e}")
            
    logger.info("Starting ROVER processing...")
    try:
        rover_wrapper = ROVERWrapper(podcasts_path=src_path, model_names=model_names)
        rover_wrapper.aggregate_and_save()
        logger.info("ROVER processing finished.")
    except Exception as e:
        logger.error(f"ROVER failed: {e}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Optimized GPU Audio Transcription")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()
    
    main(args)