import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torchaudio
from dotenv import load_dotenv
from loguru import logger
from pyannote.audio import Pipeline
from tqdm import tqdm

from src.utils import get_audio_paths, load_config

torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


diarization_pipeline = None

def init_worker(hf_token: str, device_str: str):
    global diarization_pipeline
    logger.info(f"Initializing worker for diarization on {device_str}...")
    
    try:
        torch.cuda.set_device(device_str)
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            use_auth_token=hf_token
        ).to(torch.device(device_str))
        logger.info(f"Worker initialized successfully for diarization on {device_str}.")
    except Exception as e:
        logger.error(f"Failed to initialize worker on {device_str}: {e}")
        diarization_pipeline = None

def _save_rttm_file(diarization, rttm_path: str):
    with open(rttm_path, "w") as rttm_file:
        diarization.write_rttm(rttm_file)

def process_file(path: Path, one_speaker: bool) -> Dict:

    global diarization_pipeline
    if diarization_pipeline is None:
        logger.error(f"Diarization pipeline is not initialized. Skipping {path.name}.")
        return None

    try:
        # 1. Preprocess audio: Load, convert to mono
        audio, sr = torchaudio.load(path)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # 2. Perform diarization
        diarization = diarization_pipeline({
            "waveform": audio,
            "sample_rate": sr
        })
        
        # 3. Save RTTM and check speaker count
        base_path_str = str(path.with_suffix(''))
        rttm_path = base_path_str + '.rttm'
        _save_rttm_file(diarization=diarization, rttm_path=rttm_path)

        num_speakers = len({speaker for _, _, speaker in diarization.itertracks(yield_label=True)})
        is_single_speaker = num_speakers == 1

        # 4. Handle multi-speaker files if one_speaker mode is enabled
        if not is_single_speaker and one_speaker:
            logger.info(f"Multiple speakers ({num_speakers}) detected in {path.name}. Deleting associated files.")
            for ext in ['.mp3', '_giga.txt', '_punct.txt', '_accent.txt', '_e.txt', '_e_phonemes.txt', '.rttm']:
                file_to_delete = Path(base_path_str + ext)
                if file_to_delete.exists():
                    file_to_delete.unlink()
                    logger.info(f"Deleted {file_to_delete}")
        
        # 5. Extract metadata from the filename
        file_parts = path.name.split('_')
        playlist_id = file_parts[-2] if len(file_parts) > 1 else 'N/A'
        podcast_id = file_parts[-1].split('.')[0] if len(file_parts) > 0 else 'N/A'
        
        return {
            'filepath': str(path),
            'is_single_speaker': is_single_speaker,
            'playlist_id': playlist_id,
            'podcast_id': podcast_id,
            'start': file_parts[0] if len(file_parts) > 0 else 'N/A',
            'end': file_parts[1] if len(file_parts) > 1 else 'N/A'
        }

    except Exception as e:
        logger.error(f"Failed to process {path.name}: {e}")
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_unprocessed_audio_paths(podcasts_path: str, result_csv_path: str) -> List[Path]:
    all_audio_paths = get_audio_paths(podcasts_path)
    processed_audio_paths = []
    
    if result_csv_path.exists():
        logger.info(f"Resuming from existing results file: {result_csv_path}")
        df = pd.read_csv(result_csv_path)
        processed_audio_paths = df.set_index('filepath').to_dict('index')

    unprocessed_paths = [
        Path(path) for path in processed_audio_paths.keys()
        if not isinstance(
            processed_audio_paths[path].get('is_single_speaker'),
            bool
            )
    ]
    
    return unprocessed_paths


def main(args):
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    config = load_config(args.config_path, 'separation')

    podcasts_path = config.get('podcasts_path', './data')
    one_speaker = config.get('one_speaker', False)
    num_workers_per_gpu = config.get('num_workers_diarization', 1)

    result_csv_path = Path(podcasts_path) / 'balalaika.csv'
    available_gpu_ids = list(range(torch.cuda.device_count()))
    if not available_gpu_ids:
        logger.error("No GPUs available. This script requires at least one GPU. Exiting.")
        return

    logger.info(f"""
                Podcasts path: {podcasts_path}
                One speaker mode: {one_speaker}
                Workers per GPU: {num_workers_per_gpu}
                GPUs detected: {len(available_gpu_ids)}
                """)

    all_audio_paths = get_unprocessed_audio_paths(
        podcasts_path,
        result_csv_path
    )

    if not all_audio_paths:
        logger.success("All audio files have already been processed.")
        return
    
    logger.info(f"Found {len(all_audio_paths)} new audio files to process.")

    num_devices = len(available_gpu_ids)
    files_for_each_device = [[] for _ in range(num_devices)]
    for i, path in enumerate(all_audio_paths):
        device_assignment_index = i % num_devices
        files_for_each_device[device_assignment_index].append(path)

    all_futures = []
    executors = []
    task_fn = partial(process_file, one_speaker=one_speaker)

    for i, device_id in enumerate(available_gpu_ids):
        device_str = f'cuda:{device_id}'
        files_for_this_device = files_for_each_device[i]

        if not files_for_this_device:
            continue

        logger.info(f"Creating ProcessPoolExecutor for {device_str} with {num_workers_per_gpu} workers for {len(files_for_this_device)} files.")
        
        initializer_fn = partial(init_worker, hf_token=hf_token, device_str=device_str)

        executor = ProcessPoolExecutor(
            max_workers=num_workers_per_gpu,
            initializer=initializer_fn,
            mp_context=multiprocessing.get_context('spawn')
        )
        executors.append(executor)

        for path in files_for_this_device:
            future = executor.submit(task_fn, path)
            all_futures.append(future)

    results = []
    for future in tqdm(as_completed(all_futures), total=len(all_futures), desc="Processing audio files"):
        try:
            result = future.result()
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"A task encountered an error: {e}")

    for executor in executors:
        executor.shutdown(wait=True)

    if results:
        try:
            new_results_df = pd.DataFrame(results)
            main_df = pd.read_csv(result_csv_path)

            main_df = main_df.set_index('filepath')
            new_results_df = new_results_df.set_index('filepath')

            updated_df = new_results_df.combine_first(main_df)

            updated_df = updated_df.reset_index()

            original_cols = main_df.reset_index().columns.tolist()
            new_cols = [col for col in updated_df.columns if col not in original_cols]
            final_order = original_cols + new_cols
            final_order_unique = list(dict.fromkeys(final_order))
            updated_df = updated_df[final_order_unique]

            if 'Unnamed: 0' in updated_df.columns:
                updated_df = updated_df.drop('Unnamed: 0', axis=1)

            updated_df.to_csv(result_csv_path, index=False)
            logger.success(f"Processing complete. Updated/added diarization for {len(results)} rows in {result_csv_path}")
        
        except FileNotFoundError:
            logger.error(f"Error: The result file {result_csv_path} was not found. Please run the NISQA script first.")
        except Exception as e:
            logger.error(f"An error occurred while updating the CSV file: {e}")
    else:
        logger.warning("Processing finished, but no new results were generated.")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    torchaudio.set_audio_backend('soundfile')
    
    parser = argparse.ArgumentParser(description="Perform speaker diarization on audio files using multiple GPUs.")
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the main YAML configuration file."
    )

    args = parser.parse_args()
    main(args)