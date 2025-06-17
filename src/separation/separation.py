import argparse
import os
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml
import shutil

import numpy as np
import pandas as pd
import torch
import torchaudio
from dotenv import load_dotenv
from loguru import logger
from pyannote.audio import Pipeline
from tqdm import tqdm
from transformers import pipeline 

from src.libs.nisqa.core.model_torch import model_init
from src.libs.nisqa.utils.process_utils import process
from src.utils import load_config, get_audio_paths

_global_worker = None

class Worker:
    def __init__(
        self,
        use_nisqa: bool,
        use_mono: bool,
        one_speaker: bool,
        nisqa_config_path: str,
        gpu_id: int,
        hf_token: str,
    ):
        self.use_nisqa = use_nisqa
        self.use_mono = use_mono
        self.one_speaker = one_speaker
        self.nisqa_config_path = nisqa_config_path
        self.hf_token = hf_token
        self.device = f"cuda:{gpu_id}"
        self._init_models()

    def _init_models(self):
        
        torch.cuda.set_device(self.device)

        if self.use_nisqa:
            with open(self.nisqa_config_path, "r") as f:
                args_yaml = yaml.load(f, Loader=yaml.FullLoader)

            self.nisqa_device = self.device
            args = {**args_yaml, "inf_device": self.device}
            
            self.nisqa_model, self.h0, self.c0 = model_init(args)
            self.nisqa_args = args

        if self.use_mono:
            try:
                self.diarization_model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token
                ).to(torch.device(self.device))
            except Exception as e:
                logger.warning(f"{e} on {self.device}.")

    def process_audio(self, audio_path: str) -> Dict:
        audio_path = Path(audio_path)
        frame_duration = self.nisqa_args.get("frame") if self.use_nisqa else 0
        audio_frames, sr, audio = self._preprocess_audio(str(audio_path), frame_duration)

        is_mono = True 
        if self.use_mono:
            is_mono = self._check_single_speaker(audio, sr) 

        NOI = COL = DISC = LOUD = MOS = None
        if self.use_nisqa:
            avg_out = self._nisqua_predict(audio_frames, sr) 
            NOI, COL, DISC, LOUD, MOS = avg_out

        file_parts = audio_path.name.split('_')
        playlist_id = file_parts[-2] if len(file_parts) > 0 else 'N/A'
        podcast_id = file_parts[-1].split('.')[0] if len(file_parts) > 1 else 'N/A'

        return {
            'audio_path': '/'.join(audio_path.parts[-3:]),
            'is_mono': is_mono,
            'NOI': NOI,
            'COL': COL,
            'DISC': DISC,
            'LOUD': LOUD,
            'MOS': MOS,
            'playlist_id': playlist_id,
            'podcast_id': podcast_id,
            'start': file_parts[0] if len(file_parts) > 0 else 'N/A',
            'end': file_parts[1]  if len(file_parts) > 1 else 'N/A'
        }
        

    def _preprocess_audio(self, audio_path: str, frame_duration: int):
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] != 1:
            audio = torch.mean(audio, dim=0, keepdim=True) 
        audio = audio.squeeze(0) 

        audio = audio.to(self.device)
        frame_size = int(sr * frame_duration)
        if len(audio) % frame_size != 0:
            padding = frame_size - (len(audio) % frame_size)
            audio = torch.cat([audio, torch.zeros(padding, device=self.device)]) 
        frames = torch.split(audio, frame_size)
        
        return frames, sr, audio

    def _check_single_speaker(self, waveform: torch.Tensor, sr: int) -> bool:
      
        if not self.use_mono:
            return True 
        try:
            diarization = self.diarization_model({
                "waveform": waveform.unsqueeze(0), 
                "sample_rate": sr
            })
            is_single_speaker = len({speaker for _, _, speaker in diarization.itertracks(yield_label=True)}) == 1
            
            if not is_single_speaker and self.one_speaker:
                audio_path = str(waveform.audio_path) 
                base_path = os.path.splitext(audio_path)[0]  
                
                for ext in ['.mp3', '_giga.txt', '_punct.txt', '_accent.txt', '_e.txt', '_e_phonemes.txt']:
                    file_path = base_path + ext
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"Deleted {file_path} due to multiple speakers detected")
                
            return is_single_speaker
            
        except Exception as e:
            logger.error(f"Diarization error on {self.device}: {e}")
            return True 


    def _nisqua_predict(self, frames: List[torch.Tensor], sr: int) -> np.ndarray:
        if not self.use_nisqa:
            return np.array([None, None, None, None, None])
        
        outputs = []
        h, c = self.h0.clone().to(self.device), self.c0.clone().to(self.device) 
        
        for frame in frames:
            out, h, c = process(frame.to(self.device), sr, self.nisqa_model, h, c, self.nisqa_args)
            outputs.append(out[0].cpu().numpy())
        
        return np.mean(outputs, axis=0)


def _worker_initializer(
    use_nisqa: bool,
    use_mono: bool,
    one_speaker: bool,
    nisqa_config_path: str, 
    hf_token: str,
    gpu_id_assignment_queue
    ):
    global _global_worker
    gpu_id = None
    try:
        if not gpu_id_assignment_queue.empty():
            gpu_id = gpu_id_assignment_queue.get()
        
        _global_worker = Worker(
            use_nisqa=use_nisqa,
            use_mono=use_mono,
            one_speaker=one_speaker,
            nisqa_config_path=nisqa_config_path,
            gpu_id=gpu_id,
            hf_token=hf_token
            )
    except Exception as e:
        logger.error(f"Failed to initialize worker process on {f'GPU {gpu_id}' if gpu_id is not None else 'CPU'}: {e}")

def _process_audio_task(audio_path: str) -> Dict:
    global _global_worker
    
    if _global_worker is None:
        logger.error(f"Worker not initialized in this process. Cannot process {audio_path}.")
    try:
        result = _global_worker.process_audio(audio_path)
        return result
    except Exception as e:
        logger.error(f"Error during audio processing for {audio_path} by worker on {_global_worker.device}: {e}")
        return None
    finally:
        if _global_worker and _global_worker.device.startswith("cuda"):
            torch.cuda.empty_cache()

def main(args):
    hf_token = os.getenv("HF_TOKEN")
    config = load_config(args.config_path, 'separation')

    podcasts_path = args.podcasts_path if args.podcasts_path else config.get('podcasts_path', '/../../../podcasts')
    one_speaker = args.one_speaker if args.one_speaker else config.get('one_speaker', False)
    use_nisqa = args.use_nisqa if args.use_nisqa else config.get('use_nisqa', True)
    use_mono = args.use_mono if args.use_mono else config.get('use_mono', True)
    num_workers = args.num_workers if args.num_workers else config.get('num_workers', 4)
    nisqa_config_path = args.nisqa_config if args.nisqa_config else config.get('nisqa_config', '')

    num_gpus = torch.cuda.device_count()
    actual_max_workers = num_gpus
    gpu_ids_available = list(range(num_gpus))

    logger.info(f"""
                Using params:
                Podcasts path: {podcasts_path}
                One speaker : {one_speaker}
                Use NISQA : {use_nisqa}
                Use mono : {use_mono}
                num_workers : {num_workers}
                Number of GPUs detected: {num_gpus}
                """)

    audio_paths = get_audio_paths(podcasts_path)
    result_csv_path = Path(podcasts_path) / 'results.csv'


    if os.path.exists(result_csv_path):
        logger.info('csv file exists')
        df = pd.read_csv(result_csv_path)
        processed_audio_paths = set(df['audio_path'].tolist())
    else:
        processed_audio_paths = set()
        logger.info(f'csv does not exist: found {len(audio_paths) - len(processed_audio_paths)} files')

    audio_paths_to_process = [
            audio_path for audio_path in audio_paths 
            if str('/'.join(Path(audio_path).parts[-3:])) not in processed_audio_paths
        ]

        
    if not audio_paths:
        logger.error(f"No audio files found in {podcasts_path}")
        return
    
    results = []
    num_workers_per_gpu = num_workers
    manager = multiprocessing.Manager()
    gpu_id_assignment_queue = manager.Queue()

    for gid in gpu_ids_available:
        for _ in range(num_workers_per_gpu):
            gpu_id_assignment_queue.put(gid)

    actual_max_workers = len(gpu_ids_available) * num_workers_per_gpu

    if actual_max_workers == 0:
        actual_max_workers = 1 
        if gpu_id_assignment_queue.empty():
            gpu_id_assignment_queue.put(None)

    with ProcessPoolExecutor(
        max_workers=actual_max_workers,
        mp_context=multiprocessing.get_context('spawn'),
        initializer=_worker_initializer,
        initargs=(use_nisqa, use_mono, one_speaker, nisqa_config_path, hf_token, gpu_id_assignment_queue)
    ) as executor:
        futures = [executor.submit(_process_audio_task, str(path)) for path in audio_paths_to_process]

        with tqdm(total=len(audio_paths_to_process), desc="Processing files") as pbar:
            for future in as_completed(futures):
                result = future.result() 
                if result:
                    results.append(result)
                pbar.update(1)

        csv_path = Path(podcasts_path) / "results.csv"
        pd.DataFrame(results).to_csv(
            csv_path, 
            mode='a', 
            header=not csv_path.exists(), 
            index=False
        )
        logger.success(f"Processing completed successfully. Results saved to {result_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files using multiple GPUs")
    parser.add_argument("--config_path", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("--nisqa_config", type=str, help="Path to the NISQA YAML configuration file.")
    parser.add_argument("--podcasts_path", type=str, help="Path to the directory containing podcast audio files.")
    parser.add_argument("--use_nisqa", type=bool, 
                    help="Boolean flag indicating whether to use NISQA model for audio quality estimation.")
    parser.add_argument("--use_mono", type=bool,
                    help="Boolean flag indicating whether the input audio should be converted to mono before processing.")
    parser.add_argument("--one_speaker", type=bool, 
                        help="Boolean flag to indicate if only one speaker is expected per audio file")
    parser.add_argument("--num_workers", type=int, 
                        help="Boolean flag to indicate if only one speaker is expected per audio file")
    args = parser.parse_args()
    main(args)
