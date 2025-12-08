import os
import json
import torch
import torchaudio
from typing import List, Tuple, Optional
import miniaudio
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from loguru import logger

class GigaAudioDataset(Dataset):
    def __init__(self, audio_paths: List[str], target_sr: int = 16000):
        self.audio_paths = audio_paths
        self.target_sr = target_sr

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        try:
            if not os.path.exists(path):
                return None, path
            
            waveform, sr = torchaudio.load(path)
            
            if sr != self.target_sr:
                waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            return waveform.squeeze(0), path
            
        except Exception as e:
            logger.warning(f"Error loading {path}: {e}")
            return None, path


class LengthGroupedSampler(Sampler):
    def __init__(self, data_source, json_cache_path: str, batch_size: int):
        self.data_source = data_source
        self.batch_size = batch_size
        self.json_cache_path = json_cache_path
        self.indices = self._build_indices()

    def _build_indices(self):
        length_cache = {}
        if os.path.exists(self.json_cache_path):
            try:
                with open(self.json_cache_path, 'r') as f:
                    length_cache = json.load(f)
            except Exception as e:
                logger.warning(f"Ошибка чтения кеша длин {self.json_cache_path}: {e}")

        pairs = []
        missing_count = 0
        
        for idx, path in enumerate(self.data_source.audio_paths):
            length = length_cache.get(path)
            
            if length is None:
                length = 5.0
                missing_count += 1
            
            pairs.append((idx, length))
        
        if missing_count > 0:
            logger.info(f"For {missing_count} files the length was not found in the cache (default used).")

        pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [p[0] for p in pairs]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.data_source)


def collate_giga(batch: List[Tuple[Optional[torch.Tensor], str]]):
    batch = [item for item in batch if item[0] is not None]
    
    if not batch:
        return None, None, []

    waveforms = [item[0] for item in batch]
    paths = [item[1] for item in batch]

    lengths = torch.tensor([w.size(0) for w in waveforms], dtype=torch.long)
    padded_wavs = pad_sequence(waveforms, batch_first=True, padding_value=0.0)

    return padded_wavs, lengths, paths


class ToneAudioDataset(Dataset):
    def __init__(self, audio_paths: List[str], target_sr: int = 8000):
        self.audio_paths = audio_paths
        self.target_sr = target_sr

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        try:
            if not os.path.exists(path):
                return None, path

            audio = miniaudio.decode_file(str(path), nchannels=1, sample_rate=self.target_sr)
            audio_np = np.asarray(audio.samples, dtype=np.int16).astype(np.int32)
            
            return audio_np, path

        except Exception as e:
            logger.warning(f"Error processing {path} for Tone: {e}")
            return None, path

def collate_tone(batch: List[Tuple[Optional[np.ndarray], str]]):
    valid_batch = [item for item in batch if item[0] is not None]
    
    if not valid_batch:
        return [], []

    audios = [item[0] for item in valid_batch]
    paths = [item[1] for item in valid_batch]
    
    return audios, paths


class VoskAudioDataset(Dataset):
    def __init__(self, audio_paths: List[str], target_sr: int = 16000):
        self.audio_paths = audio_paths
        self.target_sr = target_sr

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        try:
            if not os.path.exists(path):
                return None, path
            
            waveform, sr = torchaudio.load(path)
            
            if sr != self.target_sr:
                waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            return waveform.squeeze(0), path
            
        except Exception as e:
            logger.warning(f"Error loading {path}: {e}")
            return None, path

def collate_vosk(batch: List[Tuple[Optional[torch.Tensor], str]]):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return [], []

    waveforms = [item[0] for item in batch]
    paths = [item[1] for item in batch]

    return waveforms, paths