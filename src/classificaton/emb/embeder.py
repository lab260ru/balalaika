import os
import yaml
import torchaudio.compliance.kaldi as kaldi
import torch
import torchaudio
from silero_vad import load_silero_vad, get_speech_timestamps  
from loguru import logger

from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint

class ResNetEmbedder:
    def __init__(self, model_path, device, resample_rate=16000, use_vad=True):
        self.device = device
        self.resample_rate = resample_rate
        self.use_vad = use_vad
        
        config_path = os.path.join(model_path, 'config.yaml')
        model_file = os.path.join(model_path, 'avg_model.pt')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file {model_file} not found")

        with open(config_path, 'r') as fin:
            configs = yaml.safe_load(fin)  

        self.model = get_speaker_model(configs['model'])(**configs['model_args'])
        load_checkpoint(self.model, model_file)
        self.model.to(self.device).eval()  

        if self.use_vad:
            self.vad_model = load_silero_vad()

    def __call__(self, path):
        wav = self._preprocess(path)
        duration = wav.shape[-1] / self.resample_rate
        fullness = duration / self.total_duration
        if wav is None: 
            return None
            
        feats = self.compute_fbank(wav)
        feats = feats.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(feats)
            outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
            
        return outputs[0].detach().cpu(), fullness

    def _preprocess(self, path):
        try:
            waveform, sr = torchaudio.load(path)
            if sr != self.resample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, 
                    orig_freq=sr, 
                    new_freq=self.resample_rate
                )
            self.total_duration = waveform.shape[-1] / self.resample_rate
        except Exception as e:
            logger.error(f"failed to load {path} - {e}")
            return None

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.use_vad:
            speech_segments = get_speech_timestamps(
                waveform.squeeze(0),
                self.vad_model,
                threshold=0.4,
                return_seconds=False
            )
            

            segments = [waveform[:, seg['start']:seg['end']] 
                    for seg in speech_segments]
            waveform = torch.cat(segments, dim=-1)
                
        return waveform

    def compute_fbank(self,
                    waveform,
                    sample_rate=16000,
                    num_mel_bins=80,
                    frame_length=25,
                    frame_shift=10,
                    cmn=True
        )->torch.Tensor:

        feat = kaldi.fbank(waveform,
                            num_mel_bins=num_mel_bins,
                            frame_length=frame_length,
                            frame_shift=frame_shift,
                            sample_frequency=sample_rate)
        if cmn:
            feat = feat - torch.mean(feat, 0)

        return feat