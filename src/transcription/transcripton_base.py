import json
from abc import ABC, abstractmethod
from typing import Tuple, List
from pathlib import Path
from collections import defaultdict

import torch
import torchaudio
import numpy as np
import miniaudio
from loguru import logger
import pyctcdecode
import gigaam
from tone import StreamingCTCPipeline, TextPhrase
from crowdkit.aggregation import ROVER
import pandas as pd
import sherpa_onnx
import wave
from src.utils import get_audio_paths, read_file_content

class ASRWrapper(ABC):
    """Abstract Base Class for ASR model wrappers."""

    @abstractmethod
    def __init__(self, model_id: str, device: str, **kwargs):
        """Initializes the model."""
        pass

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Transcribes an audio file and returns the text."""
        pass

    @abstractmethod
    def transcribe_with_timestamps(self, audio_path: str) -> Tuple[str, str]:
        """Transcribes an audio file and returns the text and timestamps content."""
        pass


class GigaAMWrapper(ASRWrapper):
    """Wrapper for GigaAM models (CTC and RNN-T)."""
    def __init__(self, model_id: str, device: str, **kwargs):
        logger.info(f"Initializing GigaAM model '{model_id}' on {device}")
        self.model_id = model_id
        self.device = device
        self.model_type = 'ctc' if 'ctc' in model_id else 'rnnt'
        self.use_lm = 'lm' in model_id
        self.model = gigaam.load_model(self.model_type, device=device)
        self.target_sr = 16_000
        self.decoder = None
        
        # Frame duration for timestamp calculation
        self.GIGA_AM_FRAME_SIZE_MS = 40
        self.sec_per_frame = self.GIGA_AM_FRAME_SIZE_MS / 1000.0

        if self.use_lm:
            if self.model_type != 'ctc':
                logger.warning(f"LM decoding is only supported for CTC models, but model is {self.model_type}. Ignoring LM.")
                self.use_lm = False
            elif 'lm_path' not in kwargs:
                logger.error("'lm_path' is required for GigaAM with LM but was not provided.")
                self.use_lm = False
            else:
                self._init_lm(kwargs['lm_path'])

    def _init_lm(self, lm_path: str, alpha: float = 0.5, beta: float = 1.0):
        logger.info(f"Building CTC decoder with LM: {lm_path}")
        try:
            vocab = self.model.decoding.tokenizer.vocab
            self.decoder = pyctcdecode.build_ctcdecoder(
                vocab,
                kenlm_model_path=lm_path,
                alpha=alpha,
                beta=beta,
            )
        except Exception as e:
            logger.error(f"Failed to build CTC decoder: {e}")
            self.decoder = None

    def transcribe(self, audio_path: str) -> str:
        if self.use_lm and self.decoder:
            text, _ = self.transcribe_with_timestamps(audio_path)
            return text
        
        text = self.model.transcribe(audio_path)
        return text
    
    def transcribe_with_timestamps(self, audio_path: str) -> Tuple[str, str]:
        if self.model_type != 'ctc':
            logger.info(f"Timestamp generation is only available for CTC models. Model '{self.model_id}' is RNN-T. Returning empty timestamps.")
            text = self.transcribe(audio_path)
            return text, ""

        if self.model_type == 'rnnt' or self.decoder is None:
            text = self.transcribe(audio_path)
            return text, ""

        audio = self._read_audio(audio_path)
        length = torch.tensor([audio.shape[-1]], device=self.device)

        encoded, _ = self.model.forward(audio.to(self.device), length)
        logits = self.model.head(encoded).squeeze(0).detach().cpu().numpy()
        
        beams = self.decoder.decode_beams(logits, beam_width=100)
        
        best_beam = beams[0]
        result_text = best_beam[0]
        word_timestamps_raw = best_beam[2]

        word_timestamps_formatted = self._to_simple_timestamps(word_timestamps_raw)

        return result_text, word_timestamps_formatted
    
    def _read_audio(self, path_to_file: str) -> torch.Tensor:
        audio, sr = torchaudio.load(path_to_file)
        if sr != self.target_sr:
            audio = torchaudio.functional.resample(audio, sr, self.target_sr)

        if audio.dim() > 1 and audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        return audio
        
    def _to_simple_timestamps(self, word_timestamps: List[Tuple[str, Tuple[int, int]]]) -> str:
        output_lines = []
        for word, (start_frame, end_frame) in word_timestamps:
            start_time = start_frame * self.sec_per_frame
            end_time = end_frame * self.sec_per_frame
            output_lines.append(f"{word} {start_time:.3f} {end_time:.3f}")
        return "\n".join(output_lines)


class ToneWrapper(ASRWrapper):
    """Wrapper for the Tone streaming CTC model."""
    def __init__(self, model_id: str, device: str, **kwargs):
        logger.info(f"Initializing Tone model '{model_id}' on {device}")
        self.tone_pipeline = StreamingCTCPipeline.from_hugging_face()
        self.device = 'cpu' # Tone is CPU-based
        self.target_sr = 8_000

    def transcribe(self, audio_path: str) -> str:
        audio = self._read_audio(audio_path)
        phrases = self.tone_pipeline.forward_offline(audio)
        result_text = ' '.join([phrase.text for phrase in phrases]).strip()
        return result_text
    
    def _read_audio(self, path_to_file: str) -> np.ndarray:
        audio = miniaudio.decode_file(str(path_to_file), nchannels=1, sample_rate=8000)
        assert audio.sample_rate == 8000
        assert audio.nchannels == 1
        return np.asarray(audio.samples, dtype=np.int16).astype(np.int32)
        
    def transcribe_with_timestamps(self, audio_path: str) -> Tuple[str, str]:
        audio_tensor = self._read_audio(audio_path)
        phrases: List[TextPhrase] = self.tone_pipeline.forward_offline(audio_tensor)
        full_text_parts = []
        timestamp_lines = []

        # TODO: add logic for word alignment
        for phrase in phrases:
            if phrase.text:
                full_text_parts.append(phrase.text)
                timestamp_lines.append(f"{phrase.text} {phrase.start_time:.3f} {phrase.end_time:.3f}")

        result_text = " ".join(full_text_parts).strip()
        result_timestamps = "\n".join(timestamp_lines)

        return result_text, result_timestamps


class VoskWrapper(ASRWrapper):
    """Wrapper for Vosk-style models using sherpa-onnx."""
    def __init__(self, model_id: str, device: str = 'cpu', **kwargs):
        logger.info(f"Initializing sherpa-onnx based ASR model from '{model_id}' on {device}")
        model_path = Path(model_id)
        encoder_path = str(model_path / "am-onnx" / "encoder.onnx")
        decoder_path = str(model_path / "am-onnx" / "decoder.onnx")
        joiner_path = str(model_path / "am-onnx" / "joiner.onnx")
        tokens_path = str(model_path / "lang" / "tokens.txt")
        self.target_sr = 16000

        if not all(Path(p).exists() for p in [encoder_path, decoder_path, joiner_path, tokens_path]):
            raise FileNotFoundError(f"One or more required model files not found in {model_path}")

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder_path,   
            decoder=decoder_path,
            joiner=joiner_path,
            tokens=tokens_path,
            num_threads=0,
            provider=device,
            sample_rate=self.target_sr,
            decoding_method="modified_beam_search"
        )
        logger.info(f"sherpa-onnx recognizer initialized successfully on {device}.")

    def transcribe(self, audio_path: str) -> str:
        audio = self._read_audio(audio_path)
        s = self.recognizer.create_stream()
        s.accept_waveform(self.target_sr, audio)
        self.recognizer.decode_stream(s)
        return s.result.text.strip()
    
    def _read_audio(self, path_to_file: str) -> Tuple[np.ndarray, int]:
        """Reads audio and returns a float32 numpy array for sherpa-onnx."""
        audio, sr = torchaudio.load(path_to_file)

        if sr != self.target_sr:
            audio = torchaudio.functional.resample(audio, sr, self.target_sr)

        if audio.dim() > 1 and audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        audio_samples = audio.numpy().squeeze().astype(np.float32)

        if np.max(np.abs(audio_samples)) > 1.0:
            audio_samples = audio_samples / 32768.0 
            
        return audio_samples
    
    def transcribe_with_timestamps(self, audio_path: str) -> Tuple[str, str]:
        """Transcribes an audio file and returns the text and word-level timestamps."""
        # TODO: add logic for word alignment
        audio = self._read_audio(audio_path)
        s = self.recognizer.create_stream()
        s.accept_waveform(self.target_sr, audio)
        self.recognizer.decode_stream(s)

        result = s.result
        full_text = result.text.strip()
        
        timestamp_lines = []
        for token, timestamp in zip(result.tokens, result.timestamps):
            timestamp_lines.append(f"{token} {timestamp:.3f} {timestamp:.3f}")
            
        result_timestamps = "\n".join(timestamp_lines)
        
        return full_text, result_timestamps

class ROVERWrapper:
    def __init__(self, podcasts_path: str):
        self.podcasts_path = Path(podcasts_path)
        self.tokenizer = lambda s: s.lower().split()
        self.detokenizer = lambda tokens: ' '.join(tokens)
        self.rover_aggregator = ROVER(self.tokenizer, self.detokenizer)

    def _collate_transcriptions(self) -> pd.DataFrame:
        records = []
        transcription_groups = defaultdict(list)

        txt_files = list(self.podcasts_path.rglob("*.txt"))
        for txt_path in txt_files:
            base_name = txt_path.stem.rsplit('_', 1)[0]
            audio_path = txt_path.with_name(f"{base_name}.mp3")
            if audio_path.exists():
                transcription_groups[audio_path].append(txt_path)
                
        for audio_path, txt_paths in transcription_groups.items():
            for txt_path in txt_paths:
                try:
                    model_name = txt_path.stem.replace(f"{audio_path.stem}_", "")
                    text = read_file_content(txt_path)
                    if text:
                        records.append({
                            'task': str(audio_path),
                            'worker': model_name,
                            'text': text
                        })
                except Exception as e:
                    print(f"Error reading the file {txt_path}: {e}")

        return pd.DataFrame(records)

    def aggregate_and_save(self):
        df = self._collate_transcriptions() 
        df['text'] = df['text'].str.lower()
        result = self.rover_aggregator.fit_predict(df)
        
        for task_path, agg_text in result.items():
            audio_path = Path(task_path)
            output_path = audio_path.with_name(f"{audio_path.stem}_rover.txt")
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(agg_text)


