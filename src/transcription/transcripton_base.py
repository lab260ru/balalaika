import os
import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import time
import gigaam
import kaldifeat
import miniaudio
import numpy as np
import pandas as pd
import pyctcdecode
import sentencepiece as spm
import sherpa_onnx
import torch
import torchaudio
from crowdkit.aggregation import ROVER
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from tone import StreamingCTCPipeline, TextPhrase
from tqdm import tqdm
from huggingface_hub import snapshot_download, hf_hub_download 

from src.utils import read_file_content, get_audio_paths
from src.utils_asr import (AttributeDict, LmScorer, NgramLm,
                           modified_beam_search_LODR)


class ASRWrapper(ABC):
    """Abstract Base Class for ASR model wrappers."""

    @abstractmethod
    def __init__(self, model_id: str, device: str, **kwargs):
        """
        Initializes the ASR model.

        Args:
            model_id (str): The identifier for the model.
            device (str): The device for computations (e.g., 'cuda:0').
            **kwargs: Additional arguments for model configuration.
        """
        self.model_id = model_id
        self.device = device

    @abstractmethod
    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """
        Transcribes a batch of audio files and returns the texts.

        Args:
            audio_paths (List[str]): A list of paths to the audio files.

        Returns:
            List[str]: A list of transcribed texts.
        """
        pass

    @abstractmethod
    def transcribe_batch_with_timestamps(self, audio_paths: List[str]) -> Tuple[List[str], List[str]]:
        """
        Transcribes a batch of audio files, returning texts and timestamps.

        Args:
            audio_paths (List[str]): A list of paths to the audio files.

        Returns:
            Tuple[List[str], List[str]]: A tuple of two lists:
                                         - A list of transcribed texts.
                                         - A list of strings with timestamps for each audio.
        """
        pass


class GigaAMWrapper(ASRWrapper):
    """Wrapper for GigaAM models (CTC and RNN-T)."""

    GIGA_AM_FRAME_SIZE_MS = 40  # Duration of a single frame in ms for CTC models

    def __init__(self, model_id: str, device: str, **kwargs):
        super().__init__(model_id, device)
        logger.info(f"Initializing GigaAM model '{self.model_id}' on device {self.device}")

        self.model_type = 'ctc' if 'ctc' in self.model_id else 'rnnt'
        self.use_lm = 'lm' in self.model_id
        self.model = gigaam.load_model(self.model_type, device=self.device)
        self.target_sr = 16_000
        self.decoder = None
        self.sec_per_frame = self.GIGA_AM_FRAME_SIZE_MS / 1000.0
        
        if self.use_lm and not os.path.exists(kwargs['lm_path']):
            self._downlaod_lm()

        if self.use_lm:
            if self.model_type != 'ctc':
                logger.warning(f"LM decoding is only supported for CTC models, but got {self.model_type}. LM will be ignored.")
                self.use_lm = False
            elif 'lm_path' not in kwargs:
                logger.error("'lm_path' is required for GigaAM with LM but was not provided. LM will not be used.")
                self.use_lm = False
            else:
                self._init_lm(kwargs['lm_path'])

    def _init_lm(self, lm_path: str, alpha: float = 0.5, beta: float = 1.0):
        """Initializes the CTC decoder with a language model."""
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
            self.use_lm = False

    @torch.inference_mode()
    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribes a batch of audio files."""
        # If an LM is used, the transcription logic is in the method with timestamps
        if self.use_lm and self.decoder:
            texts, _ = self.transcribe_batch_with_timestamps(audio_paths)
            return texts

        texts = [self.model.transcribe(audio_path) for audio_path in audio_paths]
        return texts

    @torch.inference_mode()
    def transcribe_batch_with_timestamps(self, audio_paths: List[str]) -> Tuple[List[str], List[str]]:
        """Transcribes audio, returning text and timestamps (only for CTC with LM)."""
        if self.model_type != 'ctc' or self.decoder is None:
            logger.warning("Timestamps are only available for CTC models with a language model.")
            texts = self.transcribe_batch(audio_paths)
            # Return empty strings for timestamps
            return texts, [''] * len(texts)

        result_texts = []
        result_timestamps = []

        for audio_path in audio_paths:
            audio = self._read_audio(audio_path).to(self.device)
            length = torch.tensor([audio.shape[-1]], device=self.device)

            encoded, _ = self.model.forward(audio, length)
            logits = self.model.head(encoded).squeeze(0).detach().cpu().numpy()

            beams = self.decoder.decode_beams(logits, beam_width=100)
            best_beam = beams[0]
            text, _, word_timestamps_raw, _, _ = best_beam
            formatted_timestamps = self._to_simple_timestamps(word_timestamps_raw)

            result_texts.append(text)
            result_timestamps.append(formatted_timestamps)

        return result_texts, result_timestamps

    def _read_audio(self, path_to_file: str) -> torch.Tensor:
        """Reads an audio file, resamples it, and converts it to mono."""
        audio, sr = torchaudio.load(path_to_file)
        if sr != self.target_sr:
            audio = torchaudio.functional.resample(audio, sr, self.target_sr)
        if audio.dim() > 1 and audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        return audio

    def _to_simple_timestamps(self, word_timestamps: List[Tuple[str, Tuple[int, int]]]) -> str:
        """Formats timestamps into a string."""
        output_lines = []
        for word, (start_frame, end_frame) in word_timestamps:
            start_time = start_frame * self.sec_per_frame
            end_time = end_frame * self.sec_per_frame
            output_lines.append(f"{word} {start_time:.3f} {end_time:.3f}")
        return "\n".join(output_lines)
    
    def _downlaod_lm(self):
        return hf_hub_download(
            repo_id="MTUCI/lm",
            filename="kenlm.bin",
            local_dir="./models",
            repo_type='dataset',      
            local_dir_use_symlinks=False  
        )





class ToneWrapper(ASRWrapper):
    """Wrapper for the Tone streaming CTC model."""
    def __init__(self, model_id: str, device: str, **kwargs):
        super().__init__(model_id, device)
        logger.info(f"Initializing Tone model '{self.model_id}' on device {self.device}")
        
        if 'cuda' in self.device:
            try:
                self.device_id = int(self.device.split(':')[-1])
            except (ValueError, IndexError):
                logger.error(f"Invalid CUDA device format: {self.device}. Using ID 0.")
                self.device_id = 0
        else:
            self.device_id = -1 # ID for CPU
        
        self.tone_pipeline = StreamingCTCPipeline.from_hugging_face(device_id=self.device_id)
        self.target_sr = 8_000

    @torch.inference_mode()
    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribes a batch of audio files."""
        result_texts = []
        for audio_path in audio_paths:
            audio = self._read_audio(audio_path)
            phrases = self.tone_pipeline.forward_offline(audio)
            result_text = ' '.join([phrase.text for phrase in phrases]).strip()
            result_texts.append(result_text)
        return result_texts

    def _read_audio(self, path_to_file: str) -> np.ndarray:
        audio = miniaudio.decode_file(str(path_to_file), nchannels=1, sample_rate=8000)
        assert audio.sample_rate == 8000
        assert audio.nchannels == 1
        return np.asarray(audio.samples, dtype=np.int16).astype(np.int32) 

    @torch.inference_mode()
    def transcribe_batch_with_timestamps(self, audio_paths: List[str]) -> Tuple[List[str], List[str]]:
        """Transcribes audio, returning text and phrase-level timestamps."""
        result_texts = []
        results_timestamps = []
        for audio_path in audio_paths:
            audio_tensor = self._read_audio(audio_path)
            phrases: List[TextPhrase] = self.tone_pipeline.forward_offline(audio_tensor)
            
            full_text_parts = []
            timestamp_lines = []

            for phrase in phrases:
                if phrase.text:
                    full_text_parts.append(phrase.text)
                    # TODO: add logic for word-level alignment if needed
                    timestamp_lines.append(f"{phrase.text} {phrase.start_time:.3f} {phrase.end_time:.3f}")

            result_text = " ".join(full_text_parts).strip()
            result_timestamps = "\n".join(timestamp_lines)

            result_texts.append(result_text)
            results_timestamps.append(result_timestamps)

        return result_texts, results_timestamps


class VOSKCUDAWrapper(ASRWrapper):
    """Wrapper for a custom VOSK model adapted for CUDA."""
    def __init__(self, model_id: str, device: str, **kwargs):
        super().__init__(model_id, device)
        logger.info(f"Initializing VOSK CUDA model '{self.model_id}' on device {self.device}")
        vosk_path = Path(kwargs['vosk_path'])
        
        if not os.path.exists(str(vosk_path)):
            self._load_from_hf()

        self.nn_model_filename = str(vosk_path / 'am' / 'jit_script.pt')
        self.ngram_path = str(vosk_path / 'lm' / '2gram.fst.txt')
        self.lm_exp_dir_vosk = str(vosk_path / 'lm')
        self.bpe_model = str(vosk_path / 'lang' / 'bpe.model')
        
        self.target_sr = 16_000
        self.beam_size = kwargs['beam_size']
        self.lm_scale = kwargs['lm_scale']
        self.model = torch.jit.load(self.nn_model_filename).to(self.device).eval()

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.bpe_model)

        opts = kaldifeat.FbankOptions()
        opts.device = self.device
        opts.frame_opts.dither = 3e-5
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = self.target_sr
        opts.mel_opts.num_bins = 80
        opts.mel_opts.high_freq = -400
        self.fbank = kaldifeat.Fbank(opts)

        params = AttributeDict({
            "lm_vocab_size": 500,
            "rnn_lm_embedding_dim": 2048,
            "rnn_lm_hidden_dim": 2048,
            "rnn_lm_num_layers": 3,
            "rnn_lm_tie_weights": True,
            "lm_epoch": 99,
            "lm_exp_dir": self.lm_exp_dir_vosk,
            "lm_avg": 1,
        })

        self.LM = LmScorer(
            lm_type="rnn",
            params=params,
            device=self.device,
            lm_scale=self.lm_scale,
        ).to(self.device).eval()

        self.ngram_lm = NgramLm(
            self.ngram_path,
            backoff_id=500,
            is_binary=False,
        )
        self.ngram_lm_scale = kwargs['ngram_lm_scale']
    
    @torch.inference_mode()
    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribes a batch of audio files."""
        waves = [self._read_audio(path).to(self.device) for path in audio_paths]
        features = self.fbank(waves)
        feature_lengths = torch.tensor([f.size(0) for f in features], device=self.device)
        features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))
        encoder_out, encoder_out_lens = self.model.encoder(
            features=features,
            feature_lengths=feature_lengths,
        )
        # start_time = time.time()
        hyps = modified_beam_search_LODR(
            model=self.model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=self.beam_size,
            LODR_lm=self.ngram_lm,
            LODR_lm_scale=self.ngram_lm_scale,
            LM=self.LM,
        )
        # print(time.time() - start_time, 'hyps')
        result_texts = [self.sp.decode(hyp) for hyp in hyps]
        return result_texts 

    @torch.inference_mode()
    def transcribe_batch_with_timestamps(self, audio_paths: List[str]) -> Tuple[List[str], List[str]]:
        """This model does not support timestamp generation."""
        texts = self.transcribe_batch(audio_paths)
        return texts, [''] * len(audio_paths)

    def _read_audio(self, path_to_file: str) -> torch.Tensor:
        """Reads an audio file, resamples it, and converts it to mono."""
        audio, sr = torchaudio.load(path_to_file)
        if sr != self.target_sr:
            audio = torchaudio.functional.resample(audio, sr, self.target_sr)
        if audio.dim() > 1 and audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        return audio.squeeze(0)
    
    def _load_from_hf(self):
        return snapshot_download(
            repo_id="alphacep/vosk-model-ru",
            local_dir="./models/vosk-model-ru",         
            local_dir_use_symlinks=False  
        )

class VoskSherpaOnnxWrapper(ASRWrapper):
    """Wrapper for Vosk-style models using sherpa-onnx."""
    def __init__(self, model_id: str, device: str = 'cpu', **kwargs):
        logger.info(f"Initializing sherpa-onnx based ASR model from '{model_id}' on {device}")
        device = 'cpu'
        
        if not os.path.exists(model_id):
            self._load_from_hf()

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
            num_threads=4,
            provider=device,
            sample_rate=self.target_sr,
            decoding_method="modified_beam_search"
        )
        logger.info(f"sherpa-onnx recognizer initialized successfully on {device}.")

    @torch.inference_mode()
    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        result_texts = []
        for audio_path in audio_paths:
            audio = self._read_audio(audio_path)
            s = self.recognizer.create_stream()
            s.accept_waveform(self.target_sr, audio)
            self.recognizer.decode_stream(s)
            result_texts.append(s.result.text.strip())

        return result_texts       
    
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
    
    @torch.inference_mode()
    def transcribe_batch_with_timestamps(self, audio_paths: List[str]) -> Tuple[List[str], List[str]]:
        """Transcribes an audio file and returns the text and word-level timestamps."""
        result_texts = []
        results_timestamps = []
        for audio_path in audio_paths:
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

            result_texts.append(full_text)
            results_timestamps.append(result_timestamps)
        
        return result_texts, results_timestamps
    
    def _load_from_hf(self):
        return snapshot_download(
            repo_id="alphacep/vosk-model-ru",
            local_dir="./models/vosk-model-ru",         
            local_dir_use_symlinks=False  
        )


class ROVERWrapper:
    """Aggregates transcription results from multiple models using ROVER."""
    
    def __init__(self, podcasts_path: str, model_names: List[str]):
        """
        Args:
            podcasts_path (str): Path to the directory with transcription files.
            model_names (List[str]): A list of model suffixes to include in the aggregation.
        """
        self.podcasts_path = Path(podcasts_path)
        self.model_names = model_names
        # self.model_names = ['giga_ctc_lm', 'vosk', 'tone', 'giga_ctc', 'giga_rnnt'] 
        self.tokenizer = lambda s: s.lower().split()
        self.detokenizer = lambda tokens: ' '.join(tokens)
        self.rover_aggregator = ROVER(self.tokenizer, self.detokenizer)

    def aggregate_and_save(self):
        """
        Performs transcription aggregation and saves the results.
        
        This method now first finds all audio files and then collects the
        corresponding transcription files for each model.
        """
        logger.info("Starting transcription aggregation based on audio files.")
        
        all_audio_paths = get_audio_paths(str(self.podcasts_path))
        
        if not all_audio_paths:
            logger.warning("Audio files not found. Aggregation finished.")
            return

        records = []
        excluded_patterns = ['_rover', '_phonemes', '_accent']

        for audio_path in tqdm(all_audio_paths, desc="Aggregating transcriptions"):
            if any(pattern in audio_path.stem for pattern in excluded_patterns):
                continue
            
            for model_name in self.model_names:
                transcript_path = audio_path.with_name(f"{audio_path.stem}_{model_name}.txt")

                if not transcript_path.exists():
                    continue
                
                try:
                    text = read_file_content(transcript_path)
                    if text:
                        records.append({
                            'task': str(audio_path),
                            'worker': model_name,
                            'text': text
                        })
                except Exception as e:
                    logger.error(f"Error reading file {transcript_path}: {e}")
        
        df = pd.DataFrame(records)
        if df.empty:
            logger.warning("No transcriptions found for aggregation. Check file paths and names.")
            return

        df['text'] = df['text'].str.lower()
        logger.info(f"Running ROVER on {len(df['task'].unique())} unique audio files...")
        result = self.rover_aggregator.fit_predict(df)
        
        logger.info("Saving aggregated results...")
        for task_path, agg_text in result.items():
            audio_path = Path(task_path)
            output_path = audio_path.with_name(f"{audio_path.stem}_rover.txt")
            
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(agg_text)
            except IOError as e:
                logger.error(f"Failed to write result to {output_path}: {e}")
        
        logger.info("Aggregation complete.")