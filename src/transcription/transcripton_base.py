import os
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import gigaam
import kaldifeat
import numpy as np
import pandas as pd
import pyctcdecode
import sentencepiece as spm
import torch
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


class GigaAMWrapper(ASRWrapper):
    GIGA_AM_FRAME_SIZE_MS = 40  # Duration of a single frame in ms for CTC models

    def __init__(self, model_id: str, device: str, **kwargs):
        super().__init__(model_id, device)
        logger.info(f"Initializing GigaAM model '{self.model_id}' on device {self.device}")

        self.model_type = 'ctc' if 'ctc' in self.model_id else 'rnnt'
        self.use_lm = 'lm' in self.model_id
        
        self.model = gigaam.load_model(
            model_name=self.model_type,
            device=self.device,
            fp16_encoder=True,
            use_flash=False
        )
        self.model.eval()

        self.target_sr = 16_000
        self.decoder = None
        self.sec_per_frame = self.GIGA_AM_FRAME_SIZE_MS / 1000.0
        
        if self.use_lm:
            if 'lm_path' in kwargs and not os.path.exists(kwargs['lm_path']):
                try:
                    self._downlaod_lm()
                except Exception as e:
                    logger.warning(f"Could not download LM: {e}")

            if self.model_type != 'ctc':
                logger.warning(f"LM decoding is only supported for CTC models. LM ignored.")
                self.use_lm = False
            elif 'lm_path' not in kwargs:
                logger.error("'lm_path' required for LM usage. LM ignored.")
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
            self.use_lm = False

    @torch.inference_mode()
    def transcribe_tensors(self, batch_wav: torch.Tensor, batch_lengths: torch.Tensor) -> List[str]:
        if self.use_lm and self.decoder:
            texts, _ = self.transcribe_tensors_with_timestamps(batch_wav, batch_lengths)
            return texts

        batch_wav = batch_wav.to(self.device, non_blocking=True)
        batch_lengths = batch_lengths.to(self.device, non_blocking=True)

        encoded, encoded_len = self.model.forward(batch_wav, batch_lengths)
        transcriptions = self.model.decoding.decode(self.model.head, encoded, encoded_len)
        
        return [t if t else "" for t in transcriptions]

    @torch.inference_mode()
    def transcribe_tensors_with_timestamps(self, batch_wav: torch.Tensor, batch_lengths: torch.Tensor) -> Tuple[List[str], List[str]]:
        batch_wav = batch_wav.to(self.device, non_blocking=True)
        batch_lengths = batch_lengths.to(self.device, non_blocking=True)

        result_texts = []
        result_timestamps = []

        encoded, encoded_len = self.model.forward(batch_wav, batch_lengths)
        
        if self.model_type == 'ctc' and self.decoder:
            logits_batch = self.model.head(encoded)
            logits_batch = logits_batch.detach().cpu().numpy()
            encoded_len_cpu = encoded_len.cpu().tolist()

            for i, valid_len in enumerate(encoded_len_cpu):
                sample_logits = logits_batch[i, :valid_len, :]
                try:
                    beams = self.decoder.decode_beams(sample_logits, beam_width=100)
                    if beams:
                        best_beam = beams[0]
                        text, _, word_timestamps_raw, _, _ = best_beam
                        formatted_timestamps = self._to_simple_timestamps(word_timestamps_raw)
                        result_texts.append(text)
                        result_timestamps.append(formatted_timestamps)
                    else:
                        result_texts.append("")
                        result_timestamps.append("")
                except Exception as e:
                    logger.error(f"LM decoding error: {e}")
                    result_texts.append("")
                    result_timestamps.append("")
        else:
            transcriptions = self.model.decoding.decode(self.model.head, encoded, encoded_len)
            result_texts = [t if t else "" for t in transcriptions]
            result_timestamps = [""] * len(result_texts)

        return result_texts, result_timestamps

    def _to_simple_timestamps(self, word_timestamps: List[Tuple[str, Tuple[int, int]]]) -> str:
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
            self.device_id = -1
        
        self.tone_pipeline = StreamingCTCPipeline.from_hugging_face(device_id=self.device_id)
        self.target_sr = 8_000

    def transcribe_audio_data(self, audios: List[np.ndarray]) -> List[str]:
        result_texts = []
        
        for audio_np in audios:
            try:
                phrases = self.tone_pipeline.forward_offline(audio_np)
                result_text = ' '.join([phrase.text for phrase in phrases]).strip()
                result_texts.append(result_text)
            except Exception as e:
                logger.error(f"Tone inference error: {e}")
                result_texts.append("")
                
        return result_texts

    def transcribe_audio_data_with_timestamps(self, audios: List[np.ndarray]) -> Tuple[List[str], List[str]]:
        result_texts = []
        result_timestamps = []

        for audio_np in audios:
            try:
                phrases: List[TextPhrase] = self.tone_pipeline.forward_offline(audio_np)
                
                full_text_parts = []
                timestamp_lines = []

                for phrase in phrases:
                    if phrase.text:
                        full_text_parts.append(phrase.text)
                        timestamp_lines.append(f"{phrase.text} {phrase.start_time:.3f} {phrase.end_time:.3f}")

                result_text = " ".join(full_text_parts).strip()
                result_timestamps_str = "\n".join(timestamp_lines)

                result_texts.append(result_text)
                result_timestamps.append(result_timestamps_str)
            
            except Exception as e:
                logger.error(f"Tone inference error (timestamps): {e}")
                result_texts.append("")
                result_timestamps.append("")

        return result_texts, result_timestamps


class VOSKCUDAWrapper(ASRWrapper):
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
        self.beam_size = kwargs.get('beam_size', 10) 
        self.lm_scale = kwargs.get('lm_scale', 0.5)
        
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
        self.ngram_lm_scale = kwargs.get('ngram_lm_scale', 1.0)
    
    @torch.inference_mode()
    def transcribe_batch_data(self, waveforms: List[torch.Tensor]) -> List[str]:
        if not waveforms:
            return []

        waves_gpu = [w.to(self.device, non_blocking=True) for w in waveforms]
        features = self.fbank(waves_gpu)
        feature_lengths = torch.tensor([f.size(0) for f in features], device=self.device)
        features_padded = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))
        
        encoder_out, encoder_out_lens = self.model.encoder(
            features=features_padded,
            feature_lengths=feature_lengths,
        )

        hyps = modified_beam_search_LODR(
            model=self.model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=self.beam_size,
            LODR_lm=self.ngram_lm,
            LODR_lm_scale=self.ngram_lm_scale,
            LM=self.LM,
        )

        result_texts = [self.sp.decode(hyp) for hyp in hyps]
        return result_texts 

        
    def _load_from_hf(self):
        return snapshot_download(
            repo_id="alphacep/vosk-model-ru",
            local_dir="./models/vosk-model-ru",         
            local_dir_use_symlinks=False  
        )

class ROVERWrapper:
    def __init__(self, podcasts_path: str, model_names: List[str]):
        self.podcasts_path = Path(podcasts_path)
        self.model_names = model_names
        self.tokenizer = lambda s: s.lower().split()
        self.detokenizer = lambda tokens: ' '.join(tokens)
        self.rover_aggregator = ROVER(self.tokenizer, self.detokenizer)

    def aggregate_and_save(self):
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
                    if not text:
                        continue
                    
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