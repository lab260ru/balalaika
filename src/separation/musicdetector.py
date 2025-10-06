import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoFeatureExtractor
import torchaudio
from safetensors import safe_open
from typing import List, Dict
import time

from huggingface_hub import hf_hub_download


torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


class WavLMForMusicDetection(nn.Module):
    """
    Music detection model based on WavLM.
    Uses attention pooling + classification head.
    Outputs probability that input audio contains music.
    Supports batched inference with automatic batching and preprocessing.
    EER - 2.5-3 %
    """
    def __init__(
        self,
        base_model_name: str = 'microsoft/wavlm-base-plus',
        batch_size: int = 32,
        device: str = 'cuda'
    ) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.wavlm = AutoModel.from_pretrained(base_model_name, config=self.config)
        self.processor = AutoFeatureExtractor.from_pretrained(base_model_name)

        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.target_sample_rate = self.processor.sampling_rate

        if not os.path.exists('./models/music_detection.safetensors'):
            self._load_from_hf()

        # Attention-based pooling head
        self.pool_attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # to device
        self.to(self.device)

    def _attention_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention-based pooling over time dimension.

        Args:
            hidden_states (torch.Tensor): [batch_size, seq_len, hidden_size]
            attention_mask (torch.Tensor): [batch_size, seq_len] — mask to ignore padding

        Returns:
            torch.Tensor: [batch_size, hidden_size] — context vector
        """
        
        attention_weights = self.pool_attention(hidden_states)  # [B, T, 1]
        # Mask out padded positions
        attention_weights = attention_weights + (
            (1.0 - attention_mask.unsqueeze(-1).to(attention_weights.dtype)) * -1e9
        )

        attention_weights = F.softmax(attention_weights, dim=1)  # [B, T, 1]

        # Weighted sum over time
        weighted_sum = torch.sum(hidden_states * attention_weights, dim=1)  # [B, D]
        return weighted_sum

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for inference.

        Args:
            input_values (torch.Tensor): [batch_size, audio_seq_len] — raw audio waveform
            attention_mask (torch.Tensor): [batch_size, audio_seq_len] — input mask (1 = real, 0 = pad)

        Returns:
            torch.Tensor: [batch_size, 1] — probability that audio contains music
        """
        assert isinstance(input_values, torch.Tensor), f"Expected torch.Tensor, got {type(input_values)}"
        assert isinstance(attention_mask, torch.Tensor), f"Expected torch.Tensor, got {type(attention_mask)}"

        outputs = self.wavlm(input_values.to(self.device), attention_mask=attention_mask.to(self.device))
        hidden_states = outputs.last_hidden_state  # [B, T', D]

        # Align attention mask with downsampled hidden states
        input_length = attention_mask.size(1)
        hidden_length = hidden_states.size(1)
        ratio = input_length / hidden_length
        indices = (torch.arange(hidden_length, device=attention_mask.device) * ratio).long()
        attention_mask = attention_mask[:, indices]  # [B, T']
        attention_mask = attention_mask.bool()

        pooled = self._attention_pool(hidden_states, attention_mask) 
        logits = self.classifier(pooled)  # [B, 1]

        probs = torch.sigmoid(logits)  # [B, 1] → probability of MUSIC
        return probs

    def _prepare_batches(self, audio_paths: List[str]) -> List[List[str]]:
        """
        Split list of audio paths into batches of size `self.batch_size`.

        Args:
            audio_paths (List[str]): List of paths to audio files.

        Returns:
            List[List[str]]: List of batches, each batch is a list of paths.
        """
        batches = []
        current_batch = []
        counter = 0

        while counter < len(audio_paths):
            if len(current_batch) == self.batch_size:
                batches.append(current_batch)
                current_batch = []
            current_batch.append(audio_paths[counter])
            counter += 1

        if current_batch:
            batches.append(current_batch)

        return batches

    def _preprocess_audio_batch(self, audio_paths: List[str]) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess a batch of audio files.

        Args:
            audio_paths (List[str]): List of file paths.

        Returns:
            Dict with keys:
                "input_values": tensor [B, T]
                "attention_mask": tensor [B, T]
        """
        waveforms = []

        for audio_path in audio_paths:
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample if needed
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveforms.append(waveform.squeeze())

        # Extract features
        inputs = self.processor(
            [w.numpy() for w in waveforms],
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=True,
            truncation=False
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    def predict_proba(self, audio_paths: List[str]) -> torch.Tensor:
        """
        Predict music probability for a list of audio files.

        Args:
            audio_paths (List[str]): List of audio file paths.

        Returns:
            torch.Tensor: [N] — probabilities for each audio file.
        """

        all_probs = []

        batches = self._prepare_batches(audio_paths)

        for batch in batches:
            inputs = self._preprocess_audio_batch(batch)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                probs = self.forward(**inputs).squeeze(-1)  # [B]
            all_probs.append(probs)

        return torch.cat(all_probs, dim=0)
    
    def _load_from_hf(self):
        return hf_hub_download(
            repo_id="MTUCI/MusicDetection",
            filename="music_detection.safetensors",
            local_dir="./models",         
            local_dir_use_symlinks=False  
        )
        