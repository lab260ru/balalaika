import os
from loguru import logger
import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor
from huggingface_hub import hf_hub_download


class SmartVAD:
    """EOS (End of Speech) classifier using Smart Turn model.
    Classifies whether a speech segment represents a complete utterance.
    """

    def __init__(
            self,
            smart_vad_threshold: float = 0.4,
            device: str = 'cuda:0',
            resample_rate: int = 16_000,
            smart_vad_model: str = "pipecat-ai/smart-turn-v3"
            ):
        self.smart_vad_threshold = smart_vad_threshold
        self.sample_rate = resample_rate
        self.smart_vad_model = smart_vad_model
        self.device = device
        self.device_id = int(device.split(':')[1]) if ':' in device else 0

        if not os.path.exists(self.smart_vad_model):
            self._load_from_hf()

        self._init_smart_vad()

    def _init_smart_vad(self):
        logger.info('Initializing Smart VAD (EOS classifier)...')
        self.feature_extractor = WhisperFeatureExtractor(chunk_length=8)

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self.smart_vad_model, sess_options=so,
            providers=[
                (
                    "CUDAExecutionProvider",
                    {"device_id": self.device_id}
                ),
                "CPUExecutionProvider",
            ]
        )
        logger.info('Smart VAD (EOS classifier) initialized.')

    def _load_from_hf(self):
        return hf_hub_download(
            repo_id="pipecat-ai/smart-turn-v3",
            filename="smart-turn-v3.2-gpu.onnx",
            local_dir="./models",
            local_dir_use_symlinks=False
        )

    def predict_endpoint(self, audio_array: np.ndarray) -> dict:
        """
        Predict whether an audio segment is complete (turn ended) or incomplete.

        Args:
            audio_array: Numpy array containing audio samples at 16kHz

        Returns:
            Dictionary with 'prediction' (1=complete, 0=incomplete) and 'probability'
        """
        audio_array = self._truncate_audio(audio_array, n_seconds=8)

        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=8 * 16000,
            truncation=True,
            do_normalize=True
        )

        input_features = inputs.input_features.squeeze(0).numpy().astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)

        outputs = self.session.run(None, {"input_features": input_features})
        probability = outputs[0][0].item()
        prediction = 1 if probability > self.smart_vad_threshold else 0

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
        }

    @staticmethod
    def _truncate_audio(audio_array: np.ndarray, n_seconds: int = 8, sample_rate: int = 16000) -> np.ndarray:
        """Truncate audio to last n seconds or pad with zeros to meet n seconds."""
        max_samples = n_seconds * sample_rate
        if len(audio_array) > max_samples:
            return audio_array[-max_samples:]
        elif len(audio_array) < max_samples:
            padding = max_samples - len(audio_array)
            return np.pad(audio_array, (padding, 0), mode='constant', constant_values=0)
        return audio_array

