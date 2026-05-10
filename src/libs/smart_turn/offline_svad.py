import os
from loguru import logger
import numpy as np
import onnxruntime as ort
import torch
import torchaudio
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
        self.torch_device = torch.device(
            device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )

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
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self._use_cuda_dlpack = (
            self.torch_device.type == "cuda"
            and "CUDAExecutionProvider" in self.session.get_providers()
            and hasattr(ort.OrtValue, "from_dlpack")
        )
        logger.info('Smart VAD (EOS classifier) initialized.')

    def _load_from_hf(self):
        return hf_hub_download(
            repo_id="pipecat-ai/smart-turn-v3",
            filename="smart-turn-v3.2-gpu.onnx",
            local_dir="./models",
            local_dir_use_symlinks=False
        )

    def predict_endpoint(self, audio_array: np.ndarray, sample_rate: int | None = None) -> dict:
        """
        Predict whether an audio segment is complete (turn ended) or incomplete.

        Args:
            audio_array: Numpy array containing audio samples.
            sample_rate: Sampling rate of audio_array. Defaults to SmartVAD's
                expected sample rate for backwards-compatible callers.

        Returns:
            Dictionary with 'prediction' (1=complete, 0=incomplete) and 'probability'
        """
        source_rate = sample_rate or self.sample_rate
        if source_rate != self.sample_rate:
            audio_tensor = torch.from_numpy(
                np.asarray(audio_array, dtype=np.float32)
            )
            audio_array = torchaudio.functional.resample(
                audio_tensor,
                source_rate,
                self.sample_rate,
            ).numpy()
        audio_array = self._truncate_audio(
            audio_array,
            n_seconds=8,
            sample_rate=self.sample_rate,
        )

        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=8 * self.sample_rate,
            truncation=True,
            do_normalize=True,
            device=self.device,
        )

        input_features = inputs.input_features.contiguous().to(
            device=self.torch_device,
            dtype=torch.float32,
        )

        probability = self._run_session(input_features)
        prediction = 1 if probability > self.smart_vad_threshold else 0

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
        }

    def _run_session(self, input_features: torch.Tensor) -> float:
        if self._use_cuda_dlpack:
            try:
                output = torch.empty(
                    (input_features.shape[0], 1),
                    device=self.torch_device,
                    dtype=torch.float32,
                )
                io_binding = self.session.io_binding()
                io_binding.bind_ortvalue_input(
                    self.input_name,
                    ort.OrtValue.from_dlpack(input_features),
                )
                io_binding.bind_ortvalue_output(
                    self.output_name,
                    ort.OrtValue.from_dlpack(output),
                )
                self.session.run_with_iobinding(io_binding)
                torch.cuda.synchronize(self.torch_device)
                return float(output[0, 0].item())
            except Exception as exc:
                logger.warning(f"Smart VAD DLPack IOBinding failed, falling back to session.run: {exc}")
                self._use_cuda_dlpack = False

        input_np = input_features.cpu().numpy()
        outputs = self.session.run(None, {self.input_name: input_np})
        return float(outputs[0][0].item())

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

