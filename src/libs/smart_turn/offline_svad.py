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

    def _prepare_audio(self, audio_array: np.ndarray, source_rate: int) -> np.ndarray:
        """Resample (if needed) and truncate/pad one segment to exactly 8 s."""
        if source_rate != self.sample_rate:
            audio_tensor = torch.from_numpy(
                np.asarray(audio_array, dtype=np.float32)
            )
            audio_array = torchaudio.functional.resample(
                audio_tensor,
                source_rate,
                self.sample_rate,
            ).numpy()
        return self._truncate_audio(
            audio_array,
            n_seconds=8,
            sample_rate=self.sample_rate,
        )

    def _extract_features(self, prepared) -> torch.Tensor:
        """Run the Whisper feature extractor on one segment or a list.

        Passing a list of fixed-length 8 s windows produces a stacked
        ``(N, 80, 800)`` tensor whose rows match per-segment extraction to
        within ~1 float32 ULP (do_normalize is per-row; the batched STFT
        reductions cause the tiny epsilon — measured in tests). The
        single-segment path and the batch path share this one implementation.
        """
        inputs = self.feature_extractor(
            prepared,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=8 * self.sample_rate,
            truncation=True,
            do_normalize=True,
            device=self.device,
        )
        return inputs.input_features.contiguous().to(
            device=self.torch_device,
            dtype=torch.float32,
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
        prepared = self._prepare_audio(audio_array, source_rate)
        input_features = self._extract_features(prepared)
        probability = self._run_session(input_features)
        return self._format_result(probability)

    def _format_result(self, probability: float) -> dict:
        prediction = 1 if probability > self.smart_vad_threshold else 0
        return {
            "prediction": prediction,
            "probability": round(probability, 4),
        }

    def predict_endpoint_batch(
        self,
        audio_arrays: list,
        sample_rate: int | None = None,
    ) -> list:
        """Classify several segments in one feature-extraction + ONNX call.

        Each segment is truncated/padded to the same fixed 8 s window, stacked
        into one ``(N, 80, 800)`` feature tensor and run through a single
        ``session.run`` / IOBinding. Returns one result dict per input in order,
        each identical in shape to :meth:`predict_endpoint`. NOT bit-exact vs the
        per-segment path: the batched Whisper feature extraction differs from
        batch-1 by up to ~1 float32 ULP (verified in tests) and batched cuBLAS
        kernels add more epsilon near the 0.4 threshold, so callers gate this
        behind a config knob (default keeps batch size 1, the exact old path).
        """
        source_rate = sample_rate or self.sample_rate
        if not audio_arrays:
            return []
        prepared = [self._prepare_audio(a, source_rate) for a in audio_arrays]
        input_features = self._extract_features(prepared)
        probabilities = self._run_session_batch(input_features)
        return [self._format_result(float(p)) for p in probabilities]

    def _run_session_batch(self, input_features: torch.Tensor) -> np.ndarray:
        """Run the session over an ``(N, 80, 800)`` batch, returning ``(N,)``.

        The single-segment path is the N==1 case of this, so both share the
        same IOBinding / numpy-fallback logic. Output rows keep input order.
        """
        n = int(input_features.shape[0])
        if self._use_cuda_dlpack:
            try:
                output = torch.empty(
                    (n, 1),
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
                return output[:, 0].detach().cpu().numpy()
            except Exception as exc:
                logger.warning(f"Smart VAD DLPack IOBinding failed, falling back to session.run: {exc}")
                self._use_cuda_dlpack = False

        input_np = input_features.cpu().numpy()
        outputs = self.session.run(None, {self.input_name: input_np})
        return np.asarray(outputs[0]).reshape(-1)

    def _run_session(self, input_features: torch.Tensor) -> float:
        return float(self._run_session_batch(input_features)[0])

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

