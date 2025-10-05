import torch
import os
from loguru import logger
import numpy as np
import onnxruntime as ort
import time 
from transformers import WhisperFeatureExtractor

class OfflineVAD:
    def __init__(
            self,
            silero_vad_threshold: float = 0.4,
            smart_vad_threshold: float = 0.4,
            device: str = 'cuda:1',
            resample_rate: int = 16_000,
            max_speech_duration_s: float = 8,
            smart_vad_path: str = "pipecat-ai/smart-turn-v3"
            ):
            
        self.silero_vad_threshold = silero_vad_threshold
        self.smart_vad_threshold = smart_vad_threshold
        self.max_speech_duration_s= max_speech_duration_s
        self.sample_rate = resample_rate
        self.smart_vad_path = smart_vad_path
        self.device = device
        self.device_id = device.split(':')[1] if ':' in self.device else 0 

        # Initialize VAD models
        self._init_silero_vad()
        self._init_smart_vad()

    def _init_silero_vad(self):
        logger.info('Initializing Silero VAD model...')
        torch.hub.set_dir("./.torch_hub")
        silero_vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
            trust_repo=True
        )
        logger.info('Silero VAD model successfully initialized.')
        self.silero_vad_model = silero_vad_model

        (self.get_speech_timestamps,
            _,
            self.read_audio,
            _, _) = utils
            

    def _init_smart_vad(self):
        logger.info('Initializing Smart VAD model...')
        self.feature_extractor = WhisperFeatureExtractor(chunk_length=8)

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self.smart_vad_path, sess_options=so,     
            providers=[
                (
                    "CUDAExecutionProvider",
                    {"device_id": self.device_id}
                ),
                "CPUExecutionProvider",  
            ]
        )
        logger.info('Smart VAD model successfully initialized.')

    def _load_and_validate_audio(self, audio_path: str) -> torch.Tensor:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File not found: {audio_path}")

        wav_tensor = self.read_audio(audio_path, sampling_rate=self.sample_rate)
        return wav_tensor
    
    def predict_endpoint(self, audio_array):
        """
        Predict whether an audio segment is complete (turn ended) or incomplete.

        Args:
            audio_array: Numpy array containing audio samples at 16kHz

        Returns:
            Dictionary containing prediction results:
            - prediction: 1 for complete, 0 for incomplete
            - probability: Probability of completion (sigmoid output)
        """

        # Truncate to 8 seconds (keeping the end) or pad to 8 seconds
        audio_array = self.truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)

        # Process audio using Whisper's feature extractor
        inputs = self.feature_extractor( 
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=8 * 16000,
            truncation=True,
            do_normalize=True
        )

        # Convert to numpy and ensure correct shape for ONNX
        input_features = inputs.input_features.squeeze(0).numpy().astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)  # Add batch dimension

        # Run ONNX inference
        outputs = self.session.run(None, {"input_features": input_features})

        # Extract probability (ONNX model returns sigmoid probabilities)
        probability = outputs[0][0].item()

        # Make prediction (1 for Complete, 0 for Incomplete)
        prediction = 1 if probability > self.smart_vad_threshold else 0

        return {
            "prediction": prediction,
            "probability": probability,
        }
        
    def truncate_audio_to_last_n_seconds(self, audio_array, n_seconds=8, sample_rate=16000):
        """Truncate audio to last n seconds or pad with zeros to meet n seconds."""
        max_samples = n_seconds * sample_rate
        if len(audio_array) > max_samples:
            return audio_array[-max_samples:]
        elif len(audio_array) < max_samples:
            # Pad with zeros at the beginning
            padding = max_samples - len(audio_array)
            return np.pad(audio_array, (padding, 0), mode='constant', constant_values=0)
        return audio_array

    def process_file(self, audio_path: str) -> list:
        audio_tensor = self._load_and_validate_audio(audio_path)

        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.silero_vad_model,
            sampling_rate = self.sample_rate,
            threshold = self.silero_vad_threshold,
            max_speech_duration_s = self.max_speech_duration_s,
            min_silence_duration_ms = 200
        )

        if not speech_timestamps:
            logger.debug("No speech found in audio.")
            os.remove(audio_path) 
            return []
            
        results = []
        for segment in speech_timestamps:
            start_sample = segment['start']
            end_sample = segment['end']
            
            start_time = round(start_sample / self.sample_rate, 2)
            end_time = round(end_sample / self.sample_rate, 2)
            
            segment_audio_tensor = audio_tensor[start_sample:end_sample]
            
            # Ensure the segment_audio_tensor is on CPU for .numpy() if device is 'cuda'
            segment_audio_np = segment_audio_tensor.cpu().numpy()
            
            prediction_result = self.predict_endpoint(segment_audio_np) # Call instance method
            
            final_result = {
                "start_time": start_time,
                "end_time": end_time,
                "prediction": prediction_result['prediction'],
                "probability": round(prediction_result['probability'], 4)
            }
            results.append(final_result)
        
        return results

if __name__ == "__main__":
    from pathlib import Path
    TEST_WAV_FILES = list(Path("/home/nikita/balalaika/6271311").glob('*.mp3'))
    print(len(TEST_WAV_FILES))

    vad_processor = OfflineVAD(
        smart_vad_path='/home/nikita/yapoddataset/src/libs/smart_turn/smart-turn-v3.0.onnx',
        device='cuda:2'
        )
    
    start = time.time()
    for path in TEST_WAV_FILES:
        result = vad_processor.process_file(path)
        print(result)
    logger.info(time.time() - start)