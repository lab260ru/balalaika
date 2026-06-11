import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from typing import List, Tuple, Union
import time
import os
import huggingface_hub

# Model constants
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 128
PREEMPH = 0.97
LOG_ZERO_GUARD = 5.9604645e-8
SAMPLE_RATE = 16000

# Streaming constants defaults
CHUNK_LEN = 124
FIFO_LEN = 124
SPKCACHE_LEN = 188
RIGHT_CONTEXT = 1
SUBSAMPLING = 8
EMB_DIM = 512
NUM_SPEAKERS = 4
FRAME_DURATION = 0.08

# Cache compression params
SPKCACHE_SIL_FRAMES_PER_SPK = 3
PRED_SCORE_THRESHOLD = 0.25
STRONG_BOOST_RATE = 0.75
WEAK_BOOST_RATE = 1.5
MIN_POS_SCORES_RATE = 0.5
SIL_THRESHOLD = 0.2
MAX_INDEX = 999999


class DiarizationConfig:
    def __init__(self, onset=0.5, offset=0.5, pad_onset=0.0, pad_offset=0.0,
                 min_duration_on=0.0, min_duration_off=0.0, median_window=1):
        self.onset = onset
        self.offset = offset
        self.pad_onset = pad_onset
        self.pad_offset = pad_offset
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.median_window = median_window


class Sortformer:
    def __init__(
        self,
        model_path: str,
        config: DiarizationConfig = None,
        providers: List[str] = None,
        device: str = "cpu",
    ):
        if config is None:
            self.config = DiarizationConfig()
        else:
            self.config = config
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        if not os.path.exists(model_path):
            model_path = huggingface_hub.hf_hub_download(repo_id="altunenes/parakeet-rs", filename="diar_streaming_sortformer_4spk-v2.1.onnx", local_dir="./models")

        if providers is None:
            providers = [
                # (
                #     "TensorrtExecutionProvider",
                #     {
                #         "trt_max_workspace_size": 6 * 1024**3, 
                #         "trt_fp16_enable": True,
                #         "trt_engine_cache_enable": True,
                #         "trt_engine_cache_path": "./trt_cache",  
                #     }
                # ),
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ]
            
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        meta = self.session.get_modelmeta().custom_metadata_map
        self.chunk_len = int(meta.get("chunk_len", CHUNK_LEN))
        self.fifo_len = int(meta.get("fifo_len", FIFO_LEN))
        self.spkcache_len = int(meta.get("spkcache_len", SPKCACHE_LEN))
        self.right_context = int(meta.get("right_context", RIGHT_CONTEXT))

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=N_MELS,
            sample_rate=SAMPLE_RATE,
            n_stft=N_FFT // 2 + 1,
            norm="slaney",
            mel_scale="slaney",
        ).to(self.device)
        self.window = torch.hann_window(WIN_LENGTH, device=self.device)
        
        self.reset_state()

    def reset_state(self):
        self.spkcache = np.zeros((1, 0, EMB_DIM), dtype=np.float32)
        self.spkcache_preds = None
        self.fifo = np.zeros((1, 0, EMB_DIM), dtype=np.float32)
        self.fifo_preds = np.zeros((1, 0, NUM_SPEAKERS), dtype=np.float32)
        self.mean_sil_emb = np.zeros((1, EMB_DIM), dtype=np.float32)
        self.n_sil_frames = 0

    def extract_mel_features(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(audio, torch.Tensor):
            audio_tensor = audio.to(device=self.device, dtype=torch.float32)
        else:
            audio_tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32)).to(self.device)
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.mean(dim=0)
        preemphasized = torch.empty_like(audio_tensor)
        preemphasized[0] = audio_tensor[0]
        preemphasized[1:] = audio_tensor[1:] - PREEMPH * audio_tensor[:-1]
        spec = torch.stft(
            preemphasized,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        power_spec = spec.abs().pow(2)
        mel_spec = self.mel_scale(power_spec)
        log_mel_spec = torch.log(mel_spec + LOG_ZERO_GUARD)
        return log_mel_spec.transpose(0, 1).unsqueeze(0).cpu().numpy().astype(np.float32)

    def diarize(self, audio: Union[np.ndarray, torch.Tensor], sample_rate: int = 16000, include_tensor_outputs: bool = False) -> Union[List[List[str]], Tuple[List[List[str]], np.ndarray]]:
        if isinstance(audio, torch.Tensor):
            audio_tensor = audio.to(device=self.device, dtype=torch.float32)
        else:
            audio_tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32)).to(self.device)
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.mean(dim=0)
        if sample_rate != SAMPLE_RATE:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor,
                sample_rate,
                SAMPLE_RATE,
            )

        self.reset_state()
        
        features = self.extract_mel_features(audio_tensor)
        full_preds = self._process_features(features)
        
        if self.config.median_window > 1:
            from scipy.ndimage import median_filter
            filtered_preds = median_filter(full_preds, size=(self.config.median_window, 1))
        else:
            filtered_preds = full_preds

        audio_duration_sec = audio_tensor.numel() / SAMPLE_RATE
        segments = self._binarize(filtered_preds, audio_duration_sec)
        
        formatted_result = [segments]

        if include_tensor_outputs:
            return formatted_result, full_preds
        return formatted_result

    def _process_features(self, features: np.ndarray) -> np.ndarray:
        total_frames = features.shape[1]
        chunk_stride = self.chunk_len * SUBSAMPLING
        feed_size = (self.chunk_len + self.right_context) * SUBSAMPLING
        num_chunks = int(np.ceil(total_frames / chunk_stride))

        all_chunk_preds = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_stride
            end = min(start + feed_size, total_frames)
            current_len = end - start
            
            chunk_feat = features[:, start:end, :]

            if current_len < feed_size:
                padded = np.zeros((1, feed_size, N_MELS), dtype=np.float32)
                padded[:, :current_len, :] = chunk_feat
                chunk_feat = padded

            chunk_preds = self._streaming_update(chunk_feat, current_len)
            all_chunk_preds.append(chunk_preds)

        if len(all_chunk_preds) == 0:
            return np.zeros((0, NUM_SPEAKERS), dtype=np.float32)
            
        return np.concatenate(all_chunk_preds, axis=0)

    def _streaming_update(self, chunk_feat: np.ndarray, current_len: int) -> np.ndarray:
        spkcache_len = self.spkcache.shape[1]
        fifo_len = self.fifo.shape[1]

        inputs = {
            "chunk": chunk_feat,
            "chunk_lengths": np.array([current_len], dtype=np.int64),
            "spkcache": self.spkcache,
            "spkcache_lengths": np.array([spkcache_len], dtype=np.int64),
            "fifo": self.fifo,
            "fifo_lengths": np.array([fifo_len], dtype=np.int64)
        }

        outputs = self.session.run(["spkcache_fifo_chunk_preds", "chunk_pre_encode_embs"], inputs)
        
        preds = outputs[0]
        new_embs = outputs[1]
        
        valid_frames = int(np.ceil(current_len / SUBSAMPLING))
        fifo_preds = preds[:, spkcache_len:spkcache_len+fifo_len, :] if fifo_len > 0 else np.zeros((1, 0, NUM_SPEAKERS))
        
        keep = min(self.chunk_len, valid_frames)
        chunk_preds_idx_start = spkcache_len + fifo_len
        chunk_preds = preds[:, chunk_preds_idx_start:chunk_preds_idx_start+keep, :]
        chunk_embs = new_embs[:, :keep, :]

        self.fifo = np.concatenate([self.fifo, chunk_embs], axis=1)
        self.fifo_preds = np.concatenate([fifo_preds, chunk_preds], axis=1)
        
        fifo_len_after = self.fifo.shape[1]

        if fifo_len_after > self.fifo_len:
            pop_out_len = max(self.chunk_len, valid_frames - self.fifo_len + fifo_len)
            pop_out_len = min(pop_out_len, fifo_len_after)

            pop_out_embs = self.fifo[:, :pop_out_len, :]
            pop_out_preds = self.fifo_preds[:, :pop_out_len, :]

            self._update_silence_profile(pop_out_embs[0], pop_out_preds[0])

            self.fifo = self.fifo[:, pop_out_len:, :]
            self.fifo_preds = self.fifo_preds[:, pop_out_len:, :]

            self.spkcache = np.concatenate([self.spkcache, pop_out_embs], axis=1)
            
            if self.spkcache_preds is not None:
                self.spkcache_preds = np.concatenate([self.spkcache_preds, pop_out_preds], axis=1)

            if self.spkcache.shape[1] > self.spkcache_len:
                if self.spkcache_preds is None:
                    initial_cache_preds = preds[:, :spkcache_len, :]
                    self.spkcache_preds = np.concatenate([initial_cache_preds, pop_out_preds], axis=1)
                self._compress_spkcache()

        return chunk_preds[0]

    def _update_silence_profile(self, embs: np.ndarray, preds: np.ndarray):
        sums = np.sum(preds, axis=1)
        sil_mask = sums < SIL_THRESHOLD
        if np.any(sil_mask):
            sil_embs = embs[sil_mask]
            for emb in sil_embs:
                self.mean_sil_emb[0] = (self.mean_sil_emb[0] * self.n_sil_frames + emb) / (self.n_sil_frames + 1)
                self.n_sil_frames += 1

    def _compress_spkcache(self):
        if self.spkcache_preds is None: return

        n_frames = self.spkcache.shape[1]
        per_spk = self.spkcache_len // NUM_SPEAKERS
        if per_spk <= SPKCACHE_SIL_FRAMES_PER_SPK:
            self.spkcache = self.spkcache[:, :self.spkcache_len, :]
            self.spkcache_preds = self.spkcache_preds[:, :self.spkcache_len, :]
            return
            
        spkcache_len_per_spk = per_spk - SPKCACHE_SIL_FRAMES_PER_SPK
        strong_boost = int(spkcache_len_per_spk * STRONG_BOOST_RATE)
        weak_boost = int(spkcache_len_per_spk * WEAK_BOOST_RATE)
        min_pos = int(spkcache_len_per_spk * MIN_POS_SCORES_RATE)

        preds_2d = self.spkcache_preds[0]
        scores = self._get_log_pred_scores(preds_2d)
        scores = self._disable_low_scores(preds_2d, scores, min_pos)
        scores = self._boost_topk_scores(scores, strong_boost, 2.0)
        scores = self._boost_topk_scores(scores, weak_boost, 1.0)

        if SPKCACHE_SIL_FRAMES_PER_SPK > 0:
            padded = np.full((n_frames + SPKCACHE_SIL_FRAMES_PER_SPK, NUM_SPEAKERS), -np.inf, dtype=np.float32)
            padded[:n_frames, :] = scores
            padded[n_frames:, :] = np.inf
            scores = padded

        topk_indices, is_disabled = self._get_topk_indices(scores, n_frames)
        new_embs, new_preds = self._gather_spkcache(topk_indices, is_disabled)

        self.spkcache = new_embs
        self.spkcache_preds = new_preds

    def _get_log_pred_scores(self, preds: np.ndarray) -> np.ndarray:
        p = np.maximum(preds, PRED_SCORE_THRESHOLD)
        log_p = np.log(p)
        log_1_p = np.log(np.maximum(1.0 - preds, PRED_SCORE_THRESHOLD))
        log_1_probs_sum = np.sum(log_1_p, axis=1, keepdims=True)
        return log_p - log_1_p + log_1_probs_sum - np.log(0.5)

    def _disable_low_scores(self, preds: np.ndarray, scores: np.ndarray, min_pos: int) -> np.ndarray:
        pos_count = np.sum(scores > 0.0, axis=0)
        is_speech = preds > 0.5
        is_pos = scores > 0.0
        mask = (~is_speech) | ((~is_pos) & (pos_count >= min_pos))
        scores[mask] = -np.inf
        return scores

    def _boost_topk_scores(self, scores: np.ndarray, n_boost: int, scale_factor: float) -> np.ndarray:
        for s in range(NUM_SPEAKERS):
            col = scores[:, s].copy()
            top_idx = np.argsort(col)[::-1][:n_boost]
            valid_mask = scores[top_idx, s] != -np.inf
            scores[top_idx[valid_mask], s] -= scale_factor * np.log(0.5)
        return scores

    def _get_topk_indices(self, scores: np.ndarray, n_frames_no_sil: int) -> Tuple[List[int], List[bool]]:
        n_frames = scores.shape[0]
        flat_scores = scores.flatten('F') 
        sorted_flat_idx = np.argsort(flat_scores)[::-1]
        
        topk_flat = []
        for idx in sorted_flat_idx[:self.spkcache_len]:
            if flat_scores[idx] == -np.inf:
                topk_flat.append(MAX_INDEX)
            else:
                topk_flat.append(idx)
        topk_flat.sort()

        is_disabled = [False] * self.spkcache_len
        frame_indices = [0] * self.spkcache_len

        for i, flat_idx in enumerate(topk_flat):
            if flat_idx == MAX_INDEX:
                is_disabled[i] = True
            else:
                frame_idx = flat_idx % n_frames
                if frame_idx >= n_frames_no_sil:
                    is_disabled[i] = True
                else:
                    frame_indices[i] = frame_idx
        return frame_indices, is_disabled

    def _gather_spkcache(self, indices: List[int], is_disabled: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        new_embs = np.zeros((1, self.spkcache_len, EMB_DIM), dtype=np.float32)
        new_preds = np.zeros((1, self.spkcache_len, NUM_SPEAKERS), dtype=np.float32)
        cache_preds = self.spkcache_preds[0]
        cache_embs = self.spkcache[0]

        for i, (idx, disabled) in enumerate(zip(indices, is_disabled)):
            if disabled:
                new_embs[0, i, :] = self.mean_sil_emb[0]
            elif idx < cache_embs.shape[0]:
                new_embs[0, i, :] = cache_embs[idx]
                new_preds[0, i, :] = cache_preds[idx]
        return new_embs, np.expand_dims(new_preds[0], axis=0)

    def _binarize(self, preds: np.ndarray, audio_duration_sec: float) -> List[str]:
        raw_segments = []
        num_frames = preds.shape[0]

        for spk in range(NUM_SPEAKERS):
            raw_intervals = []
            in_seg = False
            start_t = 0.0

            for t in range(num_frames):
                p = preds[t, spk]
                if p >= self.config.onset and not in_seg:
                    in_seg = True
                    start_t = t * FRAME_DURATION
                elif p < self.config.offset and in_seg:
                    in_seg = False
                    raw_intervals.append([start_t, t * FRAME_DURATION, spk])

            if in_seg:
                raw_intervals.append([start_t, num_frames * FRAME_DURATION, spk])

            if not raw_intervals:
                continue

            merged_intervals = [raw_intervals[0]]
            for i in range(1, len(raw_intervals)):
                gap = raw_intervals[i][0] - merged_intervals[-1][1]
                if gap <= self.config.min_duration_off:
                    merged_intervals[-1][1] = raw_intervals[i][1]
                else:
                    merged_intervals.append(raw_intervals[i])

            filtered_intervals = []
            for seg in merged_intervals:
                if (seg[1] - seg[0]) >= self.config.min_duration_on:
                    filtered_intervals.append(seg)

            padded_intervals = []
            for seg in filtered_intervals:
                start_s = max(0.0, seg[0] - self.config.pad_onset)
                end_s = min(audio_duration_sec, seg[1] + self.config.pad_offset)
                
                if not padded_intervals:
                    padded_intervals.append([start_s, end_s, spk])
                else:
                    if start_s <= padded_intervals[-1][1]:
                        padded_intervals[-1][1] = max(padded_intervals[-1][1], end_s)
                    else:
                        padded_intervals.append([start_s, end_s, spk])

            raw_segments.extend(padded_intervals)

        raw_segments.sort(key=lambda x: (x[0], x[2]))

        str_segments = []
        for seg in raw_segments:
            str_segments.append(f"{seg[0]} {seg[1]} speaker_{seg[2]}")
            
        return str_segments



if __name__ == "__main__":
    model_path = "/home/nikita/balalaika/models/diar_streaming_sortformer_4spk-v2.1.onnx"
    audio_path = "/home/nikita/balalaika/datkamatka/12.mp3"
    
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    audio = waveform.squeeze(0).numpy()
    
    config = DiarizationConfig()
    diarizer = Sortformer(model_path, config=config)
    
    start_time = time.time()
    # print(audio.shape)
    results = diarizer.diarize(audio, sample_rate=16000, include_tensor_outputs=False)
    end_time = time.time()
    
    print(f"RTF: {(end_time - start_time) / (audio.shape[-1] / 16_000):.3f}")
    print(results)
