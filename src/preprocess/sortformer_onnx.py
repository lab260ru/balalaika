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


def _silence_running_mean_py(mean_row: np.ndarray, n: int, sil_embs: np.ndarray):
    """Verbatim sequential float32 running-mean update (Python reference).

    ``mean_row`` is updated in place. Returns ``(mean_row, n_after)``. The
    expression mirrors the original ``(mean * n + emb) / (n + 1)`` where
    ``mean`` is a float32 array and ``n`` a Python int, so numpy keeps every
    intermediate in float32.
    """
    for emb in sil_embs:
        mean_row = (mean_row * n + emb) / (n + 1)
        n += 1
    return mean_row, n


def _make_silence_running_mean():
    """Return a bit-exact silence running-mean updater.

    Prefers a numba ``njit`` kernel (the per-frame loop is sequential, so
    numpy cannot vectorize it without changing summation order). The kernel
    uses explicit ``float32`` casts so each scalar op matches numpy's
    ``float32_array * python_int -> float32`` promotion exactly; this was
    verified bit-identical against the Python reference on randomized inputs.
    Falls back to the pure-Python loop if numba is unavailable or fails to
    compile, so behavior is identical everywhere.
    """
    try:
        import numba  # noqa: F401

        @numba.njit(cache=True)
        def _kernel(mean_row, n, sil_embs):  # pragma: no cover - jitted
            out = mean_row.copy()
            nn = n
            for i in range(sil_embs.shape[0]):
                nf = np.float32(nn)
                np1 = np.float32(nn + 1)
                for d in range(out.shape[0]):
                    out[d] = (out[d] * nf + sil_embs[i, d]) / np1
                nn += 1
            return out, nn

        def _runner(mean_row, n, sil_embs):
            if sil_embs.shape[0] == 0:
                return mean_row, n
            out, nn = _kernel(
                np.ascontiguousarray(mean_row, dtype=np.float32),
                n,
                np.ascontiguousarray(sil_embs, dtype=np.float32),
            )
            return out, int(nn)

        return _runner
    except Exception:  # numba missing or broken -> exact Python loop
        return _silence_running_mean_py


_silence_running_mean = _make_silence_running_mean()


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
        use_io_binding: bool = False,
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
        # IOBinding keeps the streaming session's input/output tensors resident
        # on the device across the per-chunk loop instead of round-tripping
        # every state tensor through numpy. Same graph + same execution provider
        # => identical numerics (proven against the numpy path on a synthetic
        # ONNX model in tests); residency is the only difference. Default off so
        # production behavior is byte-for-byte the current numpy path until it is
        # measured on a node that has the real model.
        self._setup_io_binding(use_io_binding)

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

    def _setup_io_binding(self, use_io_binding: bool) -> None:
        self._stream_output_names = ["spkcache_fifo_chunk_preds", "chunk_pre_encode_embs"]
        self.use_io_binding = bool(use_io_binding)
        self._io_binding = None
        self._iobinding_device = "cpu"
        self._iobinding_device_id = 0
        if not self.use_io_binding:
            return
        # Bind inputs/outputs on the device the CUDA EP runs on; if the session
        # has no CUDA provider (CPU-only node) bind on CPU — still numerically
        # identical, just no residency win. ``run_with_iobinding`` over device
        # OrtValues avoids ORT re-copying state H2D/D2H on every chunk.
        try:
            session_providers = self.session.get_providers()
            if "CUDAExecutionProvider" in session_providers:
                self._iobinding_device = "cuda"
                if self.device.type == "cuda" and self.device.index is not None:
                    self._iobinding_device_id = int(self.device.index)
            self._io_binding = self.session.io_binding()
        except Exception as exc:  # pragma: no cover - defensive
            from loguru import logger

            logger.warning(
                f"Sortformer IOBinding setup failed ({exc}); using numpy session.run path."
            )
            self.use_io_binding = False
            self._io_binding = None

    def _run_session(self, inputs: dict) -> List[np.ndarray]:
        """Run the streaming session for one chunk, returning numpy outputs.

        Two residency paths sharing one input/output contract:

        * numpy (default): ``session.run`` — ORT copies inputs H2D and outputs
          D2H internally, the historical behavior.
        * IOBinding: inputs are wrapped as device ``OrtValue``s and outputs are
          bound on-device, so state tensors stay resident across chunks. Same
          graph and EP, so outputs are bit-identical (pinned in tests).
        """
        if not self.use_io_binding or self._io_binding is None:
            return self.session.run(self._stream_output_names, inputs)

        iob = self._io_binding
        iob.clear_binding_inputs()
        iob.clear_binding_outputs()
        for name, arr in inputs.items():
            arr = np.ascontiguousarray(arr)
            iob.bind_ortvalue_input(
                name,
                ort.OrtValue.ortvalue_from_numpy(
                    arr, self._iobinding_device, self._iobinding_device_id
                ),
            )
        for name in self._stream_output_names:
            iob.bind_output(name, self._iobinding_device, self._iobinding_device_id)
        self.session.run_with_iobinding(iob)
        return iob.copy_outputs_to_cpu()

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

        outputs = self._run_session(inputs)

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
            # The running mean is updated one frame at a time so the result
            # feeds back into the model; the float32 rounding of that exact
            # scalar op order must be preserved (a closed-form
            # (mean*n + sum)/(n+k) drifts ~1e-7 and would perturb embeddings).
            # ``_silence_running_mean`` reproduces the verbatim sequential
            # arithmetic bit-for-bit — a numba kernel when available (with the
            # same explicit float32 casts), else the original Python loop.
            self.mean_sil_emb[0], self.n_sil_frames = _silence_running_mean(
                self.mean_sil_emb[0], int(self.n_sil_frames), sil_embs
            )

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
        if n_boost <= 0:
            return scores
        # Per-column descending top-n_boost, vectorized across speakers. The
        # per-column ``np.argsort(col)[::-1]`` tie-order is preserved because
        # ``np.argsort(scores, axis=0)[::-1]`` reverses the same stable
        # ascending sort independently per column.
        order = np.argsort(scores, axis=0)[::-1][:n_boost]
        cols = np.arange(NUM_SPEAKERS)
        valid = scores[order, cols] != -np.inf
        sel_rows = order[valid]
        sel_cols = np.broadcast_to(cols, order.shape)[valid]
        scores[sel_rows, sel_cols] -= scale_factor * np.log(0.5)
        return scores

    def _get_topk_indices(self, scores: np.ndarray, n_frames_no_sil: int) -> Tuple[List[int], List[bool]]:
        n_frames = scores.shape[0]
        flat_scores = scores.flatten('F')
        sorted_flat_idx = np.argsort(flat_scores)[::-1][:self.spkcache_len]
        # -inf entries map to the sentinel MAX_INDEX, then the whole top-k list
        # is sorted ascending (sentinels sink to the end). Fully vectorized;
        # the trailing slots stay at their defaults when fewer than
        # spkcache_len entries exist (matches the original list semantics).
        topk = np.where(
            flat_scores[sorted_flat_idx] == -np.inf, MAX_INDEX, sorted_flat_idx
        ).astype(np.int64)
        topk.sort()

        is_disabled = np.zeros(self.spkcache_len, dtype=bool)
        frame_indices = np.zeros(self.spkcache_len, dtype=np.int64)
        k = topk.shape[0]
        if k:
            is_sentinel = topk == MAX_INDEX
            frame_idx = np.where(is_sentinel, 0, topk % n_frames)
            disabled_k = is_sentinel | (frame_idx >= n_frames_no_sil)
            is_disabled[:k] = disabled_k
            keep = ~disabled_k
            frame_indices[:k][keep] = frame_idx[keep]
        return frame_indices.tolist(), is_disabled.tolist()

    def _gather_spkcache(self, indices: List[int], is_disabled: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        new_embs = np.zeros((1, self.spkcache_len, EMB_DIM), dtype=np.float32)
        new_preds = np.zeros((1, self.spkcache_len, NUM_SPEAKERS), dtype=np.float32)
        cache_preds = self.spkcache_preds[0]
        cache_embs = self.spkcache[0]

        idx = np.asarray(indices, dtype=np.int64)
        disabled = np.asarray(is_disabled, dtype=bool)
        new_embs[0, disabled, :] = self.mean_sil_emb[0]
        valid = (~disabled) & (idx < cache_embs.shape[0])
        valid_idx = idx[valid]
        new_embs[0, valid, :] = cache_embs[valid_idx]
        new_preds[0, valid, :] = cache_preds[valid_idx]
        return new_embs, np.expand_dims(new_preds[0], axis=0)

    @staticmethod
    def _threshold_intervals(active: np.ndarray, spk: int) -> List[list]:
        """Rise/fall edge detection for a boolean activity mask (onset==offset).

        Equivalent to the per-frame state machine when ``onset == offset``:
        a segment opens on the first frame with ``p >= onset`` and closes on
        the first subsequent frame with ``p < offset``. With a single
        threshold those two events are exactly the rising/falling edges of the
        boolean mask, found vectorized via ``np.diff`` over a zero-padded
        int8 view. Returns ``[start_s, end_s, spk]`` intervals in frame order.
        """
        if not active.any():
            return []
        padded = np.concatenate(([0], active.view(np.int8), [0]))
        diff = np.diff(padded)
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)
        return [
            [float(s) * FRAME_DURATION, float(e) * FRAME_DURATION, spk]
            for s, e in zip(starts, ends)
        ]

    def _binarize(self, preds: np.ndarray, audio_duration_sec: float) -> List[str]:
        raw_segments = []
        num_frames = preds.shape[0]
        # The default DiarizationConfig uses onset == offset (0.5), which turns
        # the per-frame hysteresis into a plain threshold; that case is
        # vectorized with edge detection. True hysteresis (onset != offset) is
        # sequential by nature, so it keeps the verbatim per-frame loop.
        single_threshold = self.config.onset == self.config.offset

        for spk in range(NUM_SPEAKERS):
            if single_threshold:
                raw_intervals = self._threshold_intervals(
                    preds[:, spk] >= self.config.onset, spk
                )
            else:
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
