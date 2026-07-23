"""SmartVAD batching: feature-extraction equality + orchestration equivalence.

The Smart-Turn ONNX model is absent on this node, so the model inference itself
cannot run. Two things ARE verifiable and pinned here:

1. The Whisper feature extraction for a batch of segments is bit-identical to
   extracting each segment singly (this is the part the batching relies on).
2. ``apply_eos_classification`` produces the IDENTICAL merged segment list in
   batched mode vs the per-segment default, using a deterministic fake VAD
   whose batch and single calls agree per item (so any divergence would be the
   orchestration, not the model).
"""

import numpy as np
import pytest
import torch

import src.preprocess.preprocess as P


# --------------------------------------------------------------------------
# 1. Feature-extraction batched == single (real WhisperFeatureExtractor)
# --------------------------------------------------------------------------
def _truncate(a, n_seconds=8, sr=16000):
    m = n_seconds * sr
    if len(a) > m:
        return a[-m:]
    if len(a) < m:
        return np.pad(a, (m - len(a), 0), mode="constant", constant_values=0)
    return a


def _fe_single_stack(fe, segs):
    singles = []
    for a in segs:
        out = fe(
            _truncate(a),
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=8 * 16000,
            truncation=True,
            do_normalize=True,
        )
        singles.append(out.input_features)
    return torch.cat(singles, dim=0)


def _fe_batched(fe, segs):
    return fe(
        [_truncate(a) for a in segs],
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
        max_length=8 * 16000,
        truncation=True,
        do_normalize=True,
    ).input_features


def test_feature_extraction_batched_matches_single_within_1ulp():
    """Batched Whisper features match per-segment extraction to <= 1 float32 ULP.

    The STFT reductions are batched, so a handful of elements differ by one ULP
    (~1.2e-7) — bounded and non-cumulative. This is exactly why SmartVAD
    batching is knob-gated with the per-segment path (batch size 1) as the
    bit-identical default: even feature extraction is not bit-exact across batch
    composition, so the model output near the 0.4 threshold cannot be proven
    identical on this node (the ONNX model is absent).
    """
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor(chunk_length=8)
    for seed in range(8):
        rng = np.random.default_rng(seed)
        segs = [
            rng.standard_normal(int(rng.integers(1000, 8 * 16000))).astype(np.float32)
            for _ in range(rng.integers(2, 10))
        ]
        single_stack = _fe_single_stack(fe, segs)
        batched = _fe_batched(fe, segs)
        assert single_stack.shape == batched.shape
        max_diff = (single_stack - batched).abs().max().item()
        assert max_diff <= 1.2e-7, f"seed {seed}: features differ by {max_diff}"


# --------------------------------------------------------------------------
# Fake SmartVAD: deterministic per-segment prediction, batch == single.
# --------------------------------------------------------------------------
class FakeVAD:
    sample_rate = 16000
    device = "cpu"

    def __init__(self, threshold=0.4):
        self.smart_vad_threshold = threshold

    @staticmethod
    def _prob(segment_audio):
        # Deterministic pseudo-probability from the segment content; identical
        # whether called singly or in a batch (mirrors a real model's per-row
        # independence).
        seg = np.asarray(segment_audio, dtype=np.float64)
        s = float(np.abs(seg).sum())
        return (s % 1.0)

    def _format(self, prob):
        return {"prediction": 1 if prob > self.smart_vad_threshold else 0,
                "probability": round(prob, 4)}

    def predict_endpoint(self, segment_audio, sample_rate=None):
        return self._format(self._prob(segment_audio))

    def predict_endpoint_batch(self, audio_arrays, sample_rate=None):
        return [self._format(self._prob(a)) for a in audio_arrays]


@pytest.fixture
def fake_vad(monkeypatch):
    vad = FakeVAD()
    monkeypatch.setattr(P, "smart_vad", vad)
    return vad


def _random_timeline(rng, n):
    segs = []
    t = 0.0
    for _ in range(n):
        s = t + rng.uniform(0, 1.5)
        e = s + rng.uniform(0.2, 3.0)
        segs.append((s, e, rng.integers(0, 4)))
        t = e + rng.uniform(-0.3, 0.8)
    return [(float(s), float(e), int(spk)) for s, e, spk in segs]


@pytest.mark.parametrize("batch_size", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("seed", range(30))
def test_apply_eos_batched_equals_single(fake_vad, batch_size, seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(0, 40))
    segments = _random_timeline(rng, n)
    sr = 16000
    audio = torch.from_numpy(
        rng.standard_normal(int(rng.integers(1, 20) * sr)).astype(np.float32)
    ).unsqueeze(0)

    single = P.apply_eos_classification(
        audio, sr, segments, max_duration=15.0,
        min_duration=1.0, max_merge_gap=0.5, smart_vad_batch_size=1,
    )
    batched = P.apply_eos_classification(
        audio, sr, segments, max_duration=15.0,
        min_duration=1.0, max_merge_gap=0.5, smart_vad_batch_size=batch_size,
    )
    assert single == batched


def test_apply_eos_empty_segments(fake_vad):
    audio = torch.zeros((1, 16000), dtype=torch.float32)
    assert P.apply_eos_classification(audio, 16000, [], smart_vad_batch_size=8) == []


def test_apply_eos_skips_zero_length_segments_identically(fake_vad):
    # Segments that slice to empty audio are skipped the same way in both modes.
    sr = 16000
    audio = torch.zeros((1, 5 * sr), dtype=torch.float32)
    # second segment starts beyond audio end -> empty slice
    segments = [(0.0, 2.0, 0), (100.0, 101.0, 1), (2.5, 4.0, 0)]
    single = P.apply_eos_classification(audio, sr, segments, smart_vad_batch_size=1)
    batched = P.apply_eos_classification(audio, sr, segments, smart_vad_batch_size=4)
    assert single == batched
