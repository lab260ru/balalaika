"""Batched RNN-T greedy decode must equal stock onnx-asr decode exactly.

Two layers of equivalence are pinned here:

1. **Loop logic** (no ONNX, deterministic, fast): tiny synthetic
   transducers that drive the *real* stock
   ``_AsrWithTransducerDecoding._decoding`` loop, compared token-for-token
   and timestamp-for-timestamp against the batched loops in
   ``src.transcription.fast_rnnt``.  This exercises the bookkeeping that is
   easy to get subtly wrong: the GigaAM predictor-advance-on-emit cache
   dance, ``max_tokens_per_step`` capping, the blank-or-maxstep frame
   advance, per-sequence ragged lengths within a batch, and logprob
   emission.  Both the GigaAM (stateful LSTM) and Kaldi (context cache)
   shapes are covered, with random and adversarial encoder outputs.

2. **Patch safety / fallback**: ``patch_model`` leaves non-transducer and
   unrecognized topologies untouched, is idempotent, and falls back to
   stock decode on a broken backend.

The real-model 250-file CPU equivalence (text + timestamps + tokens,
0/250 at batch sizes 1/4/8 for giga_rnnt and vosk) is proven by
``benchmarking/micro/bench_rnnt.py`` and recorded in report.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from onnx_asr.asr import _AsrWithTransducerDecoding
from src.transcription import fast_rnnt

ENC_DIM = 64
PRED_HIDDEN = 8


# ===================================================================== #
#  synthetic GigaAM-style transducer (stateful LSTM predictor)          #
# ===================================================================== #
class FakeGigaAsr(_AsrWithTransducerDecoding):
    """Minimal transducer with GigaAM's predictor-advances-on-emit dance.

    The "LSTM" is ``dec = tanh(W_x[token] + h @ W_h)`` with ``h`` carried as
    the new decoder output, faithfully reproducing onnx-asr's ``_decode``
    state caching (decoder runs only when ``len(prev_state) == 2``).
    """

    PRED_HIDDEN = PRED_HIDDEN

    def __init__(self, vocab=6, max_tokens=3, seed=0, joiner_scale=0.5):
        self._vocab_size = vocab
        self._blank_idx = vocab - 1
        self._mtps = max_tokens
        self.use_low_precision = False
        rng = np.random.default_rng(seed)
        self.W_x = rng.standard_normal((vocab, self.PRED_HIDDEN)).astype(np.float32)
        self.W_h = (rng.standard_normal((self.PRED_HIDDEN, self.PRED_HIDDEN)) * 0.3).astype(np.float32)
        self.W_je = (rng.standard_normal((ENC_DIM, vocab)) * 0.05).astype(np.float32)
        self.W_jd = (rng.standard_normal((self.PRED_HIDDEN, vocab)) * joiner_scale).astype(np.float32)

    @property
    def _max_tokens_per_step(self):
        return self._mtps

    @property
    def _subsampling_factor(self):
        return 4

    @property
    def _preprocessor_name(self):
        return "fake"

    @staticmethod
    def _get_model_files(quantization=None):
        return {}

    def _encode(self, *a):  # pragma: no cover - not used in tests
        raise NotImplementedError

    # ---- stock per-utterance interface (used by stock _decoding) ----
    def _create_state(self):
        return [
            np.zeros((1, 1, self.PRED_HIDDEN), np.float32),
            np.zeros((1, 1, self.PRED_HIDDEN), np.float32),
        ]

    def _predict(self, token, h):
        return np.tanh(self.W_x[token][None, None, :] + h @ self.W_h).astype(np.float32)

    def _joint(self, enc, dec):
        return (enc @ self.W_je + dec.reshape(-1) @ self.W_jd).astype(np.float32)

    def _decode(self, prev_tokens, prev_state, encoder_out):
        if len(prev_state) == 2:
            tok = prev_tokens[-1] if prev_tokens else self._blank_idx
            dec = self._predict(tok, prev_state[0])
            s1, s2 = dec, prev_state[1]
            prev_state[:] = (dec, s1, s2)
        else:
            dec, s1, s2 = prev_state
        joint = self._joint(encoder_out, dec)
        return np.squeeze(joint), -1, [s1, s2]


class FakeGigaBackend:
    """Batched backend mirroring :class:`FakeGigaAsr` math (no ONNX)."""

    def __init__(self, asr):
        self._asr = asr
        self._blank_idx = asr._blank_idx
        self._pred_hidden = asr.PRED_HIDDEN

    def initial_state(self, n):
        h = np.zeros((1, n, self._pred_hidden), np.float32)
        c = np.zeros((1, n, self._pred_hidden), np.float32)
        return h, c

    def run_decoder(self, tokens, state):
        h, c = state  # h: [1, k, H]
        dec = np.tanh(self._asr.W_x[tokens][None, :, :] + h @ self._asr.W_h).astype(np.float32)
        dec_b = dec.transpose(1, 0, 2)  # [k, 1, H]
        return dec_b, (dec, c)

    def run_joiner(self, enc_frames, decoder_out):
        dec = decoder_out.reshape(decoder_out.shape[0], -1)  # [k, H]
        return (enc_frames @ self._asr.W_je + dec @ self._asr.W_jd).astype(np.float32)


# ===================================================================== #
#  synthetic Kaldi-style transducer (stateless context predictor)       #
# ===================================================================== #
class FakeKaldiAsr(_AsrWithTransducerDecoding):
    CONTEXT_SIZE = 2

    def __init__(self, vocab=7, max_tokens=1, seed=1):
        self._vocab_size = vocab
        self._blank_idx = 0
        self._mtps = max_tokens
        self.use_low_precision = False
        rng = np.random.default_rng(seed)
        self.E = (rng.standard_normal((vocab, vocab, PRED_HIDDEN)) * 0.7).astype(np.float32)
        self.W_je = (rng.standard_normal((ENC_DIM, vocab)) * 0.05).astype(np.float32)
        self.W_jd = (rng.standard_normal((PRED_HIDDEN, vocab)) * 0.5).astype(np.float32)

    @property
    def _max_tokens_per_step(self):
        return self._mtps

    @property
    def _subsampling_factor(self):
        return 4

    @property
    def _preprocessor_name(self):
        return "fake"

    @staticmethod
    def _get_model_files(quantization=None):
        return {}

    def _encode(self, *a):  # pragma: no cover
        raise NotImplementedError

    def _create_state(self):
        return {}

    def _predict(self, context):
        a, b = context
        # clamp the sentinel -1 to a valid embedding row, like real graphs do
        return self.E[a % self.E.shape[0], b]

    def _joint(self, enc, dec):
        return (enc @ self.W_je + dec @ self.W_jd).astype(np.float32)

    def _decode(self, prev_tokens, prev_state, encoder_out):
        context = (-1, self._blank_idx, *prev_tokens)[-self.CONTEXT_SIZE:]
        decoder_out = prev_state.get(context)
        if decoder_out is None:
            decoder_out = self._predict(context)
            prev_state[context] = decoder_out
        joint = self._joint(encoder_out, decoder_out)
        return np.squeeze(joint), -1, prev_state


class FakeKaldiBackend:
    def __init__(self, asr):
        self._asr = asr
        self._blank_idx = asr._blank_idx
        self._context_size = asr.CONTEXT_SIZE
        self._cache = {}

    def reset(self):
        self._cache = {}

    def context_for(self, prev_tokens):
        return (-1, self._blank_idx, *prev_tokens)[-self._context_size:]

    def decoder_out_for(self, contexts):
        out = []
        for ctx in contexts:
            if ctx not in self._cache:
                self._cache[ctx] = self._asr._predict(ctx)[None, :]
            out.append(self._cache[ctx])
        return out

    def run_joiner(self, enc_frames, decoder_out):
        return (enc_frames @ self._asr.W_je + decoder_out @ self._asr.W_jd).astype(np.float32)


# ===================================================================== #
#  helpers                                                              #
# ===================================================================== #
def _stock_decode(asr, enc, enc_lens, need_logprobs=None):
    return [
        (list(t), list(ts), None if lp is None else list(lp))
        for t, ts, lp in asr._decoding(enc, enc_lens, need_logprobs=need_logprobs)
    ]


def _assert_equal(stock, fast, logprob_atol=0.0):
    """Tokens and timestamps must be bit-exact (the stage's products);
    logprobs may differ by ``logprob_atol`` to absorb fp32 matmul
    reassociation between the per-row stock joiner and the batched joiner
    (the argmax token stream is unaffected — see report.md)."""
    assert len(stock) == len(fast)
    for i, (s, f) in enumerate(zip(stock, fast)):
        assert list(s[0]) == list(f[0]), f"seq {i} tokens differ:\n  stock {s[0]}\n  fast  {f[0]}"
        assert list(s[1]) == list(f[1]), f"seq {i} timestamps differ"
        if s[2] is None:
            assert f[2] is None
        else:
            assert np.allclose(s[2], f[2], atol=logprob_atol, rtol=0), f"seq {i} logprobs differ"


# ===================================================================== #
#  GigaAM loop equivalence                                             #
# ===================================================================== #
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("max_tokens", [1, 2, 3])
def test_gigaam_batched_matches_stock(seed, max_tokens):
    asr = FakeGigaAsr(max_tokens=max_tokens, seed=seed)
    backend = FakeGigaBackend(asr)
    rng = np.random.default_rng(seed)
    b = int(rng.integers(1, 7))
    tmax = int(rng.integers(3, 20))
    enc = rng.standard_normal((b, tmax, ENC_DIM)).astype(np.float32)
    enc_lens = rng.integers(1, tmax + 1, size=b).astype(np.int64)

    # stock processes each sequence over its own length; emulate the slice
    # the batched loop sees (the same enc array, enc_lens bound it).
    stock = _stock_decode(asr, enc, enc_lens)
    fast = list(fast_rnnt._decoding_gigaam_batched(asr, backend, enc, enc_lens, None))
    _assert_equal(stock, fast)


def test_gigaam_logprobs_match_stock():
    asr = FakeGigaAsr(seed=3)
    backend = FakeGigaBackend(asr)
    rng = np.random.default_rng(9)
    enc = rng.standard_normal((4, 15, ENC_DIM)).astype(np.float32)
    enc_lens = np.array([15, 10, 1, 7], dtype=np.int64)
    stock = _stock_decode(asr, enc, enc_lens, need_logprobs="yes")
    fast = [
        (t, ts, lp)
        for t, ts, lp in fast_rnnt._decoding_gigaam_batched(asr, backend, enc, enc_lens, "yes")
    ]
    _assert_equal(stock, fast, logprob_atol=1e-5)


def test_gigaam_all_blank_and_max_emit():
    """A predictor biased to never blank hits max_tokens_per_step every frame;
    one biased to always blank emits nothing."""
    for scale, seed in [(40.0, 0), (-40.0, 1)]:
        asr = FakeGigaAsr(seed=seed, joiner_scale=scale)
        backend = FakeGigaBackend(asr)
        enc = np.random.default_rng(seed).standard_normal((3, 8, ENC_DIM)).astype(np.float32)
        enc_lens = np.array([8, 5, 2], dtype=np.int64)
        _assert_equal(
            _stock_decode(asr, enc, enc_lens),
            list(fast_rnnt._decoding_gigaam_batched(asr, backend, enc, enc_lens, None)),
        )


def test_gigaam_single_sequence_batches():
    """Each batch size gives the same per-sequence result when the encoder
    output is identical (the loop itself is batch-invariant)."""
    asr = FakeGigaAsr(seed=5)
    backend = FakeGigaBackend(asr)
    rng = np.random.default_rng(5)
    enc = rng.standard_normal((8, 14, ENC_DIM)).astype(np.float32)
    enc_lens = rng.integers(1, 15, size=8).astype(np.int64)
    full = list(fast_rnnt._decoding_gigaam_batched(asr, backend, enc, enc_lens, None))
    for i in range(8):
        one = list(
            fast_rnnt._decoding_gigaam_batched(
                asr, backend, enc[i : i + 1], enc_lens[i : i + 1], None
            )
        )
        assert list(one[0][0]) == list(full[i][0])
        assert list(one[0][1]) == list(full[i][1])


# ===================================================================== #
#  Kaldi loop equivalence                                              #
# ===================================================================== #
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("max_tokens", [1, 2])
def test_kaldi_batched_matches_stock(seed, max_tokens):
    asr = FakeKaldiAsr(max_tokens=max_tokens, seed=seed)
    backend = FakeKaldiBackend(asr)
    rng = np.random.default_rng(seed + 100)
    b = int(rng.integers(1, 7))
    tmax = int(rng.integers(3, 20))
    enc = rng.standard_normal((b, tmax, ENC_DIM)).astype(np.float32)
    enc_lens = rng.integers(1, tmax + 1, size=b).astype(np.int64)
    stock = _stock_decode(asr, enc, enc_lens)
    fast = list(fast_rnnt._decoding_kaldi_batched(asr, backend, enc, enc_lens, None))
    _assert_equal(stock, fast)


def test_kaldi_logprobs_match_stock():
    asr = FakeKaldiAsr(seed=2)
    backend = FakeKaldiBackend(asr)
    rng = np.random.default_rng(11)
    enc = rng.standard_normal((3, 12, ENC_DIM)).astype(np.float32)
    enc_lens = np.array([12, 6, 3], dtype=np.int64)
    stock = _stock_decode(asr, enc, enc_lens, need_logprobs="yes")
    fast = list(fast_rnnt._decoding_kaldi_batched(asr, backend, enc, enc_lens, "yes"))
    _assert_equal(stock, fast, logprob_atol=1e-5)


def test_kaldi_context_cache_shared_across_batch():
    """The shared decoder-out cache must not change outputs vs the stock
    per-utterance cache."""
    asr = FakeKaldiAsr(seed=7)
    backend = FakeKaldiBackend(asr)
    rng = np.random.default_rng(7)
    enc = rng.standard_normal((6, 18, ENC_DIM)).astype(np.float32)
    enc_lens = rng.integers(1, 19, size=6).astype(np.int64)
    _assert_equal(
        _stock_decode(asr, enc, enc_lens),
        list(fast_rnnt._decoding_kaldi_batched(asr, backend, enc, enc_lens, None)),
    )


# ===================================================================== #
#  dynamic-batch rebuild routes ORT options through make_session_options #
# ===================================================================== #
def _tiny_onnx_batch1(path) -> None:
    """Write a minimal ONNX model with a batch-1 leading dim to relabel."""
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


class _FakeSession:
    """Stand-in for an onnxruntime InferenceSession with a real model file."""

    def __init__(self, model_path):
        self._model_path = str(model_path)

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_provider_options(self):
        return {"CPUExecutionProvider": {}}


def test_rebuild_dynamic_batch_uses_make_session_options(tmp_path, monkeypatch):
    """The dynamic-batch decoder/joiner sessions must build their
    SessionOptions through src.utils.gpu.make_session_options so the
    runtime.threads_per_worker intra-op cap is honoured inside the spawned
    ASR workers (rather than a hand-rolled SessionOptions that bypasses it)."""
    import onnxruntime as rt

    from src.utils import gpu

    model_path = tmp_path / "tiny.onnx"
    _tiny_onnx_batch1(model_path)

    calls = []
    real_make = gpu.make_session_options

    def _recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return real_make(*args, **kwargs)

    # Patch the name as fast_rnnt references it (mirrors transcription.py's
    # `from src.utils.gpu import make_session_options`).
    monkeypatch.setattr(fast_rnnt, "make_session_options", _recorder, raising=True)

    sess = _FakeSession(model_path)
    out = fast_rnnt._rebuild_dynamic_batch(
        sess, input_batch_axis={"x": 0}, output_batch_axis={"y": 0}
    )
    assert isinstance(out, rt.InferenceSession)
    assert len(calls) == 1, "make_session_options not used to build ORT options"


def test_gigaam_backend_builds_both_sessions_via_make_session_options(
    tmp_path, monkeypatch
):
    """Both the decoder and the joiner dynamic-batch sessions must be built
    through make_session_options (two calls for a GigaAM backend)."""
    from src.utils import gpu

    model_path = tmp_path / "tiny.onnx"
    _tiny_onnx_batch1(model_path)

    calls = []
    real_make = gpu.make_session_options

    def _recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return real_make(*args, **kwargs)

    monkeypatch.setattr(fast_rnnt, "make_session_options", _recorder, raising=True)

    # Rebuild relabels axis 0 on "x"/"y"; reuse the same tiny graph for both
    # the decoder and joiner stand-ins so _GigaamBackend.__init__ runs.
    class _Asr:
        PRED_HIDDEN = PRED_HIDDEN

        def __init__(self):
            self._blank_idx = 5
            self._decoder = _FakeSession(model_path)
            self._joiner = _FakeSession(model_path)

    real_rebuild = fast_rnnt._rebuild_dynamic_batch

    def _rebuild(session, input_batch_axis, output_batch_axis):
        return real_rebuild(
            session, input_batch_axis={"x": 0}, output_batch_axis={"y": 0}
        )

    monkeypatch.setattr(fast_rnnt, "_rebuild_dynamic_batch", _rebuild, raising=True)

    fast_rnnt._GigaamBackend(_Asr())
    assert len(calls) == 2, (
        "expected make_session_options for both decoder and joiner sessions, "
        f"got {len(calls)}"
    )


# ===================================================================== #
#  patch_model safety / fallback                                       #
# ===================================================================== #
class _FakeCtcModel:
    """A non-transducer model: no _max_tokens_per_step -> must stay stock."""

    class _Asr:
        def _decoding(self, *a, **k):
            return iter([])

    def __init__(self):
        self.asr = self._Asr()


def test_patch_leaves_non_transducer_untouched():
    model = _FakeCtcModel()
    before = model.asr._decoding.__func__
    out = fast_rnnt.patch_model(model)
    assert out is model
    assert model.asr._decoding.__func__ is before
    assert not fast_rnnt.is_patched(model)


def test_patch_leaves_unknown_topology_untouched():
    """A transducer base we don't recognize (no GigaAM/Kaldi class) stays stock."""

    class Unknown(_AsrWithTransducerDecoding):
        def __init__(self):
            self._blank_idx = 0
            self.use_low_precision = False

        @property
        def _max_tokens_per_step(self):
            return 1

        @property
        def _subsampling_factor(self):
            return 4

        @property
        def _preprocessor_name(self):
            return "x"

        @staticmethod
        def _get_model_files(quantization=None):
            return {}

        def _encode(self, *a):
            raise NotImplementedError

        def _create_state(self):
            return {}

        def _decode(self, *a):
            raise NotImplementedError

    class M:
        pass

    m = M()
    m.asr = Unknown()
    before = m.asr._decoding.__func__
    fast_rnnt.patch_model(m)
    assert m.asr._decoding.__func__ is before
    assert not fast_rnnt.is_patched(m)


def test_patch_is_idempotent():
    asr = FakeGigaAsr(seed=0)

    class M:
        pass

    m = M()
    m.asr = asr
    # Give it a fake backend builder so patch succeeds without ONNX.
    import src.transcription.fast_rnnt as fr

    orig = fr._make_backend
    fr._make_backend = lambda a: (FakeGigaBackend(a), "gigaam")
    try:
        fr.patch_model(m)
        assert fr.is_patched(m)
        patched_once = m.asr._decoding
        fr.patch_model(m)  # second call is a no-op
        assert m.asr._decoding is patched_once
    finally:
        fr._make_backend = orig


def test_patched_decode_matches_stock_through_recognize_path():
    """End-to-end through the patched _decoding (the method recognize_batch
    calls), comparing to a pristine stock instance's _decoding."""
    import src.transcription.fast_rnnt as fr

    stock_asr = FakeGigaAsr(seed=4)
    fast_asr = FakeGigaAsr(seed=4)

    class M:
        pass

    m = M()
    m.asr = fast_asr
    orig = fr._make_backend
    fr._make_backend = lambda a: (FakeGigaBackend(a), "gigaam")
    try:
        fr.patch_model(m)
    finally:
        fr._make_backend = orig
    assert fr.is_patched(m)

    rng = np.random.default_rng(123)
    enc = rng.standard_normal((5, 16, ENC_DIM)).astype(np.float32)
    enc_lens = rng.integers(1, 17, size=5).astype(np.int64)
    stock = _stock_decode(stock_asr, enc, enc_lens)
    fast = [(list(t), list(ts), lp) for t, ts, lp in m.asr._decoding(enc, enc_lens)]
    _assert_equal(stock, fast)


def test_fallback_on_broken_backend():
    """If the batched loop raises, the patched _decoding falls back to stock
    for that batch (proven by getting the correct stock answer back)."""
    import src.transcription.fast_rnnt as fr

    asr = FakeGigaAsr(seed=6)

    class BrokenBackend(FakeGigaBackend):
        def run_joiner(self, *a, **k):
            raise RuntimeError("boom")

    class M:
        pass

    m = M()
    m.asr = asr
    orig = fr._make_backend
    fr._make_backend = lambda a: (BrokenBackend(a), "gigaam")
    try:
        fr.patch_model(m)
    finally:
        fr._make_backend = orig

    rng = np.random.default_rng(321)
    enc = rng.standard_normal((3, 10, ENC_DIM)).astype(np.float32)
    enc_lens = np.array([10, 5, 2], dtype=np.int64)
    # stock reference from an unpatched twin
    ref = _stock_decode(FakeGigaAsr(seed=6), enc, enc_lens)
    got = [(list(t), list(ts), lp) for t, ts, lp in m.asr._decoding(enc, enc_lens)]
    _assert_equal(ref, got)
