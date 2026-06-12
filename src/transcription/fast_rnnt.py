"""Batched stateful RNN-T greedy decode for onnx-asr transducer models.

Stock ``onnx_asr`` decodes RNN-T utterances strictly one at a time
(``_AsrWithTransducerDecoding._decoding`` loops ``for encodings in
encoder_out``), and within each utterance fires a *batch-1* decoder/joiner
ONNX call per greedy step.  For ``gigaam-v3-rnnt`` that is 200+ sequential
ONNX ``Run`` calls per file — the stage-7 critical path (~7 it/s vs ~60 for
the CTC models; see report.md §9.1).

This module keeps the **exact** stock greedy algorithm but runs the
decoder and joiner across all utterances of a batch in lockstep, so each
greedy step is one batched ONNX call instead of B sequential ones.  The
weights, vocab, ``max_tokens_per_step`` semantics, blank handling, fp32
numerics and emitted ``(tokens, timestamps, logprobs)`` tuples are
identical to stock — only the batch dimension is added.

Two transducer topologies are supported, both detected at patch time:

* **GigaAM v2/v3 RNN-T** (``GigaamV2Rnnt``): an LSTM predictor whose ONNX
  decoder/joiner graphs hardcode a batch dim of 1.  We rewrite just that
  leading dim to a symbolic ``B`` in-memory (the ops are batch-agnostic;
  verified bit-identical decoder output and <4e-6 joiner output vs the
  stock B=1 graph) and build fresh dynamic-batch sessions reusing the
  stock sessions' providers/options.

* **Kaldi / Vosk transducer** (``KaldiTransducer``): a stateless 2-token
  context predictor whose graphs already carry a dynamic batch dim ``N``;
  no rewrite is needed (verified bit-identical batched vs single).  The
  per-context decoder-output cache is shared across the whole batch, so a
  context decoded for one utterance is reused for every other.

Anything else (NeMo transducers, unknown attributes, missing sessions)
falls back to the stock per-utterance ``_decoding`` automatically.

Knob: ``transcription.use_fast_rnnt`` (default True — equivalence is
exact for argmax token streams; pinned by tests/test_fast_rnnt.py and the
250-file CPU equivalence proof in report.md).
"""
from __future__ import annotations

from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import onnxruntime as rt
from loguru import logger

from src.utils.gpu import make_session_options

# log_softmax lives in the installed package; import defensively so a future
# onnx-asr layout change degrades to "no fast path" rather than crashing.
try:  # pragma: no cover - import guard
    from onnx_asr.utils import log_softmax as _log_softmax
except Exception:  # pragma: no cover
    def _log_softmax(logits, axis=None):
        tmp = logits - np.max(logits, axis=axis, keepdims=axis is not None)
        tmp = tmp - np.log(np.sum(np.exp(tmp), axis=axis, keepdims=axis is not None))
        return tmp


# ====================================================================== #
#  dynamic-batch graph rewrite (GigaAM only)                             #
# ====================================================================== #
def _rebuild_dynamic_batch(
    session: rt.InferenceSession,
    input_batch_axis: dict,
    output_batch_axis: dict,
) -> rt.InferenceSession:
    """Clone ``session`` with the named tensors' batch axis made symbolic.

    The stock GigaAM decoder/joiner graphs pin the leading batch dim to 1.
    We load the original ONNX file (``_model_path``), relabel exactly the
    requested axes to a free symbol, and build a new session with the SAME
    providers and provider options as the stock one, so device placement
    and execution provider are unchanged.
    """
    import onnx

    path = getattr(session, "_model_path", None)
    if not path:
        raise ValueError("session has no _model_path to rebuild from")

    model = onnx.load(path)

    def _relabel(value_infos, axis_by_name) -> None:
        for vi in value_infos:
            axis = axis_by_name.get(vi.name)
            if axis is None:
                continue
            dim = vi.type.tensor_type.shape.dim[axis]
            dim.ClearField("dim_value")
            dim.dim_param = "fast_rnnt_batch"

    _relabel(model.graph.input, input_batch_axis)
    _relabel(model.graph.output, output_batch_axis)

    # Route through make_session_options so these dynamic-batch decoder/joiner
    # sessions inherit runtime.threads_per_worker's intra-op cap (+ no-spin)
    # like every other ORT session the ASR worker builds; OMP_NUM_THREADS alone
    # does not reliably bound ORT's own intra-op pool. It sets ORT_ENABLE_ALL
    # graph optimization, matching the prior hand-rolled options.
    sess_options = make_session_options()

    providers = session.get_providers()
    provider_options = list(session.get_provider_options().values())
    return rt.InferenceSession(
        model.SerializeToString(),
        sess_options,
        providers=providers,
        provider_options=provider_options,
    )


# ====================================================================== #
#  per-topology batched predictor/joiner backends                        #
# ====================================================================== #
class _GigaamBackend:
    """Batched LSTM-predictor backend for ``GigaamV2Rnnt`` / v3 RNN-T."""

    PRED_HIDDEN = 320

    def __init__(self, asr) -> None:
        self._asr = asr
        self._blank_idx = asr._blank_idx
        self._pred_hidden = getattr(asr, "PRED_HIDDEN", self.PRED_HIDDEN)
        # decoder: x[B,1] int64, h.1/c.1 [1,B,320] -> dec[B,1,320], h/c[1,B,320]
        self._decoder = _rebuild_dynamic_batch(
            asr._decoder,
            input_batch_axis={"x": 0, "h.1": 1, "c.1": 1},
            output_batch_axis={"dec": 0, "h": 1, "c": 1},
        )
        # joiner: enc[B,768,1], dec[B,320,1] -> joint[B,1,1,V]
        self._joiner = _rebuild_dynamic_batch(
            asr._joiner,
            input_batch_axis={"enc": 0, "dec": 0},
            output_batch_axis={"joint": 0},
        )

    def initial_state(self, n: int) -> Tuple[npt.NDArray, npt.NDArray]:
        h = np.zeros((1, n, self._pred_hidden), dtype=np.float32)
        c = np.zeros((1, n, self._pred_hidden), dtype=np.float32)
        return h, c

    def run_decoder(
        self, tokens: npt.NDArray, state: Tuple[npt.NDArray, npt.NDArray]
    ) -> Tuple[npt.NDArray, Tuple[npt.NDArray, npt.NDArray]]:
        """Advance the predictor for the selected rows.

        ``tokens`` is int64 ``[k]`` (the previous emitted token, or blank at
        the start), ``state`` is the prior ``(h, c)`` for those rows.
        Returns ``(decoder_out[k,1,320], (h[1,k,320], c[1,k,320]))``.
        """
        h, c = state
        dec, h2, c2 = self._decoder.run(
            ["dec", "h", "c"],
            {"x": tokens[:, None], "h.1": h, "c.1": c},
        )
        return dec, (h2, c2)

    def run_joiner(
        self, enc_frames: npt.NDArray, decoder_out: npt.NDArray
    ) -> npt.NDArray:
        """Joint over a batch. ``enc_frames`` is ``[k, enc_dim]`` float32,
        ``decoder_out`` is ``[k, 1, 320]``.  Returns logits ``[k, V]``."""
        (joint,) = self._joiner.run(
            ["joint"],
            {"enc": enc_frames[:, :, None], "dec": decoder_out.transpose(0, 2, 1)},
        )
        # joint is [k,1,1,V]
        return joint.reshape(joint.shape[0], joint.shape[-1])


class _KaldiBackend:
    """Batched context-predictor backend for ``KaldiTransducer`` / Vosk.

    The predictor is stateless given the last ``CONTEXT_SIZE`` tokens; stock
    caches ``decoder_out`` per context tuple.  We keep that cache but share
    it across the whole batch and run unseen contexts as one batched
    decoder call.  The joiner graph is already dynamic-batch.
    """

    def __init__(self, asr) -> None:
        self._asr = asr
        self._blank_idx = asr._blank_idx
        self._context_size = asr.CONTEXT_SIZE
        self._decoder = asr._decoder
        self._joiner = asr._joiner
        self._cache: dict = {}

    def reset(self) -> None:
        self._cache = {}

    def decoder_out_for(self, contexts: List[tuple]) -> List[npt.NDArray]:
        """Return ``decoder_out`` (``[1, D]``) for each context, batching the
        ONNX decoder over the unique uncached ones."""
        missing = []
        for ctx in contexts:
            if ctx not in self._cache and ctx not in missing:
                missing.append(ctx)
        if missing:
            y = np.asarray(missing, dtype=np.int64)
            (decoder_out,) = self._decoder.run(["decoder_out"], {"y": y})
            for i, ctx in enumerate(missing):
                self._cache[ctx] = decoder_out[i : i + 1]
        return [self._cache[ctx] for ctx in contexts]

    def context_for(self, prev_tokens: List[int]) -> tuple:
        return (-1, self._blank_idx, *prev_tokens)[-self._context_size :]

    def run_joiner(
        self, enc_frames: npt.NDArray, decoder_out: npt.NDArray
    ) -> npt.NDArray:
        """``enc_frames`` ``[k, D]``, ``decoder_out`` ``[k, D]`` -> ``[k, V]``."""
        (logit,) = self._joiner.run(
            ["logit"], {"encoder_out": enc_frames, "decoder_out": decoder_out}
        )
        return logit


# ====================================================================== #
#  batched greedy decode loops                                           #
# ====================================================================== #
def _decoding_gigaam_batched(
    asr,
    backend: _GigaamBackend,
    encoder_out: npt.NDArray,
    encoder_out_lens: npt.NDArray,
    need_logprobs,
) -> Iterator[Tuple[Iterable[int], Iterable[int], Optional[Iterable[float]]]]:
    """Batched equivalent of ``GigaamV2Rnnt`` greedy decode.

    Reproduces the stock loop in ``_AsrWithTransducerDecoding._decoding``
    exactly, in lockstep across the batch:

    * the predictor (decoder ONNX) advances only after a non-blank emit
      (or at the very start), matching stock's ``len(prev_state) == 2``
      cache dance;
    * the joiner runs every step;
    * ``t`` advances when the argmax is blank OR ``emitted == max_tokens``;
    * GigaAM's ``_decode`` always returns ``step = -1``, so the ``step > 0``
      branch is dead and not replicated.
    """
    b = encoder_out.shape[0]
    blank = asr._blank_idx
    max_tokens = asr._max_tokens_per_step

    if asr.use_low_precision:  # mirror stock's TRT length clamp
        encoder_out_lens = np.minimum(encoder_out_lens, encoder_out.shape[1])
    enc_lens = encoder_out_lens.astype(np.int64)

    t = np.zeros(b, dtype=np.int64)
    emitted = np.zeros(b, dtype=np.int64)
    finished = t >= enc_lens
    # predictor state per sequence; need_decoder marks rows whose cached
    # decoder_out is stale (start, or after an emit).
    h, c = backend.initial_state(b)
    decoder_out = np.zeros((b, 1, backend._pred_hidden), dtype=np.float32)
    need_decoder = np.ones(b, dtype=bool)
    last_token = np.full(b, blank, dtype=np.int64)

    tokens: List[List[int]] = [[] for _ in range(b)]
    timestamps: List[List[int]] = [[] for _ in range(b)]
    logprobs: List[List[float]] = [[] for _ in range(b)]

    while not finished.all():
        active = np.flatnonzero(~finished)

        # 1) advance the predictor for active rows that need it.
        rows = active[need_decoder[active]]
        if rows.size:
            dec, (nh, nc) = backend.run_decoder(
                last_token[rows], (h[:, rows], c[:, rows])
            )
            decoder_out[rows] = dec
            h[:, rows] = nh
            c[:, rows] = nc
            need_decoder[rows] = False

        # 2) joiner over all active rows at their current frame.
        enc_frames = encoder_out[active, t[active]]
        logits = backend.run_joiner(enc_frames, decoder_out[active])
        toks = logits.argmax(axis=-1)

        # 3) per-row emit / advance bookkeeping (vectorized, stock order).
        is_blank = toks == blank
        emit = ~is_blank
        for local_i, seq in enumerate(active):
            tok = int(toks[local_i])
            if emit[local_i]:
                tokens[seq].append(tok)
                timestamps[seq].append(int(t[seq]))
                emitted[seq] += 1
                last_token[seq] = tok
                need_decoder[seq] = True
                if need_logprobs:
                    logprobs[seq].append(float(_log_softmax(logits[local_i])[tok]))
            # stock: elif token == blank or emitted == max_tokens -> t += 1
            if is_blank[local_i] or emitted[seq] == max_tokens:
                t[seq] += 1
                emitted[seq] = 0

        finished = t >= enc_lens

    for seq in range(b):
        yield tokens[seq], timestamps[seq], (logprobs[seq] if need_logprobs else None)


def _decoding_kaldi_batched(
    asr,
    backend: _KaldiBackend,
    encoder_out: npt.NDArray,
    encoder_out_lens: npt.NDArray,
    need_logprobs,
) -> Iterator[Tuple[Iterable[int], Iterable[int], Optional[Iterable[float]]]]:
    """Batched equivalent of ``KaldiTransducer`` (Vosk) greedy decode.

    The predictor is stateless given the last ``CONTEXT_SIZE`` tokens, so
    every step we gather each active row's context, fetch ``decoder_out``
    from the shared cache (running unseen contexts as one batched ONNX
    call), and run one batched joiner call.  ``max_tokens_per_step`` is 1
    for Vosk; the general ``elif`` is preserved regardless.
    """
    b = encoder_out.shape[0]
    blank = asr._blank_idx
    max_tokens = asr._max_tokens_per_step
    backend.reset()

    if asr.use_low_precision:
        encoder_out_lens = np.minimum(encoder_out_lens, encoder_out.shape[1])
    enc_lens = encoder_out_lens.astype(np.int64)

    t = np.zeros(b, dtype=np.int64)
    emitted = np.zeros(b, dtype=np.int64)
    finished = t >= enc_lens

    tokens: List[List[int]] = [[] for _ in range(b)]
    timestamps: List[List[int]] = [[] for _ in range(b)]
    logprobs: List[List[float]] = [[] for _ in range(b)]

    while not finished.all():
        active = np.flatnonzero(~finished)
        contexts = [backend.context_for(tokens[seq]) for seq in active]
        decoder_outs = backend.decoder_out_for(contexts)
        decoder_out = np.concatenate(decoder_outs, axis=0)  # [k, D]
        enc_frames = encoder_out[active, t[active]]
        logits = backend.run_joiner(enc_frames, decoder_out)
        toks = logits.argmax(axis=-1)

        is_blank = toks == blank
        emit = ~is_blank
        for local_i, seq in enumerate(active):
            tok = int(toks[local_i])
            if emit[local_i]:
                tokens[seq].append(tok)
                timestamps[seq].append(int(t[seq]))
                emitted[seq] += 1
                if need_logprobs:
                    logprobs[seq].append(float(_log_softmax(logits[local_i])[tok]))
            if is_blank[local_i] or emitted[seq] == max_tokens:
                t[seq] += 1
                emitted[seq] = 0

        finished = t >= enc_lens

    for seq in range(b):
        yield tokens[seq], timestamps[seq], (logprobs[seq] if need_logprobs else None)


# ====================================================================== #
#  patching entry point                                                  #
# ====================================================================== #
def _resolve_asr(model):
    """Return the underlying transducer ``asr`` object of a loaded onnx-asr
    model (``TextResultsAsrAdapter`` / ``with_timestamps`` adapter), or
    ``None`` if this isn't a recognizable transducer."""
    asr = getattr(model, "asr", None)
    if asr is None:
        return None
    # Must be a transducer (RNN-T): has the per-step _decode + greedy state.
    needed = ("_decoding", "_blank_idx", "_max_tokens_per_step", "use_low_precision")
    if not all(hasattr(asr, name) for name in needed):
        return None
    return asr


def _make_backend(asr):
    """Build the batched backend for ``asr``'s topology, or ``None`` if the
    topology is not one we batch (caller then leaves stock decode in place)."""
    cls_names = {c.__name__ for c in type(asr).__mro__}
    # GigaAM RNN-T family (v2/v3, e2e): LSTM predictor, hardcoded batch-1.
    if "GigaamV2Rnnt" in cls_names:
        if not (hasattr(asr, "_decoder") and hasattr(asr, "_joiner")):
            return None, None
        return _GigaamBackend(asr), "gigaam"
    # Kaldi / Vosk transducer: context predictor, already dynamic-batch.
    if "KaldiTransducer" in cls_names:
        if not (
            hasattr(asr, "_decoder")
            and hasattr(asr, "_joiner")
            and hasattr(asr, "CONTEXT_SIZE")
        ):
            return None, None
        return _KaldiBackend(asr), "kaldi"
    return None, None


def patch_model(model, *, strict: bool = False):
    """Patch a loaded onnx-asr transducer model for batched greedy decode.

    Monkeypatches ``asr._decoding`` — the single method that
    ``recognize_batch`` (and thus ``recognize_batch`` in
    ``src/utils/datasets/transcription.py``) routes through — with a
    batched equivalent.  Returns ``model`` unchanged (and unpatched) for any
    non-transducer or unrecognized topology, so it is always safe to call:
    CTC models, NeMo transducers, Whisper, etc. keep stock decode.

    Args:
        model: the object returned by ``onnx_asr.load_model(...)`` (or its
            ``.with_timestamps()`` / ``.with_vad()`` adapter — the same
            ``asr`` object is shared, so patching once covers all adapters).
        strict: if True, re-raise build errors instead of falling back
            (used by tests).  Default False: any failure logs a warning and
            leaves stock decode untouched.

    Returns:
        ``model`` (mutated in place when patched).
    """
    asr = _resolve_asr(model)
    if asr is None:
        return model
    if getattr(asr, "_fast_rnnt_patched", False):
        return model

    try:
        backend, kind = _make_backend(asr)
    except Exception as exc:
        if strict:
            raise
        logger.warning(f"fast_rnnt: backend build failed ({exc}); using stock decode")
        return model

    if backend is None:
        # Recognized transducer base but unbatched topology (e.g. NeMo): leave stock.
        return model

    stock_decoding = asr._decoding

    if kind == "gigaam":
        batched = _decoding_gigaam_batched
    else:
        batched = _decoding_kaldi_batched

    def _fast_decoding(encoder_out, encoder_out_lens, /, **kwargs):
        need_logprobs = kwargs.get("need_logprobs")
        try:
            # Materialize so any failure surfaces here (before yielding) and
            # we can fall back to the stock generator for this batch.
            return iter(
                list(
                    batched(asr, backend, encoder_out, encoder_out_lens, need_logprobs)
                )
            )
        except Exception as exc:
            logger.warning(
                f"fast_rnnt: batched decode failed ({exc}); falling back to stock for this batch"
            )
            return stock_decoding(encoder_out, encoder_out_lens, **kwargs)

    asr._decoding = _fast_decoding
    asr._fast_rnnt_patched = True
    asr._fast_rnnt_kind = kind
    logger.info(
        f"fast_rnnt: patched {type(asr).__name__} ({kind}) for batched greedy decode"
    )
    return model


def is_patched(model) -> bool:
    """True if ``model``'s asr has the batched decode installed."""
    asr = getattr(model, "asr", None)
    return bool(getattr(asr, "_fast_rnnt_patched", False))
