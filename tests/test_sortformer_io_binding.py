"""Sortformer IOBinding vs numpy session path: identical outputs.

The real Sortformer ONNX model is absent on this node, so a tiny synthetic
ONNX model with the SAME input names and the SAME two output names
(``spkcache_fifo_chunk_preds``, ``chunk_pre_encode_embs``) and dynamic shapes
is built with the ``onnx`` package. Both residency paths of
``Sortformer._run_session`` (numpy ``session.run`` vs device IOBinding) are
driven over randomized inputs and must produce bit-identical outputs — this is
the equivalence the knob relies on (the only difference is tensor residency,
not the graph or execution provider).
"""

import os
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from src.preprocess.sortformer_onnx import Sortformer

NUM_SPEAKERS = 4
EMB_DIM = 8  # small stand-in for the real 512 to keep the test fast


def _build_synthetic_model(path: str) -> None:
    """A streaming-shaped model echoing the real session's I/O contract.

    Inputs mirror ``_streaming_update``: chunk/spkcache/fifo embeddings plus
    their int64 length scalars. Outputs reuse the real output names. The math
    is arbitrary but exercises every input so a residency bug would surface.
    """
    chunk = helper.make_tensor_value_info("chunk", TensorProto.FLOAT, [1, "T", EMB_DIM])
    chunk_lengths = helper.make_tensor_value_info("chunk_lengths", TensorProto.INT64, [1])
    spkcache = helper.make_tensor_value_info("spkcache", TensorProto.FLOAT, [1, "S", EMB_DIM])
    spkcache_lengths = helper.make_tensor_value_info("spkcache_lengths", TensorProto.INT64, [1])
    fifo = helper.make_tensor_value_info("fifo", TensorProto.FLOAT, [1, "F", EMB_DIM])
    fifo_lengths = helper.make_tensor_value_info("fifo_lengths", TensorProto.INT64, [1])

    # combined = concat([spkcache, fifo, chunk], axis=1) -> length S+F+T
    preds_out = helper.make_tensor_value_info(
        "spkcache_fifo_chunk_preds", TensorProto.FLOAT, [1, "L", NUM_SPEAKERS]
    )
    embs_out = helper.make_tensor_value_info(
        "chunk_pre_encode_embs", TensorProto.FLOAT, [1, "T", EMB_DIM]
    )

    nodes = [
        helper.make_node("Concat", ["spkcache", "fifo", "chunk"], ["combined"], axis=1),
        # preds = first NUM_SPEAKERS channels of combined, scaled
        helper.make_node(
            "Slice", ["combined", "start0", "endK", "axis2"], ["combined_k"]
        ),
        helper.make_node("Mul", ["combined_k", "half"], ["spkcache_fifo_chunk_preds"]),
        # embs = chunk * 3 + 1
        helper.make_node("Mul", ["chunk", "three"], ["chunk3"]),
        helper.make_node("Add", ["chunk3", "one"], ["chunk_pre_encode_embs"]),
    ]
    initializers = [
        helper.make_tensor("start0", TensorProto.INT64, [1], [0]),
        helper.make_tensor("endK", TensorProto.INT64, [1], [NUM_SPEAKERS]),
        helper.make_tensor("axis2", TensorProto.INT64, [1], [2]),
        helper.make_tensor("half", TensorProto.FLOAT, [1], [0.5]),
        helper.make_tensor("three", TensorProto.FLOAT, [1], [3.0]),
        helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0]),
    ]
    graph = helper.make_graph(
        nodes,
        "synthetic_sortformer",
        [chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths],
        [preds_out, embs_out],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _make_sortformer(model_path, use_io_binding, providers):
    import torch

    diar = Sortformer.__new__(Sortformer)
    # _setup_io_binding reads self.device to pick the IOBinding device id when a
    # CUDA provider is present, so set it (cuda:0 when CUDA is in the provider
    # list, else cpu) — otherwise the CUDA residency path would silently fall
    # back to numpy and the test wouldn't actually exercise it.
    diar.device = torch.device(
        "cuda:0" if "CUDAExecutionProvider" in providers and torch.cuda.is_available() else "cpu"
    )
    diar.session = ort.InferenceSession(model_path, providers=providers)
    diar._setup_io_binding(use_io_binding)
    return diar


def _rand_inputs(rng, t, s, f):
    return {
        "chunk": rng.standard_normal((1, t, EMB_DIM)).astype(np.float32),
        "chunk_lengths": np.array([t], dtype=np.int64),
        "spkcache": rng.standard_normal((1, s, EMB_DIM)).astype(np.float32),
        "spkcache_lengths": np.array([s], dtype=np.int64),
        "fifo": rng.standard_normal((1, f, EMB_DIM)).astype(np.float32),
        "fifo_lengths": np.array([f], dtype=np.int64),
    }


@pytest.fixture(scope="module")
def synthetic_model():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "synthetic_sortformer.onnx")
    _build_synthetic_model(path)
    yield path


_PROVIDER_SETS = [["CPUExecutionProvider"]]
if "CUDAExecutionProvider" in ort.get_available_providers():
    _PROVIDER_SETS.append(["CUDAExecutionProvider", "CPUExecutionProvider"])


@pytest.mark.parametrize("providers", _PROVIDER_SETS)
def test_io_binding_matches_numpy_path(synthetic_model, providers):
    numpy_diar = _make_sortformer(synthetic_model, use_io_binding=False, providers=providers)
    iob_diar = _make_sortformer(synthetic_model, use_io_binding=True, providers=providers)

    # IOBinding must actually be active (not silently fallen back to numpy),
    # else the equivalence below would be vacuous.
    assert iob_diar.use_io_binding is True
    assert iob_diar._io_binding is not None
    if "CUDAExecutionProvider" in providers:
        assert iob_diar._iobinding_device == "cuda"

    rng = np.random.default_rng(123)
    # Mimic the streaming shape signatures the real loop cycles through.
    for t, s, f in [(1, 0, 0), (5, 0, 3), (5, 188, 124), (5, 124, 0), (3, 50, 90)]:
        inputs = _rand_inputs(rng, t, s, f)
        out_np = numpy_diar._run_session(dict(inputs))
        out_io = iob_diar._run_session(dict(inputs))
        assert len(out_np) == len(out_io) == 2
        for a, b in zip(out_np, out_io):
            assert np.array_equal(a, b), f"shape {(t, s, f)} provider {providers}"


def test_io_binding_disabled_uses_session_run(synthetic_model):
    diar = _make_sortformer(
        synthetic_model, use_io_binding=False, providers=["CPUExecutionProvider"]
    )
    assert diar.use_io_binding is False
    assert diar._io_binding is None
    rng = np.random.default_rng(0)
    t, s, f = 4, 10, 5
    out = diar._run_session(_rand_inputs(rng, t, s, f))
    # preds keeps the combined length (S+F+T) and the first NUM_SPEAKERS channels.
    assert out[0].shape == (1, s + f + t, NUM_SPEAKERS)
    assert out[1].shape == (1, t, EMB_DIM)


def test_io_binding_many_random_shapes(synthetic_model):
    numpy_diar = _make_sortformer(
        synthetic_model, use_io_binding=False, providers=["CPUExecutionProvider"]
    )
    iob_diar = _make_sortformer(
        synthetic_model, use_io_binding=True, providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(2024)
    for _ in range(60):
        t = int(rng.integers(1, 8))
        s = int(rng.integers(0, 200))
        f = int(rng.integers(0, 130))
        inputs = _rand_inputs(rng, t, s, f)
        out_np = numpy_diar._run_session(dict(inputs))
        out_io = iob_diar._run_session(dict(inputs))
        for a, b in zip(out_np, out_io):
            assert np.array_equal(a, b)
