"""Micro-benchmark for the ORT intra-op thread-cap (thread-pool hygiene).

Measures a small CPU ONNX Runtime session at 1 worker and at 4 parallel
workers, with the intra-op pool UNCAPPED (ORT default = physical cores) vs
CAPPED (runtime.threads_per_worker). Proves two things:

  * 1 worker: capping must NOT meaningfully regress latency (single-worker
    runs already don't oversubscribe).
  * 4 parallel workers: on a CPU shared with other jobs, the uncapped run has
    4 x physical-core intra-op teams fighting for the box, so the cap reduces
    total wall-clock / per-call latency.

CPU-only; never touches a GPU. Synthetic MatMul model so it runs anywhere.

    python -m benchmarking.micro.bench_ort_threads --label check
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import onnxruntime as ort


def _build_model(path: Path, dim: int = 512) -> None:
    """A chain of MatMuls: enough CPU work that intra-op threading matters."""
    from onnx import TensorProto, helper, numpy_helper

    inits, nodes = [], []
    cur = "x"
    for i in range(8):
        w = np.random.RandomState(i).randn(dim, dim).astype(np.float32)
        inits.append(numpy_helper.from_array(w, name=f"w{i}"))
        nodes.append(helper.make_node("MatMul", [cur, f"w{i}"], [f"h{i}"]))
        cur = f"h{i}"
    nodes.append(helper.make_node("Identity", [cur], ["y"]))
    graph = helper.make_graph(
        nodes,
        "matmul_chain",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, dim])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, dim])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path.write_bytes(model.SerializeToString())


def _run_session(args) -> float:
    model_path, intra_op, iters, batch, dim = args
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_op:
        so.intra_op_num_threads = intra_op
        so.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        str(model_path), so, providers=["CPUExecutionProvider"]
    )
    x = np.random.RandomState(0).randn(batch, dim).astype(np.float32)
    sess.run(None, {"x": x})  # warm
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, {"x": x})
    return time.perf_counter() - t0


def _parallel(model_path, intra_op, iters, batch, dim, workers) -> float:
    payload = [(str(model_path), intra_op, iters, batch, dim)] * workers
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_run_session, payload))
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--cap", type=int, default=4, help="intra-op cap to test")
    args = ap.parse_args()

    out_dir = REPO_ROOT / "benchmarking" / "reports" / "micro"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "_ort_threads_model.onnx"
    _build_model(model_path, dim=args.dim)

    cores = os.cpu_count() or 1
    results = {}
    # 1 worker: uncapped vs capped (latency must not regress).
    results["1w_uncapped_s"] = _run_session(
        (model_path, 0, args.iters, args.batch, args.dim)
    )
    results["1w_capped_s"] = _run_session(
        (model_path, args.cap, args.iters, args.batch, args.dim)
    )
    # 4 parallel workers: uncapped vs capped (cap should help under contention).
    results["4w_uncapped_s"] = _parallel(
        model_path, 0, args.iters, args.batch, args.dim, 4
    )
    results["4w_capped_s"] = _parallel(
        model_path, args.cap, args.iters, args.batch, args.dim, 4
    )

    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "cpu_count": cores,
        "iters": args.iters,
        "batch": args.batch,
        "dim": args.dim,
        "cap": args.cap,
        **{k: round(v, 4) for k, v in results.items()},
    }
    out_file = out_dir / "ort_threads.jsonl"
    with out_file.open("a") as f:
        f.write(json.dumps(row) + "\n")

    print(f"cpu_count={cores} iters={args.iters} batch={args.batch} dim={args.dim} cap={args.cap}")
    print(f"  1 worker : uncapped={results['1w_uncapped_s']:.3f}s  "
          f"capped={results['1w_capped_s']:.3f}s  "
          f"(cap/uncap = {results['1w_capped_s']/results['1w_uncapped_s']:.2f}x)")
    print(f"  4 workers: uncapped={results['4w_uncapped_s']:.3f}s  "
          f"capped={results['4w_capped_s']:.3f}s  "
          f"(cap/uncap = {results['4w_capped_s']/results['4w_uncapped_s']:.2f}x)")
    print(f"  -> {out_file}")


if __name__ == "__main__":
    main()
