"""ONNX export + optimized-inference benchmark (M7b).

Exports the classifier to ONNX (dynamic batch axis) and times ONNX Runtime against
PyTorch eager. On this Mac, ORT can use the CoreML execution provider (Neural Engine/
GPU); on an NVIDIA box you'd add the TensorRT/CUDA providers and FP16/INT8 for the big
win. FP16 export is supported here for that path; CPU/CoreML FP16 gains are limited.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from gpu_corruptnet.bench import BatchResult


def export_onnx(
    net: torch.nn.Module,
    path: str | Path,
    img_size: int = 224,
    fp16: bool = False,
    opset: int = 17,
) -> Path:
    net = net.eval().cpu()
    dummy = torch.randn(1, 3, img_size, img_size)
    if fp16:
        net, dummy = net.half(), dummy.half()
    path = Path(path)
    torch.onnx.export(
        net,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,  # mature TorchScript exporter (no onnxscript dependency)
    )
    return path


def make_session(path: str | Path, provider: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=[provider])


def benchmark_onnx(
    session,
    img_size: int = 224,
    batch_sizes: tuple[int, ...] = (1, 8, 32),
    warmup: int = 10,
    iters: int = 50,
    fp16: bool = False,
) -> list[BatchResult]:
    name = session.get_inputs()[0].name
    dtype = np.float16 if fp16 else np.float32
    results: list[BatchResult] = []
    for bs in batch_sizes:
        x = np.random.randn(bs, 3, img_size, img_size).astype(dtype)
        for _ in range(warmup):
            session.run(None, {name: x})
        t = np.empty(iters, dtype=np.float64)
        for i in range(iters):
            t0 = time.perf_counter()
            session.run(None, {name: x})
            t[i] = (time.perf_counter() - t0) * 1000.0
        mean_ms = float(t.mean())
        results.append(
            BatchResult(
                batch_size=bs,
                mean_ms=mean_ms,
                p50_ms=float(np.percentile(t, 50)),
                p95_ms=float(np.percentile(t, 95)),
                p99_ms=float(np.percentile(t, 99)),
                throughput_ips=bs / (mean_ms / 1000.0),
            )
        )
    return results
