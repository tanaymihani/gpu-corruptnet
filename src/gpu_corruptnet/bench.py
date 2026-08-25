"""Latency / throughput benchmark harness (M7).

Measures inference latency correctly: mandatory warmup, then device-appropriate
synchronized timing (CUDA events on GPU; synchronize + perf_counter on MPS/CPU),
reported as p50/p95/p99 and throughput (images/sec) across a batch-size sweep.

Note: throughput matters to a GPU company. Latency depends only on the *architecture*
and input size, not trained weights, so these numbers are meaningful pre-training.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass
class BatchResult:
    batch_size: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_ips: float  # images / second

    def as_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "throughput_ips": self.throughput_ips,
        }


@torch.no_grad()
def _time_batch(
    net: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
) -> np.ndarray:
    net.eval()
    for _ in range(warmup):  # clears context init / stabilizes clocks
        net(x)
    _sync(device)

    times = np.empty(iters, dtype=np.float64)
    if device.type == "cuda":
        for i in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            net(x)
            end.record()
            torch.cuda.synchronize()
            times[i] = start.elapsed_time(end)  # ms
    else:
        for i in range(iters):
            _sync(device)
            t0 = time.perf_counter()
            net(x)
            _sync(device)
            times[i] = (time.perf_counter() - t0) * 1000.0
    return times


def benchmark(
    net: torch.nn.Module,
    device: torch.device,
    img_size: int = 224,
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    warmup: int = 20,
    iters: int = 100,
) -> list[BatchResult]:
    net = net.to(device).eval()
    results: list[BatchResult] = []
    for bs in batch_sizes:
        x = torch.randn(bs, 3, img_size, img_size, device=device)
        t = _time_batch(net, x, device, warmup, iters)
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


def model_stats(net: torch.nn.Module) -> dict:
    params = sum(p.numel() for p in net.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in net.parameters()) / 1e6
    return {"params_millions": params / 1e6, "size_mb": size_mb}
