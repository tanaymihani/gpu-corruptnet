"""Single-frame latency across backends in ONE run on ONE machine, with a logit-parity check.

    python scripts/latency_parity.py model.pt --out results/latency_parity.json

Backends: PyTorch eager on CPU and MPS, ONNX Runtime on CPU and through the CoreML
execution provider (whichever of these the machine has). Every backend scores the same
real frames; parity = max |logit difference| vs PyTorch-CPU and whether the predicted
label sets agree. Timing: warmup, then per-call wall time (synchronised on MPS), batch 1.
The machine's load average is recorded, because a shared machine inflates every row.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from gpu_corruptnet.data.clean_sources import read_manifest
from gpu_corruptnet.export import export_onnx
from gpu_corruptnet.models import load_classifier


def frames(n: int, size: int) -> np.ndarray:
    from PIL import Image

    from gpu_corruptnet.data.corrupt_dataset import IMAGENET_MEAN, IMAGENET_STD

    rows = [r for r in read_manifest("realdata/clean_frames.csv")
            if Path("data/real_frames", r["file"]).exists()][:n]
    out = []
    for r in rows:
        with Image.open(Path("data/real_frames", r["file"])) as im:
            im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        x = (np.asarray(im, np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        out.append(x.transpose(2, 0, 1))
    return np.stack(out).astype(np.float32)


def timeit(fn, x: np.ndarray, warmup: int, iters: int) -> dict:
    for i in range(warmup):
        fn(x[i % len(x)][None])
    t = []
    for i in range(iters):
        t0 = time.perf_counter()
        fn(x[i % len(x)][None])
        t.append((time.perf_counter() - t0) * 1000)
    t = np.asarray(t)
    return {"p50_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
            "mean_ms": float(t.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="results/latency_parity.json")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    net, meta = load_classifier(args.ckpt)
    size = int(meta["img_size"])
    x = frames(args.n, size)
    load_before = os.getloadavg()

    rows: dict = {}
    with torch.no_grad():
        ref = net(torch.from_numpy(x)).numpy()

    def record(name: str, fn, logits: np.ndarray) -> None:
        rows[name] = {
            **timeit(fn, x, args.warmup, args.iters),
            "max_abs_logit_diff": float(np.abs(logits - ref).max()),
            "label_set_agreement": float(np.mean(np.all((logits > 0) == (ref > 0), axis=1))),
        }
        print(name, json.dumps({k: round(v, 4) for k, v in rows[name].items()}), flush=True)

    with torch.no_grad():
        record("pytorch_cpu", lambda b: net(torch.from_numpy(b)), ref)
        if torch.backends.mps.is_available():
            m = net.to("mps")

            def mps(b):
                y = m(torch.from_numpy(b).to("mps"))
                torch.mps.synchronize()
                return y

            record("pytorch_mps", mps, np.concatenate([mps(x[i:i + 8]).cpu().numpy()
                                                       for i in range(0, len(x), 8)]))
            net = net.to("cpu")

    import onnxruntime as ort

    with tempfile.TemporaryDirectory() as d:
        path = export_onnx(net, Path(d) / "m.onnx", img_size=size)
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        for name, prov in (("onnxruntime_cpu", "CPUExecutionProvider"),
                           ("onnxruntime_coreml", "CoreMLExecutionProvider")):
            if prov not in ort.get_available_providers():
                continue
            sess = ort.InferenceSession(str(path), so, providers=[prov])
            run = lambda b, s=sess: s.run(None, {"input": b})[0]  # noqa: E731
            record(name, run, np.concatenate([run(x[i:i + 1]) for i in range(len(x))]))

    report = {
        "checkpoint": args.ckpt, "batch_size": 1, "input": f"{size}x{size} real game frames",
        "machine": platform.platform(), "processor": platform.processor(),
        "torch": torch.__version__, "onnxruntime": ort.__version__, "threads": args.threads,
        "load_average_before_after": [load_before, os.getloadavg()],
        "backends": rows,
        "note": "One run, one machine; parity vs PyTorch-CPU logits on the same frames.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
