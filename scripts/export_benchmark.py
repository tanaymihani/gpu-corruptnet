"""Export to ONNX and compare PyTorch eager vs ONNX Runtime, reporting the speedup.

    python scripts/export_benchmark.py --arch resnet50 --img-size 224
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import onnxruntime as ort
import torch

from gpu_corruptnet.bench import benchmark, model_stats
from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES
from gpu_corruptnet.export import benchmark_onnx, export_onnx, make_session
from gpu_corruptnet.models import build_classifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "efficientnet_b4"])
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out-dir", default="runs")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net = build_classifier(args.arch, num_classes=len(ARTIFACT_CLASSES), pretrained=False)
    print(f"arch={args.arch}  params={model_stats(net)['params_millions']:.1f}M\n")

    onnx_path = out / f"model_{args.arch}.onnx"
    export_onnx(net, onnx_path, img_size=args.img_size)
    print(f"exported {onnx_path}")

    bs = tuple(args.batch_sizes)
    report: dict = {"arch": args.arch, "img_size": args.img_size, "backends": {}}

    # PyTorch eager on CPU (fair same-hardware baseline vs ORT-CPU).
    eager = benchmark(net, torch.device("cpu"), args.img_size, bs, warmup=5, iters=args.iters)
    report["backends"]["torch_eager_cpu"] = [r.as_dict() for r in eager]

    # ONNX Runtime across whatever providers this machine offers.
    available = ort.get_available_providers()
    wanted = [p for p in ("CoreMLExecutionProvider", "CUDAExecutionProvider",
                          "TensorrtExecutionProvider", "CPUExecutionProvider") if p in available]
    ort_results = {}
    for prov in wanted:
        try:
            sess = make_session(onnx_path, prov)
            res = benchmark_onnx(sess, args.img_size, bs, warmup=5, iters=args.iters)
            report["backends"][f"onnx_{prov}"] = [r.as_dict() for r in res]
            ort_results[prov] = res
        except Exception as e:  # noqa: BLE001 - provider may not support all ops
            print(f"  ({prov} unavailable: {e})")

    # Report: single-frame latency + throughput, and speedup vs eager-CPU.
    base = {r.batch_size: r for r in eager}
    b1, blast = bs[0], bs[-1]

    def row(label: str, p50: float, ips: float, sp: str) -> None:
        print(f"{label:>32} {p50:>12.2f} {ips:>12.1f} {sp:>9}")

    print(f"\n{'backend':>32} {'p50@b1(ms)':>12} {'ips@b' + str(blast):>12} {'speedup':>9}")
    row("torch eager (CPU)", base[b1].p50_ms, base[blast].throughput_ips, "1.00x")
    for prov, res in ort_results.items():
        by_bs = {r.batch_size: r for r in res}
        speedup = base[b1].mean_ms / by_bs[b1].mean_ms
        label = f"ONNX {prov.replace('ExecutionProvider', '')}"
        row(label, by_bs[b1].p50_ms, by_bs[blast].throughput_ips, f"{speedup:.2f}x")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out / f"export_bench_{args.arch}_{stamp}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
