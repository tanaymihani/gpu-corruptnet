"""Benchmark inference latency/throughput for the corruption classifier.

Examples:
    python scripts/benchmark.py --arch resnet50
    python scripts/benchmark.py --arch efficientnet_b4 --img-size 224 --iters 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from gpu_corruptnet.bench import benchmark, model_stats
from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES
from gpu_corruptnet.models import build_classifier
from gpu_corruptnet.train import pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "efficientnet_b4"])
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out-dir", default="runs")
    args = ap.parse_args()

    device = pick_device()
    net = build_classifier(args.arch, num_classes=len(ARTIFACT_CLASSES), pretrained=False)
    stats = model_stats(net)
    print(
        f"arch={args.arch}  device={device}  img={args.img_size}  "
        f"params={stats['params_millions']:.1f}M  size={stats['size_mb']:.0f}MB\n"
    )

    results = benchmark(
        net,
        device,
        img_size=args.img_size,
        batch_sizes=tuple(args.batch_sizes),
        warmup=args.warmup,
        iters=args.iters,
    )

    print(f"{'batch':>5} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'throughput':>12}")
    print(f"{'':>5} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'(img/s)':>12}")
    for r in results:
        print(
            f"{r.batch_size:>5} {r.mean_ms:>8.2f} {r.p50_ms:>8.2f} {r.p95_ms:>8.2f} "
            f"{r.p99_ms:>8.2f} {r.throughput_ips:>12.1f}"
        )

    report = {
        "arch": args.arch,
        "device": str(device),
        "img_size": args.img_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "model": stats,
        "results": [r.as_dict() for r in results],
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out / f"bench_{args.arch}_{stamp}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
