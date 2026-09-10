"""Export a trained checkpoint to TorchScript for the C++ (libtorch) inference path.

    python scripts/export_torchscript.py model_resnet50_<stamp>.pt --out cpp/model_ts.pt
"""

from __future__ import annotations

import argparse

import torch

from gpu_corruptnet.models import load_classifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="cpp/model_ts.pt")
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    model, _ = load_classifier(args.ckpt, map_location="cpu")
    model.eval()
    example = torch.randn(1, 3, args.img_size, args.img_size)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    traced.save(args.out)
    print(f"wrote {args.out}  (TorchScript, load in C++ with torch::jit::load)")


if __name__ == "__main__":
    main()
