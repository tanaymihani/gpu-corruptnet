"""Grad-CAM on real game frames: which region made the model flag each injected artifact.

    python scripts/make_gradcam_real.py model.pt --out assets/gradcam_real.png

Uses free-licence frames from realdata/clean_frames.csv (credits in the figure), injects
artifacts at native resolution, resizes to the model input, and overlays the Grad-CAM of the
injected class from the last ResNet block. The title of each panel is the model's calibrated
probability for that class (temperature from the calibration JSON).
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from make_report_figures import _preprocess, grad_cam
from PIL import Image

from gpu_corruptnet.calibration import sigmoid
from gpu_corruptnet.corruptions import apply
from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES
from gpu_corruptnet.data.clean_sources import read_manifest
from gpu_corruptnet.models import load_classifier

PANELS = [("SuperTuxKart", "discoloration"), ("0 A.D.", "triangulation"),
          ("FlightGear", "parallel_lines"), ("Xonotic", "morse_code")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--calibration", default="results/calibration_real_native.json")
    ap.add_argument("--out", default="assets/gradcam_real.png")
    args = ap.parse_args()

    net, _ = load_classifier(args.ckpt)
    t = json.loads(Path(args.calibration).read_text()).get("temperature", 1.0)
    rows = read_manifest("realdata/clean_frames.csv")
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(2, len(PANELS), figsize=(2.9 * len(PANELS), 6.0))
    credits = []
    for j, (title, name) in enumerate(PANELS):
        cands = [r for r in rows
                 if r["title"] == title and Path("data/real_frames", r["file"]).exists()]
        row = cands[len(cands) // 3]
        frame = np.asarray(Image.open(Path("data/real_frames", row["file"])).convert("RGB"))
        corrupted = apply(name, frame, severity=4, rng=rng)
        disp = np.asarray(Image.fromarray(corrupted).resize((224, 224), Image.BILINEAR))
        x = _preprocess(disp)
        with torch.no_grad():
            p = float(sigmoid(net(x).numpy()[0] / t)[ARTIFACT_CLASSES.index(name)])
        cam = grad_cam(net, x, ARTIFACT_CLASSES.index(name), net.layer4)
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        cam = np.asarray(cam_img, dtype=np.float32) / 255.0
        overlay = disp * (0.25 + 0.75 * cam[..., None])  # dim what the model did not look at
        axes[0, j].imshow(disp)
        axes[0, j].set_title(f"{title}: {name.replace('_', ' ')}", fontsize=9, loc="left")
        axes[1, j].imshow(overlay.astype(np.uint8))
        axes[1, j].set_title(f"Grad-CAM, p({name.replace('_', ' ')}) = {p:.2f}", fontsize=9,
                             loc="left")
        for r in (0, 1):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
        credits.append(f"{title} ({row['license']}, {html.unescape(row['artist'])[:30]})")
    fig.suptitle("Where the real-frame model looks (Grad-CAM; dimmed = ignored), artifacts "
                 "injected into free-licence game frames", x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.005, "Frames via Wikimedia Commons: " + "; ".join(credits), fontsize=7,
             color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(args.out, dpi=140, facecolor="#fcfcfb")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
