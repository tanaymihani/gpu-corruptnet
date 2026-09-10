"""Generate README figures from a trained checkpoint: per-class F1, a reliability
diagram (calibration before/after temperature scaling), and Grad-CAM saliency.

    python scripts/make_report_figures.py model_resnet50_<stamp>.pt

Model-free of any run logs — it re-evaluates the checkpoint on STL-10 test frames
corrupted on the fly, so every number/plot is reproducible from the weights alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from gpu_corruptnet.calibration import TemperatureScaler, expected_calibration_error, sigmoid
from gpu_corruptnet.corruptions import apply
from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES
from gpu_corruptnet.data.clean_sources import load_clean_splits
from gpu_corruptnet.data.corrupt_dataset import IMAGENET_MEAN, IMAGENET_STD, CorruptionDataset
from gpu_corruptnet.metrics import multilabel_metrics
from gpu_corruptnet.models import load_classifier

SEEN, UNSEEN = "#4E79A7", "#F28E2B"  # CVD-safe categorical pair
INK, GRID, REF = "#333333", "#e6e6e6", "#999999"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cccccc", "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect(model, clean, base_seed, device, img_size=224):
    ds = CorruptionDataset(clean, base_seed=base_seed, img_size=img_size)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            logits.append(model(x.to(device)).float().cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def fig_per_class_f1(seen_m, unseen_m, out: Path):
    names = list(ARTIFACT_CLASSES)
    seen = [seen_m["per_class_f1"][n] for n in names]
    unseen = [unseen_m["per_class_f1"][n] for n in names]
    order = np.argsort(seen)  # ascending -> best at top in barh
    names = [names[i] for i in order]
    seen = [seen[i] for i in order]
    unseen = [unseen[i] for i in order]

    y = np.arange(len(names))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(y + h / 2, seen, height=h, color=SEEN, label="seen content")
    ax.barh(y - h / 2, unseen, height=h, color=UNSEEN, label="unseen content")
    for yi, v in zip(y + h / 2, seen, strict=True):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8, color=INK)
    for yi, v in zip(y - h / 2, unseen, strict=True):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1")
    ax.set_title("Per-class F1 — seen vs. unseen content", loc="left", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_reliability(logits, labels, temperature, out: Path):
    ece_b, diag_b = expected_calibration_error(sigmoid(logits), labels, n_bins=12)
    ece_a, diag_a = expected_calibration_error(sigmoid(logits / temperature), labels, n_bins=12)

    def xy(diag):
        pts = [(d["confidence"], d["accuracy"]) for d in diag if d["count"] > 0]
        return np.array(pts).T if pts else (np.array([]), np.array([]))

    bx, by = xy(diag_b)
    ax_, ay = xy(diag_a)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], "--", color=REF, lw=1.5, label="perfect")
    ax.plot(bx, by, "o-", color=UNSEEN, lw=2, ms=6, label=f"before (ECE {ece_b * 100:.2f}%)")
    ax.plot(ax_, ay, "o-", color=SEEN, lw=2, ms=6,
            label=f"after T={temperature:.2f} (ECE {ece_a * 100:.2f}%)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("Reliability — temperature scaling", loc="left", fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _preprocess(frame224: np.ndarray) -> torch.Tensor:
    x = frame224.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous()


def grad_cam(model, x, cls, target_layer):
    store = {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0]))
    model.zero_grad()
    out = model(x)
    out[0, cls].backward()
    a, g = store["a"][0], store["g"][0]
    cam = torch.relu((g.mean(dim=(1, 2))[:, None, None] * a).sum(0))
    cam = (cam / (cam.max() + 1e-8)).detach().cpu().numpy()
    h1.remove()
    h2.remove()
    return cam


def fig_gradcam(model, frame, device, out: Path):
    layer = model.layer4
    shown = ["parallel_lines", "morse_code", "triangulation", "screen_tearing"]
    fig, axes = plt.subplots(2, len(shown), figsize=(3 * len(shown), 6.2))
    for j, name in enumerate(shown):
        corrupted = apply(name, frame, severity=4, rng=0)
        disp = np.asarray(Image.fromarray(corrupted).resize((224, 224), Image.BILINEAR))
        x = _preprocess(disp).to(device)
        cls = ARTIFACT_CLASSES.index(name)
        cam = grad_cam(model, x, cls, layer)
        cam_img = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224)))
        axes[0, j].imshow(disp)
        axes[0, j].set_title(f"injected: {name}", fontsize=9)
        axes[1, j].imshow(disp)
        axes[1, j].imshow(cam_img, cmap="jet", alpha=0.5)
        for r in (0, 1):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("frame", fontsize=9)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=9)
    fig.suptitle("Grad-CAM — which region triggered each corruption flag", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out-dir", default="assets")
    args = ap.parse_args()

    device = pick_device()
    model, _ = load_classifier(args.ckpt, map_location=str(device))
    model.to(device)
    print(f"loaded {args.ckpt} on {device}")

    splits = load_clean_splits(n_seen=3000, n_unseen=900)
    print("eval frames:", splits.summary())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sl, syl = collect(model, splits.seen_test, 3, device)
    ul, uyl = collect(model, splits.unseen_test, 4, device)
    cl, cyl = collect(model, splits.cal, 5, device)

    seen_m = multilabel_metrics(syl, sigmoid(sl), list(ARTIFACT_CLASSES))
    unseen_m = multilabel_metrics(uyl, sigmoid(ul), list(ARTIFACT_CLASSES))
    print(f"seen macroF1={seen_m['macro_f1']:.3f}  unseen macroF1={unseen_m['macro_f1']:.3f}")

    fig_per_class_f1(seen_m, unseen_m, out / "per_class_f1.png")
    T = TemperatureScaler().fit(cl, cyl).temperature
    fig_reliability(sl, syl, T, out / "reliability.png")
    fig_gradcam(model, splits.seen_test[0], device, out / "gradcam.png")
    print(f"wrote {out}/per_class_f1.png, reliability.png, gradcam.png")


if __name__ == "__main__":
    main()
