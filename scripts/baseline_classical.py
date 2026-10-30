"""Glitchify-style classical baseline: hand-crafted DFT + HOG + colour features with a
one-vs-rest logistic regression, on exactly the same corrupted splits as the deep models.

    python scripts/baseline_classical.py --out-dir runs

Writes metrics_classical_<stamp>.json and preds_classical_<stamp>.npz (logits = log-odds of
the predicted probability, so calibrate.py and the uncertainty analysis apply unchanged).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES
from gpu_corruptnet.data.clean_sources import load_clean_splits
from gpu_corruptnet.data.corrupt_dataset import IMAGENET_MEAN, IMAGENET_STD, CorruptionDataset
from gpu_corruptnet.metrics import multilabel_metrics

SIZE = 128


def hog(gray: np.ndarray, cell: int = 16, bins: int = 9) -> np.ndarray:
    """Dalal-Triggs HOG: unsigned 9-bin cell histograms, 2x2-cell blocks, L2-Hys norm.
    (OpenCV 5 dropped cv2.HOGDescriptor, so this is plain NumPy.)"""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    b = np.minimum((ang / (180.0 / bins)).astype(int), bins - 1)
    nc = gray.shape[0] // cell
    hist = np.zeros((nc, nc, bins), np.float32)
    cy, cx = np.indices(gray.shape) // cell
    np.add.at(hist, (cy.ravel(), cx.ravel(), b.ravel()), mag.ravel())
    blocks = []
    for i in range(nc - 1):
        for j in range(nc - 1):
            v = hist[i : i + 2, j : j + 2].ravel()
            v = np.minimum(v / (np.linalg.norm(v) + 1e-6), 0.2)
            blocks.append(v / (np.linalg.norm(v) + 1e-6))
    return np.concatenate(blocks)


def _radial_profile(mag: np.ndarray, n_bins: int = 32) -> np.ndarray:
    h, w = mag.shape
    yy, xx = np.indices((h, w))
    r = np.hypot(yy - h / 2, xx - w / 2)
    bins = np.minimum((r / r.max() * n_bins).astype(int), n_bins - 1)
    return np.bincount(bins.ravel(), mag.ravel(), n_bins) / np.maximum(
        np.bincount(bins.ravel(), minlength=n_bins), 1
    )


def features(img: np.ndarray) -> np.ndarray:
    """(SIZE, SIZE, 3) uint8 -> 1D feature vector."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hog_f = hog(gray)
    spec = np.fft.fftshift(np.abs(np.fft.fft2(gray.astype(np.float32))))
    logmag = np.log1p(spec)
    c = SIZE // 2
    dft = np.concatenate(
        [
            _radial_profile(logmag),
            logmag[c, :].reshape(16, -1).mean(1),  # horizontal-frequency axis: vertical lines
            logmag[:, c].reshape(16, -1).mean(1),  # vertical-frequency axis: horizontal lines
        ]
    )
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = np.histogram(hsv[..., 0], bins=18, range=(0, 180))[0] / gray.size
    sat = np.histogram(hsv[..., 1], bins=8, range=(0, 256))[0] / gray.size
    stats = np.concatenate([img.reshape(-1, 3).mean(0), img.reshape(-1, 3).std(0)]) / 255.0
    rows = np.abs(np.diff(gray.astype(np.float32), axis=0)).mean(1)  # row-to-row jumps (tearing)
    edges = [rows.max() / (rows.mean() + 1e-6), rows.std() / (rows.mean() + 1e-6)]
    return np.concatenate([hog_f, dft, hue, sat, stats, edges]).astype(np.float32)


def featurize(ds: CorruptionDataset, epochs: tuple[int, ...] = (0,)) -> tuple[np.ndarray, ...]:
    xs, ys = [], []
    for ep in epochs:
        ds.set_epoch(ep)
        for i in range(len(ds)):
            x, y = ds[i]
            img = (x.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN) * 255.0
            xs.append(features(np.clip(img, 0, 255).astype(np.uint8)))
            ys.append(y.numpy())
    return np.stack(xs), np.stack(ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--n-seen", type=int, default=None)
    ap.add_argument("--n-unseen", type=int, default=None)
    ap.add_argument("--train-epochs", type=int, default=2, help="corruption draws per frame")
    ap.add_argument("--C", type=float, default=0.5)
    args = ap.parse_args()

    t0 = time.time()
    splits = load_clean_splits(args.data_root, args.n_seen, args.n_unseen, seed=1337)
    seeds = {"train": 1, "cal": 5, "seen_test": 3, "unseen_test": 4}
    data = {}
    for name, seed in seeds.items():
        ds = CorruptionDataset(getattr(splits, name), base_seed=seed, img_size=SIZE)
        eps = tuple(range(1, args.train_epochs + 1)) if name == "train" else (0,)
        data[name] = featurize(ds, eps)
        print(f"{name}: {data[name][0].shape}  ({time.time() - t0:.0f}s)", flush=True)

    clf = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(LogisticRegression(C=args.C, max_iter=3000), n_jobs=-1),
    )
    clf.fit(*data["train"])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results = {"config": vars(args), "features": "HOG + DFT radial/axis profiles + colour"}
    arrays = {}
    for name in ("cal", "seen_test", "unseen_test"):
        x, y = data[name]
        p = np.clip(clf.predict_proba(x), 1e-6, 1 - 1e-6)
        arrays[f"{name}_logits"] = np.log(p / (1 - p)).astype(np.float32)
        arrays[f"{name}_labels"] = y
        if name != "cal":
            results[name] = multilabel_metrics(y, p, list(ARTIFACT_CLASSES))
            print(f"{name:12s} macroF1={results[name]['macro_f1']:.3f}")
    results["minutes"] = (time.time() - t0) / 60
    (out / f"metrics_classical_{stamp}.json").write_text(json.dumps(results, indent=2))
    np.savez(out / f"preds_classical_{stamp}.npz", **arrays)


if __name__ == "__main__":
    main()
