"""Clean-frame corpus with an honest *content* split.

STL-10 (96x96 real photos) is the substrate. We hold out whole object classes as
"unseen content" so the unseen-content test set measures generalization to frames
whose content the model never trained on — mirroring Glitchify's seen/unseen games
protocol.

Real game frames (``load_real_frames``) are the second substrate: free-licence screenshots
listed in ``realdata/clean_frames.csv``, split by *game title* so held-out titles are unseen
content in the same sense.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# STL-10 class order: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
UNSEEN_CONTENT_CLASSES = (8, 9)  # ship, truck -> never seen during training


@dataclass
class CleanSplits:
    train: np.ndarray
    val: np.ndarray
    cal: np.ndarray  # disjoint calibration split for temperature scaling + conformal
    seen_test: np.ndarray
    unseen_test: np.ndarray

    def summary(self) -> str:
        return (
            f"train={len(self.train)} val={len(self.val)} cal={len(self.cal)} "
            f"seen_test={len(self.seen_test)} unseen_test={len(self.unseen_test)}"
        )


def _stl10_arrays(root: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    # Read the labelled binaries directly when present (same layout torchvision uses), so a
    # cache only needs train/test and not the 2.7 GB unlabeled file torchvision insists on.
    base = os.path.join(root, "stl10_binary")
    x_path, y_path = os.path.join(base, f"{split}_X.bin"), os.path.join(base, f"{split}_y.bin")
    if os.path.exists(x_path) and os.path.exists(y_path):
        imgs = np.fromfile(x_path, dtype=np.uint8).reshape(-1, 3, 96, 96).transpose(0, 3, 2, 1)
        labels = np.fromfile(y_path, dtype=np.uint8).astype(np.int64) - 1
        return np.ascontiguousarray(imgs), labels

    from torchvision.datasets import STL10

    ds = STL10(root=root, split=split, download=True)
    imgs = ds.data.transpose(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C), uint8 RGB
    labels = np.asarray(ds.labels)
    return imgs.astype(np.uint8), labels


def load_clean_splits(
    root: str = "data",
    n_seen: int | None = None,
    n_unseen: int | None = None,
    val_frac: float = 0.15,
    cal_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 1337,
) -> CleanSplits:
    """Build train/val/cal/seen_test/unseen_test arrays of clean uint8 frames.

    ``seed`` only controls the split, so keep it fixed (1337) across training seeds to get
    identical test sets."""
    rng = np.random.default_rng(seed)
    imgs, labels = _stl10_arrays(root, "train")
    imgs2, labels2 = _stl10_arrays(root, "test")
    imgs = np.concatenate([imgs, imgs2])
    labels = np.concatenate([labels, labels2])

    is_unseen = np.isin(labels, UNSEEN_CONTENT_CLASSES)
    seen_imgs = imgs[~is_unseen]
    unseen_imgs = imgs[is_unseen]

    rng.shuffle(seen_imgs)
    rng.shuffle(unseen_imgs)
    if n_seen is not None:
        seen_imgs = seen_imgs[:n_seen]
    if n_unseen is not None:
        unseen_imgs = unseen_imgs[:n_unseen]

    n = len(seen_imgs)
    n_val = int(n * val_frac)
    n_cal = int(n * cal_frac)
    n_test = int(n * test_frac)
    val = seen_imgs[:n_val]
    cal = seen_imgs[n_val : n_val + n_cal]
    seen_test = seen_imgs[n_val + n_cal : n_val + n_cal + n_test]
    train = seen_imgs[n_val + n_cal + n_test :]
    return CleanSplits(
        train=train, val=val, cal=cal, seen_test=seen_test, unseen_test=unseen_imgs
    )


@dataclass
class RealFrames:
    """Paths to real game frames, split by title."""

    train: list[Path]
    val: list[Path]
    cal: list[Path]
    seen_test: list[Path]
    unseen_test: list[Path]
    unseen_titles: tuple[str, ...]

    def summary(self) -> str:
        return (
            f"train={len(self.train)} val={len(self.val)} cal={len(self.cal)} "
            f"seen_test={len(self.seen_test)} unseen_test={len(self.unseen_test)} "
            f"(unseen titles: {', '.join(self.unseen_titles)})"
        )


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_real_frames(
    manifest: str | Path = "realdata/clean_frames.csv",
    root: str | Path = "data/real_frames",
    unseen_titles: tuple[str, ...] | None = None,
    val_frac: float = 0.1,
    cal_frac: float = 0.15,
    test_frac: float = 0.2,
    seed: int = 1337,
) -> RealFrames:
    """Split real frames: whole titles held out as unseen content, the rest shuffled into
    train/val/cal/seen_test. Only rows whose file exists under ``root`` are used."""
    rows = [r for r in read_manifest(manifest) if (Path(root) / r["file"]).exists()]
    if unseen_titles is None:
        unseen_titles = tuple(sorted({r["title"] for r in rows if r.get("split") == "unseen"}))
    rng = np.random.default_rng(seed)
    unseen = [Path(root) / r["file"] for r in rows if r["title"] in unseen_titles]
    seen = [Path(root) / r["file"] for r in rows if r["title"] not in unseen_titles]
    order = rng.permutation(len(seen))
    seen = [seen[i] for i in order]
    n = len(seen)
    n_val, n_cal, n_test = int(n * val_frac), int(n * cal_frac), int(n * test_frac)
    return RealFrames(
        val=seen[:n_val],
        cal=seen[n_val : n_val + n_cal],
        seen_test=seen[n_val + n_cal : n_val + n_cal + n_test],
        train=seen[n_val + n_cal + n_test :],
        unseen_test=sorted(unseen),
        unseen_titles=tuple(unseen_titles),
    )
