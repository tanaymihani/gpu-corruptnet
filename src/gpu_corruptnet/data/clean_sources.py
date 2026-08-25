"""Clean-frame corpus with an honest *content* split.

STL-10 (96x96 real photos) is the substrate. We hold out whole object classes as
"unseen content" so the unseen-content test set measures generalization to frames
whose content the model never trained on — mirroring Glitchify's seen/unseen games
protocol. Swap in real game frames later without touching downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# STL-10 class order: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
UNSEEN_CONTENT_CLASSES = (8, 9)  # ship, truck -> never seen during training


@dataclass
class CleanSplits:
    train: np.ndarray
    val: np.ndarray
    seen_test: np.ndarray
    unseen_test: np.ndarray

    def summary(self) -> str:
        return (
            f"train={len(self.train)} val={len(self.val)} "
            f"seen_test={len(self.seen_test)} unseen_test={len(self.unseen_test)}"
        )


def _stl10_arrays(root: str, split: str) -> tuple[np.ndarray, np.ndarray]:
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
    test_frac: float = 0.15,
    seed: int = 1337,
) -> CleanSplits:
    """Build train/val/seen_test/unseen_test arrays of clean uint8 frames."""
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
    n_test = int(n * test_frac)
    val = seen_imgs[:n_val]
    seen_test = seen_imgs[n_val : n_val + n_test]
    train = seen_imgs[n_val + n_test :]
    return CleanSplits(train=train, val=val, seen_test=seen_test, unseen_test=unseen_imgs)
