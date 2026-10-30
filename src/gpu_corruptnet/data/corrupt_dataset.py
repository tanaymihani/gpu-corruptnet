"""On-the-fly multi-label corruption dataset.

Each item deterministically corrupts a clean frame from a per-(epoch, index) seed, so:
  * storage stays ~zero (nothing is written to disk),
  * the eval set is fully reproducible regardless of shuffling / workers, and
  * training sees a fresh corruption of every frame each epoch (call ``set_epoch``).

Frames can be a uniform (N, H, W, 3) array (STL-10) or a list of variable-size frames or
image paths (real game frames). Artifacts are injected at the frame's native resolution and
the result is resized to the model input afterwards, the way a GPU fault appears in a
rendered frame before any downscaling. ``resize_first=True`` flips that order (artifacts
painted at model resolution) for the native-resolution ablation.

Label vector is multi-hot over ARTIFACT_CLASSES; an all-zero vector means "clean".
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from gpu_corruptnet.corruptions import apply, available
from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CorruptionDataset(Dataset):
    def __init__(
        self,
        clean: np.ndarray | Sequence[np.ndarray | str | Path],
        *,
        base_seed: int,
        img_size: int = 224,
        clean_fraction: float = 0.3,
        max_labels: int = 2,
        severity_range: tuple[int, int] = (1, 5),
        corruptions: list[str] | None = None,
        hflip: bool = False,
        resize_first: bool = False,
        crop_scale: tuple[float, float] | None = None,
    ) -> None:
        if isinstance(clean, np.ndarray) and (
            clean.ndim != 4 or clean.shape[-1] != 3 or clean.dtype != np.uint8
        ):
            raise ValueError("clean must be (N, H, W, 3) uint8")
        self.clean = clean
        self.base_seed = base_seed
        self.img_size = img_size
        self.clean_fraction = clean_fraction
        self.max_labels = max_labels
        self.severity_range = severity_range
        # Only corrupt with injectors that are actually importable in this env.
        pool = corruptions if corruptions is not None else available()
        self.corruptions = [c for c in pool if c in available()]
        self.class_index = {c: i for i, c in enumerate(ARTIFACT_CLASSES)}
        self.hflip = hflip
        self.resize_first = resize_first
        # Random crop (fraction of each side) before corruption: training augmentation for
        # large native-resolution frames. None = use the whole frame.
        self.crop_scale = crop_scale
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Re-draw corruptions for a new epoch. Epoch 0 reproduces the original per-index
        seeds, so eval sets (never re-seeded) are unchanged."""
        self.epoch = epoch

    def _frame(self, idx: int) -> np.ndarray:
        item = self.clean[idx]
        if isinstance(item, (str, Path)):
            from PIL import Image

            with Image.open(item) as im:
                return np.asarray(im.convert("RGB"))
        return np.array(item, dtype=np.uint8, copy=True)

    def _rng(self, idx: int) -> np.random.Generator:
        if self.epoch == 0:
            return np.random.default_rng(self.base_seed * 1_000_003 + idx)
        return np.random.default_rng([self.base_seed, self.epoch, idx])

    def __len__(self) -> int:
        return len(self.clean)

    def _resize(self, img: np.ndarray) -> np.ndarray:
        from PIL import Image

        if img.shape[:2] != (self.img_size, self.img_size):
            img = np.asarray(
                Image.fromarray(img).resize((self.img_size, self.img_size), Image.BILINEAR)
            )
        return img

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = self._rng(idx)
        img = self._frame(idx)
        if self.crop_scale is not None:
            img = _random_crop(img, self.crop_scale, rng)
        if self.resize_first:
            img = self._resize(img)
        label = np.zeros(len(ARTIFACT_CLASSES), dtype=np.float32)

        if self.corruptions and rng.random() >= self.clean_fraction:
            k = int(rng.integers(1, self.max_labels + 1))
            chosen = rng.choice(self.corruptions, size=k, replace=False)
            for name in chosen:
                sev = int(rng.integers(self.severity_range[0], self.severity_range[1] + 1))
                img = apply(name, img, severity=sev, rng=rng)
                label[self.class_index[name]] = 1.0

        img = self._resize(img)
        if self.hflip and rng.random() < 0.5:  # benign, non-artifact augmentation
            img = np.ascontiguousarray(img[:, ::-1])

        x = img.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        return x, torch.from_numpy(label)


def _random_crop(
    img: np.ndarray, scale: tuple[float, float], rng: np.random.Generator
) -> np.ndarray:
    h, w = img.shape[:2]
    f = float(rng.uniform(*scale))
    ch, cw = max(8, int(h * f)), max(8, int(w * f))
    y = int(rng.integers(0, h - ch + 1))
    x = int(rng.integers(0, w - cw + 1))
    return np.ascontiguousarray(img[y : y + ch, x : x + cw])
