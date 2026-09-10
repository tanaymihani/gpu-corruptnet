"""Unsupervised anomaly head (M4) — detect *unknown* corruption without labels.

Trained on clean frames only, it flags any frame whose deep features are far from the
clean manifold (a lightweight, image-level PatchCore: kNN distance to a coreset memory
bank of pretrained-backbone features). This addresses the supervised head's blind spot —
corruption types it never saw — and never uses a single corruption label.
"""

from __future__ import annotations

import numpy as np


class NearestNeighborAnomaly:
    """Image-level PatchCore-lite: anomaly score = mean distance to k nearest clean features."""

    def __init__(self, k: int = 3) -> None:
        self.k = k

    def fit(self, feats: np.ndarray, coreset: int = 2000, seed: int = 0) -> NearestNeighborAnomaly:
        from sklearn.neighbors import NearestNeighbors

        feats = np.asarray(feats, dtype=np.float32)
        if len(feats) > coreset:  # random coreset keeps the memory bank small
            idx = np.random.default_rng(seed).choice(len(feats), coreset, replace=False)
            feats = feats[idx]
        self.nn_ = NearestNeighbors(n_neighbors=self.k).fit(feats)
        return self

    def score(self, feats: np.ndarray) -> np.ndarray:
        dist, _ = self.nn_.kneighbors(np.asarray(feats, dtype=np.float32))
        return dist.mean(axis=1)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC with labels 1 = anomaly (corrupted), 0 = clean."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


class DeepFeatureExtractor:
    """Pooled 2048-d features from an ImageNet-pretrained ResNet-50 backbone (no head)."""

    def __init__(self, device: str | None = None) -> None:
        import torch
        from torchvision import models

        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        net.fc = torch.nn.Identity()
        self.net = net.eval().to(device)

    def embed(self, imgs: np.ndarray, img_size: int = 224, batch_size: int = 64) -> np.ndarray:
        import torch
        from PIL import Image

        from gpu_corruptnet.data.corrupt_dataset import IMAGENET_MEAN, IMAGENET_STD

        out = []
        for i in range(0, len(imgs), batch_size):
            xs = []
            for im in imgs[i : i + batch_size]:
                r = np.asarray(Image.fromarray(im).resize((img_size, img_size), Image.BILINEAR))
                r = (r.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
                xs.append(r.transpose(2, 0, 1))
            x = torch.from_numpy(np.stack(xs)).to(self.device)
            with torch.no_grad():
                out.append(self.net(x).float().cpu().numpy())
        return np.concatenate(out)
