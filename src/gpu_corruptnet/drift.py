"""Drift / OOD monitor for deployed inference (M9) — "fixing deployed AI".

When frames arriving in production drift away from the training distribution, model
quality silently degrades. This module flags that: per-feature Population Stability
Index (PSI) for *distribution* drift, and a Mahalanobis OOD score for *per-frame*
novelty. It operates on any feature matrix, so it works on cheap image descriptors
now (model-free) and on model embeddings once the classifier is trained.
"""

from __future__ import annotations

import numpy as np

DESCRIPTOR_NAMES = (
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
    "brightness", "edge_density", "colorfulness",
)

# Standard PSI interpretation thresholds.
PSI_NO_DRIFT = 0.1
PSI_SIGNIFICANT = 0.25


def image_descriptors(imgs: np.ndarray) -> np.ndarray:
    """Cheap model-free descriptors for (N, H, W, 3) uint8 frames -> (N, 9) floats.

    Captures the signals corruption disturbs: per-channel color stats, brightness,
    edge density, and colorfulness.
    """
    x = imgs.astype(np.float32) / 255.0
    mean_c = x.mean(axis=(1, 2))                      # (N, 3)
    std_c = x.std(axis=(1, 2))                        # (N, 3)
    gray = x.mean(axis=3)                             # (N, H, W)
    brightness = gray.mean(axis=(1, 2))
    gx = np.abs(np.diff(gray, axis=2)).mean(axis=(1, 2))
    gy = np.abs(np.diff(gray, axis=1)).mean(axis=(1, 2))
    edge_density = 0.5 * (gx + gy)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = np.sqrt(rg.var(axis=(1, 2)) + yb.var(axis=(1, 2)))
    return np.column_stack([mean_c, std_c, brightness, edge_density, colorfulness])


def population_stability_index(ref: np.ndarray, cur: np.ndarray, edges: np.ndarray) -> float:
    eps = 1e-6
    ref_frac = np.histogram(ref, bins=edges)[0] / max(len(ref), 1) + eps
    cur_frac = np.histogram(cur, bins=edges)[0] / max(len(cur), 1) + eps
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


class DriftMonitor:
    """Fit a reference (training) feature distribution, then score fresh batches."""

    def __init__(self, psi_bins: int = 10, feature_names: tuple[str, ...] | None = None) -> None:
        self.psi_bins = psi_bins
        self.feature_names = feature_names

    def fit(self, ref: np.ndarray) -> DriftMonitor:
        ref = np.asarray(ref, dtype=np.float64)
        self.n_features_ = ref.shape[1]
        if self.feature_names is None:
            self.feature_names = tuple(f"f{i}" for i in range(self.n_features_))

        # Quantile bin edges per feature (open-ended tails so new extremes still bin).
        self.edges_ = []
        for j in range(self.n_features_):
            e = np.unique(np.quantile(ref[:, j], np.linspace(0, 1, self.psi_bins + 1)))
            if len(e) < 2:  # constant feature
                e = np.array([ref[:, j].min() - 1, ref[:, j].max() + 1])
            e[0], e[-1] = -np.inf, np.inf
            self.edges_.append(e)

        self.mean_ = ref.mean(axis=0)
        cov = np.cov(ref, rowvar=False) + 1e-6 * np.eye(self.n_features_)
        self.cov_inv_ = np.linalg.inv(np.atleast_2d(cov))
        self.ref_ = ref
        # OOD threshold: 99th percentile of reference Mahalanobis distances.
        self.ood_threshold_ = float(np.quantile(self.mahalanobis(ref), 0.99))
        return self

    def mahalanobis(self, x: np.ndarray) -> np.ndarray:
        diff = np.asarray(x, dtype=np.float64) - self.mean_
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, self.cov_inv_, diff))

    def ood_flags(self, x: np.ndarray) -> np.ndarray:
        return self.mahalanobis(x) > self.ood_threshold_

    def report(self, cur: np.ndarray) -> dict:
        cur = np.asarray(cur, dtype=np.float64)
        psi = {
            self.feature_names[j]: population_stability_index(
                self.ref_[:, j], cur[:, j], self.edges_[j]
            )
            for j in range(self.n_features_)
        }
        max_psi = max(psi.values())
        if max_psi < PSI_NO_DRIFT:
            status = "no_drift"
        elif max_psi < PSI_SIGNIFICANT:
            status = "moderate_drift"
        else:
            status = "significant_drift"
        return {
            "status": status,
            "drifted": max_psi >= PSI_SIGNIFICANT,
            "max_psi": max_psi,
            "psi_per_feature": psi,
            "ood_rate": float(self.ood_flags(cur).mean()),
            "n_samples": int(len(cur)),
        }
