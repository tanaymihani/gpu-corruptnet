"""Evaluation helpers for the results write-up: bootstrap CIs, classwise calibration,
Mondrian (per-class) conformal sets, and embedding OOD scores.

All functions take raw logits / probabilities + multi-hot labels as produced by
``train.collect_logits`` (preds_*.npz), so they apply to every model and baseline.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from gpu_corruptnet.calibration import sigmoid


def macro_f1(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> float:
    return float(f1_score(y, p >= thr, average="macro", zero_division=0))


def binary_f1(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> float:
    return float(f1_score(y.sum(1) > 0, (p >= thr).any(1), zero_division=0))


def bootstrap_ci(
    metric, y: np.ndarray, p: np.ndarray, n_boot: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    """(point estimate, lo, hi) for ``metric(y, p)`` by resampling frames with replacement."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idxs = (rng.integers(0, n, n) for _ in range(n_boot))
    stats = np.array([metric(y[idx], p[idx]) for idx in idxs])
    a = (1 - level) / 2
    return float(metric(y, p)), float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def per_class_f1_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """(C, 3) array of per-class F1 with bootstrap 95% CIs (resampling frames)."""
    rng = np.random.default_rng(seed)
    n, c = y.shape
    point = f1_score(y, p >= 0.5, average=None, zero_division=0)
    boots = np.empty((n_boot, c))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = f1_score(y[idx], p[idx] >= 0.5, average=None, zero_division=0)
    return np.column_stack([point, np.quantile(boots, 0.025, 0), np.quantile(boots, 0.975, 0)])


def _ece_1d(p: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def classwise_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> tuple[float, list]:
    """Mean over classes of each class's own binary ECE (not pooled over all pairs, where the
    ~90% easy negatives make every model look calibrated)."""
    per = [_ece_1d(probs[:, c], labels[:, c], n_bins) for c in range(labels.shape[1])]
    return float(np.mean(per)), per


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


class MondrianConformal:
    """Class-conditional split conformal: one threshold per class, fit on calibration frames
    where that class is present, so P(c in set | c present) >= 1 - alpha for every class."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.tau: np.ndarray | None = None

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> MondrianConformal:
        taus = []
        for c in range(labels.shape[1]):
            s = np.sort(probs[labels[:, c] > 0.5, c])
            k = int(np.floor(self.alpha * (len(s) + 1)))
            taus.append(0.0 if k <= 0 or len(s) == 0 else float(s[k - 1]))
        self.tau = np.asarray(taus)
        return self

    def predict_set(self, probs: np.ndarray) -> np.ndarray:
        return probs >= self.tau[None, :]


def set_coverage(sets: np.ndarray, labels: np.ndarray) -> dict:
    """Full-inclusion coverage split by frame type, plus per-class (conditional) coverage."""
    pos = labels > 0.5
    corrupted = pos.any(1)
    full = np.all(sets | ~pos, axis=1)
    per_class = [float(sets[pos[:, c], c].mean()) if pos[:, c].any() else float("nan")
                 for c in range(labels.shape[1])]
    return {
        "coverage_all_frames": float(full.mean()),
        "coverage_corrupted_frames": float(full[corrupted].mean()),
        "per_class_coverage": per_class,
        "min_class_coverage": float(np.nanmin(per_class)),
        "avg_set_size_corrupted": float(sets[corrupted].sum(1).mean()),
        "avg_set_size_clean": float(sets[~corrupted].sum(1).mean()) if (~corrupted).any() else 0.0,
        "n_corrupted": int(corrupted.sum()),
    }


class EmbeddingOOD:
    """Mahalanobis distance in the top-k PCA subspace of reference embeddings."""

    def __init__(self, k: int = 64) -> None:
        self.k = k

    def fit(self, ref: np.ndarray) -> EmbeddingOOD:
        ref = np.asarray(ref, np.float64)
        self.mean = ref.mean(0)
        _, s, vt = np.linalg.svd(ref - self.mean, full_matrices=False)
        self.comps = vt[: self.k]
        self.var = (s[: self.k] ** 2) / (len(ref) - 1)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        p = (np.asarray(x, np.float64) - self.mean) @ self.comps.T
        return np.sqrt(((p**2) / self.var).sum(1))

    @classmethod
    def from_reference_json(cls, path: str) -> tuple[EmbeddingOOD, float]:
        """Load the reference written by scripts/export_web.py; returns (scorer, threshold)."""
        import base64
        import json

        r = json.loads(open(path).read())
        f32 = lambda key: np.frombuffer(base64.b64decode(r[key]), np.float32)  # noqa: E731
        det = cls(k=r["k"])
        det.mean = f32("mean_b64").astype(np.float64)
        det.comps = f32("components_b64").astype(np.float64).reshape(r["k"], r["dim"])
        det.var = f32("variance_b64").astype(np.float64)
        return det, float(r["threshold"])


def auroc(neg_scores: np.ndarray, pos_scores: np.ndarray) -> float:
    y = np.r_[np.zeros(len(neg_scores)), np.ones(len(pos_scores))]
    return float(roc_auc_score(y, np.r_[neg_scores, pos_scores]))


def probs_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return sigmoid(np.asarray(logits) / temperature)
