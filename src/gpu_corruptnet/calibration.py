"""Uncertainty quantification for the corruption classifier (M5).

Three pieces, all operating on raw logits + multi-hot labels from a *held-out
calibration split* (disjoint from test):

1. TemperatureScaler   -- single-parameter logit scaling to fix over/under-confidence.
2. expected_calibration_error -- ECE + reliability bins (per-class positive reliability).
3. ConformalLabelSets  -- split-conformal prediction SETS of artifact types with a
   finite-sample coverage guarantee P(true labels subset of set) >= 1 - alpha.

Note on method choice: MAPIE's APS/RAPS are single-label multiclass procedures; corruption
detection is genuinely multi-label, so we use a full-inclusion split-conformal construction
(threshold on the hardest true label's score) which gives an exact marginal guarantee here.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class TemperatureScaler:
    """Learn a scalar T>0 that minimizes BCE of ``sigmoid(logits / T)`` on a
    calibration set. T>1 tempers overconfidence; T<1 sharpens underconfidence."""

    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> TemperatureScaler:
        import torch

        z = torch.tensor(np.asarray(logits), dtype=torch.float64)
        y = torch.tensor(np.asarray(labels), dtype=torch.float64)
        log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
        bce = torch.nn.BCEWithLogitsLoss()

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = bce(z / torch.exp(log_t), y)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(torch.exp(log_t).item())
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits) / self.temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return sigmoid(self.transform(logits))


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> tuple[float, list[dict]]:
    """Multi-label ECE: each (sample, class) pair is a binary event; measure how well
    the predicted P(present) matches empirical frequency of present. Returns (ece, bins)."""
    p = np.asarray(probs).ravel()
    y = np.asarray(labels).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)

    ece = 0.0
    n = len(p)
    diagram: list[dict] = []
    for b in range(n_bins):
        m = bin_idx == b
        count = int(m.sum())
        if count == 0:
            diagram.append({"confidence": 0.0, "accuracy": 0.0, "count": 0})
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())
        ece += (count / n) * abs(acc - conf)
        diagram.append({"confidence": conf, "accuracy": acc, "count": count})
    return ece, diagram


class ConformalLabelSets:
    """Split-conformal multi-label prediction sets with full-inclusion coverage.

    Calibrated so that, for a fresh frame, the set {classes with prob >= tau} contains
    *all* of its true artifact labels with probability >= 1 - alpha (clean frames, which
    have no labels, are covered vacuously)."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.tau: float = 0.0

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> ConformalLabelSets:
        probs = np.asarray(probs)
        labels = np.asarray(labels)
        # Nonconformity per labeled sample: the hardest true label's probability.
        scores = [
            probs[i, labels[i] > 0.5].min()
            for i in range(len(probs))
            if (labels[i] > 0.5).any()
        ]
        scores = np.sort(np.asarray(scores))
        n = len(scores)
        if n == 0:
            self.tau = 0.0
            return self
        k = int(np.floor(self.alpha * (n + 1)))
        self.tau = 0.0 if k <= 0 else float(scores[k - 1])
        return self

    def predict_set(self, probs: np.ndarray) -> np.ndarray:
        return np.asarray(probs) >= self.tau

    def evaluate(self, probs: np.ndarray, labels: np.ndarray) -> dict:
        probs = np.asarray(probs)
        labels = np.asarray(labels)
        sets = self.predict_set(probs)
        covered, sizes = [], []
        for i in range(len(probs)):
            pos = np.where(labels[i] > 0.5)[0]
            covered.append(bool(np.all(sets[i, pos])) if len(pos) else True)
            sizes.append(int(sets[i].sum()))
        return {
            "alpha": self.alpha,
            "target_coverage": 1 - self.alpha,
            "empirical_coverage": float(np.mean(covered)),
            "avg_set_size": float(np.mean(sizes)),
            "tau": self.tau,
            "n_samples": int(len(probs)),
        }
