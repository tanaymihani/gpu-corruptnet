"""Multi-label corruption metrics. Recall is prioritized (catching corruptions)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support


def multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    threshold: float = 0.5,
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Derived binary "corrupted vs clean": corrupted iff any artifact predicted.
    bin_true = (y_true.sum(1) > 0).astype(int)
    bin_pred = (y_pred.sum(1) > 0).astype(int)
    bin_p, bin_r, bin_f, _ = precision_recall_fscore_support(
        bin_true, bin_pred, average="binary", zero_division=0
    )

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_precision": float(p),
        "macro_recall": float(r),
        "binary_f1": float(bin_f),
        "binary_precision": float(bin_p),
        "binary_recall": float(bin_r),
        "per_class_f1": {c: float(v) for c, v in zip(class_names, per_class_f1, strict=True)},
        "n_samples": int(len(y_true)),
    }
