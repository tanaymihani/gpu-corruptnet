"""Evaluate the unsupervised anomaly head: AUROC for clean vs. corrupted frames.

    python scripts/anomaly_eval.py

Fits on clean STL-10 frames only (no corruption labels), then scores held-out clean
frames vs. the same frames with random corruptions injected.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gpu_corruptnet.anomaly import DeepFeatureExtractor, NearestNeighborAnomaly, auroc
from gpu_corruptnet.corruptions import apply, available
from gpu_corruptnet.data.clean_sources import load_clean_splits


def main() -> None:
    splits = load_clean_splits(n_seen=3000, n_unseen=200)
    ext = DeepFeatureExtractor()
    print(f"feature extractor on {ext.device}; fitting on clean frames")

    det = NearestNeighborAnomaly(k=3).fit(ext.embed(splits.train[:2000]))

    test_clean = splits.seen_test
    rng = np.random.default_rng(0)

    def corrupt(f: np.ndarray) -> np.ndarray:
        name = rng.choice(available())
        return apply(name, f, severity=int(rng.integers(2, 6)), rng=int(rng.integers(1e9)))

    corrupted = np.stack([corrupt(f) for f in test_clean])

    s_clean = det.score(ext.embed(test_clean))
    s_corr = det.score(ext.embed(corrupted))
    scores = np.concatenate([s_clean, s_corr])
    labels = np.concatenate([np.zeros(len(s_clean)), np.ones(len(s_corr))])
    score = auroc(scores, labels)

    print(f"\nanomaly AUROC (clean vs corrupted) = {score:.3f}  on {len(test_clean)} frames/class")
    Path("runs").mkdir(exist_ok=True)
    Path("runs/anomaly.json").write_text(
        json.dumps({"auroc": score, "n_test": len(test_clean), "k": 3}, indent=2)
    )


if __name__ == "__main__":
    main()
