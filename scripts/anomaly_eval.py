"""Evaluate the unsupervised anomaly head: AUROC for clean vs. corrupted frames.

    python scripts/anomaly_eval.py                 # quick: 450 frames, mixed corruptions
    python scripts/anomaly_eval.py --per-type      # full seen-test split, AUROC per type

Fits on clean STL-10 frames only (no corruption labels), then scores held-out clean
frames vs. the same frames with corruptions injected. Because the head never sees a
corruption, every type is "left out"; ``--per-type`` reports how well each one is caught.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gpu_corruptnet.anomaly import DeepFeatureExtractor, NearestNeighborAnomaly, auroc
from gpu_corruptnet.corruptions import apply, available
from gpu_corruptnet.data.clean_sources import load_clean_splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", action="store_true")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="runs/anomaly.json")
    args = ap.parse_args()

    if args.per_type:
        splits = load_clean_splits(args.data_root)
    else:
        splits = load_clean_splits(args.data_root, n_seen=3000, n_unseen=200)
    ext = DeepFeatureExtractor()
    print(f"feature extractor on {ext.device}; fitting on clean frames")

    det = NearestNeighborAnomaly(k=3).fit(ext.embed(splits.train[:2000]))

    test_clean = splits.seen_test
    rng = np.random.default_rng(0)
    s_clean = det.score(ext.embed(test_clean))

    def corrupt(f: np.ndarray, name: str | None = None) -> np.ndarray:
        name = name or rng.choice(available())
        return apply(name, f, severity=int(rng.integers(2, 6)), rng=int(rng.integers(1e9)))

    def auroc_vs_clean(frames: np.ndarray) -> float:
        s_corr = det.score(ext.embed(frames))
        scores = np.concatenate([s_clean, s_corr])
        labels = np.concatenate([np.zeros(len(s_clean)), np.ones(len(s_corr))])
        return auroc(scores, labels)

    score = auroc_vs_clean(np.stack([corrupt(f) for f in test_clean]))
    print(f"\nanomaly AUROC (clean vs corrupted) = {score:.3f}  on {len(test_clean)} frames/class")
    record: dict = {"auroc": score, "n_test": len(test_clean), "k": 3}

    if args.per_type:
        record["per_type_auroc"] = {}
        for name in available():
            a = auroc_vs_clean(np.stack([corrupt(f, name) for f in test_clean]))
            record["per_type_auroc"][name] = a
            print(f"  {name:18s} AUROC={a:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
