"""Apply M5 calibration + conformal prediction to a saved preds_*.npz from training.

Fits temperature scaling and the conformal threshold on the (disjoint) calibration
split, then reports ECE before/after and conformal coverage/set-size on the test splits.

Usage:
    python scripts/calibrate.py runs/preds_resnet50_YYYYMMDD-HHMMSS.npz [--alpha 0.1]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gpu_corruptnet.calibration import (
    ConformalLabelSets,
    TemperatureScaler,
    expected_calibration_error,
    sigmoid,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", type=Path, help="runs/preds_*.npz written by training")
    ap.add_argument("--alpha", type=float, default=0.1, help="miscoverage; target = 1-alpha")
    ap.add_argument("--bins", type=int, default=15)
    args = ap.parse_args()

    d = np.load(args.preds)
    cal_logits, cal_labels = d["cal_logits"], d["cal_labels"]

    # Fit temperature + conformal threshold on the calibration split only.
    ts = TemperatureScaler().fit(cal_logits, cal_labels)
    conf = ConformalLabelSets(alpha=args.alpha).fit(ts.predict_proba(cal_logits), cal_labels)

    report = {
        "preds": str(args.preds),
        "temperature": ts.temperature,
        "alpha": args.alpha,
        "splits": {},
    }
    print(f"temperature T = {ts.temperature:.3f}  (T>1 tempers overconfidence)\n")
    for split in ("seen_test", "unseen_test"):
        if f"{split}_logits" not in d:
            continue
        logits, labels = d[f"{split}_logits"], d[f"{split}_labels"]
        ece_before, _ = expected_calibration_error(sigmoid(logits), labels, args.bins)
        probs = ts.predict_proba(logits)
        ece_after, _ = expected_calibration_error(probs, labels, args.bins)
        cov = conf.evaluate(probs, labels)
        report["splits"][split] = {
            "ece_before": ece_before,
            "ece_after": ece_after,
            "conformal": cov,
        }
        print(
            f"{split:12s}  ECE {ece_before * 100:5.2f}% -> {ece_after * 100:5.2f}%   "
            f"conformal coverage={cov['empirical_coverage']:.3f} "
            f"(target {cov['target_coverage']:.2f})  avg_set={cov['avg_set_size']:.2f}"
        )

    out = args.preds.parent / (args.preds.stem.replace("preds", "calibration") + ".json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
