"""Re-evaluate a trained checkpoint on the full cal / seen / unseen splits and save raw
logits + penultimate embeddings (the same preds_*.npz layout training writes).

    python scripts/eval_checkpoint.py model.pt --tag deployed --out-dir runs
    python scripts/eval_checkpoint.py model.pt --substrate real --tag deployed-real

Eval sets are deterministic (fixed split seed and per-index corruption seeds), so this
reproduces the test sets the checkpoint was scored on when it was trained.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES
from gpu_corruptnet.metrics import multilabel_metrics
from gpu_corruptnet.models import load_classifier
from gpu_corruptnet.train import TrainConfig, build_splits, collect_logits, pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--substrate", default="stl10", choices=["stl10", "real"])
    ap.add_argument("--resize-first", action="store_true")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--n-ref", type=int, default=3000, help="train frames for OOD reference")
    args = ap.parse_args()

    device = pick_device()
    net, ckpt = load_classifier(args.ckpt)
    net = net.to(device)
    cfg = TrainConfig(
        img_size=ckpt["img_size"], substrate=args.substrate, data_root=args.data_root,
        resize_first=args.resize_first,
    )
    ds, summary = build_splits(cfg)
    print(f"device={device}  {summary}")

    def run(d) -> tuple[np.ndarray, ...]:
        dl = DataLoader(d, batch_size=args.batch_size, num_workers=args.num_workers)
        return collect_logits(net, dl, device, feats=True)

    t0 = time.time()
    arrays, results = {}, {"ckpt": args.ckpt, "clean_split": summary, "substrate": args.substrate}
    for name in ("cal", "seen_test", "unseen_test"):
        logits, labels, feats = run(ds[name])
        arrays.update({f"{name}_logits": logits, f"{name}_labels": labels,
                       f"{name}_feats": feats.astype(np.float16)})
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
        results[name] = multilabel_metrics(labels, probs, list(ARTIFACT_CLASSES))
        print(f"{name:12s} macroF1={results[name]['macro_f1']:.3f}  ({time.time() - t0:.0f}s)",
              flush=True)
    ds["train"].set_epoch(0)
    _, ref_labels, ref_feats = run(Subset(ds["train"], range(min(args.n_ref, len(ds["train"])))))
    arrays["train_ref_feats"], arrays["train_ref_labels"] = ref_feats.astype(np.float16), ref_labels

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"preds_{args.tag}.npz", **arrays)
    (out / f"metrics_{args.tag}.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out / f'preds_{args.tag}.npz'}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
