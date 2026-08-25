"""CLI to train the corruption classifier.

Examples:
    python scripts/train_classifier.py --smoke                # fast local sanity run
    python scripts/train_classifier.py --arch resnet50 --epochs 8 --img-size 224
"""

from __future__ import annotations

import argparse

from gpu_corruptnet.train import TrainConfig, run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "efficientnet_b4"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--n-seen", type=int, default=None)
    ap.add_argument("--n-unseen", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-freeze", action="store_true", help="fine-tune the whole backbone")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--db-url", default=None, help="optional SQLAlchemy URL to log run + metrics")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to validate the pipeline")
    args = ap.parse_args()

    cfg = TrainConfig(
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        n_seen=args.n_seen,
        n_unseen=args.n_unseen,
        num_workers=args.num_workers,
        freeze_backbone=not args.no_freeze,
        data_root=args.data_root,
        out_dir=args.out_dir,
        db_url=args.db_url,
        seed=args.seed,
    )
    if args.smoke:
        cfg.epochs = 2
        cfg.img_size = 96
        cfg.batch_size = 32
        cfg.n_seen = 800
        cfg.n_unseen = 200

    run(cfg)


if __name__ == "__main__":
    main()
