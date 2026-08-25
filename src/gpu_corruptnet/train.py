"""Train the multi-label corruption classifier and log real metrics.

Reports macro-F1 on val, a seen-content test set, and a disjoint unseen-content
test set (the honest generalization number). Writes a JSON run record to runs/.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES
from gpu_corruptnet.data.clean_sources import load_clean_splits
from gpu_corruptnet.data.corrupt_dataset import CorruptionDataset
from gpu_corruptnet.metrics import multilabel_metrics
from gpu_corruptnet.models import build_classifier
from gpu_corruptnet.utils.seed import seed_everything


@dataclass
class TrainConfig:
    arch: str = "resnet50"
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    img_size: int = 128
    n_seen: int | None = None
    n_unseen: int | None = None
    clean_fraction: float = 0.3
    max_labels: int = 2
    freeze_backbone: bool = True
    seed: int = 1337
    num_workers: int = 0
    data_root: str = "data"
    out_dir: str = "runs"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(net: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    net.eval()
    probs, trues = [], []
    for x, y in loader:
        logits = net(x.to(device))
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        trues.append(y.numpy())
    return multilabel_metrics(
        np.concatenate(trues), np.concatenate(probs), list(ARTIFACT_CLASSES)
    )


def run(cfg: TrainConfig) -> dict:
    seed_everything(cfg.seed)
    device = pick_device()
    print(f"device={device}  arch={cfg.arch}")

    splits = load_clean_splits(cfg.data_root, cfg.n_seen, cfg.n_unseen, seed=cfg.seed)
    print("clean frames:", splits.summary())

    def make(clean: np.ndarray, base_seed: int, hflip: bool) -> CorruptionDataset:
        return CorruptionDataset(
            clean,
            base_seed=base_seed,
            img_size=cfg.img_size,
            clean_fraction=cfg.clean_fraction,
            max_labels=cfg.max_labels,
            hflip=hflip,
        )

    def loader(ds: CorruptionDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers
        )

    train_dl = loader(make(splits.train, 1, True), True)
    val_dl = loader(make(splits.val, 2, False), False)
    seen_dl = loader(make(splits.seen_test, 3, False), False)
    unseen_dl = loader(make(splits.unseen_test, 4, False), False)

    net = build_classifier(
        cfg.arch, num_classes=len(ARTIFACT_CLASSES), freeze_backbone=cfg.freeze_backbone
    ).to(device)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=cfg.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    history, val_m = [], {}
    for epoch in range(1, cfg.epochs + 1):
        net.train()
        t0, running, nb = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        val_m = evaluate(net, val_dl, device)
        history.append(
            {"epoch": epoch, "train_loss": running / max(1, nb), "val_macro_f1": val_m["macro_f1"]}
        )
        print(
            f"epoch {epoch}/{cfg.epochs}  loss={running / max(1, nb):.4f}  "
            f"val_macroF1={val_m['macro_f1']:.3f}  ({time.time() - t0:.1f}s)"
        )

    results = {
        "config": asdict(cfg),
        "device": str(device),
        "clean_split": splits.summary(),
        "history": history,
        "val": val_m,
        "seen_test": evaluate(net, seen_dl, device),
        "unseen_test": evaluate(net, unseen_dl, device),
    }
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out / f"metrics_{cfg.arch}_{stamp}.json"
    path.write_text(json.dumps(results, indent=2))

    print("\n=== RESULTS ===")
    for split in ("seen_test", "unseen_test"):
        m = results[split]
        print(
            f"{split:12s}  macroF1={m['macro_f1']:.3f}  "
            f"binaryF1={m['binary_f1']:.3f}  binaryRecall={m['binary_recall']:.3f}"
        )
    print(f"wrote {path}")
    return results
