"""Train the multi-label corruption classifier and log real metrics.

Reports macro-F1 on val, a seen-content test set, and a disjoint unseen-content
test set (the honest generalization number). Writes a JSON run record to runs/.

Two clean substrates: STL-10 photos (``substrate="stl10"``) and real game frames
(``substrate="real"``, artifacts injected at native resolution). ``split_seed`` fixes the
clean split, so runs that differ only in ``seed`` share identical test sets.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gpu_corruptnet.corruptions import available
from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES
from gpu_corruptnet.data.clean_sources import load_clean_splits, load_real_frames
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
    split_seed: int = 1337
    num_workers: int = 0
    data_root: str = "data"
    out_dir: str = "runs"
    tag: str = ""  # appended to output file names (e.g. "s3", "loto-shader")
    db_url: str | None = None  # e.g. postgresql+psycopg2://user:pw@host/db (optional)
    exclude: list[str] = field(default_factory=list)  # corruption types withheld from training
    max_train_minutes: float | None = None  # stop early (and say so) past this budget
    substrate: str = "stl10"  # "stl10" | "real"
    real_manifest: str = "realdata/clean_frames.csv"
    real_root: str = "data/real_frames"
    real_repeats: int = 8  # corruption draws per real frame per epoch (train)
    real_eval_repeats: int = 4  # corruption draws per real frame (val/cal/test)
    resize_first: bool = False  # ablation: paint artifacts at model resolution, not native
    init_ckpt: str | None = None  # warm start (e.g. the STL-10 model before real frames)
    save_feats: bool = False  # also dump penultimate embeddings per split


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _head(net: torch.nn.Module) -> torch.nn.Module:
    """The final Linear layer; its input is the embedding used for OOD scoring."""
    return [m for m in net.modules() if isinstance(m, torch.nn.Linear)][-1]


@torch.no_grad()
def collect_logits(
    net: torch.nn.Module, loader: DataLoader, device: torch.device, feats: bool = False
) -> tuple[np.ndarray, ...]:
    """Return raw (logits, labels[, embeddings]) for a split — feeds metrics and calibration."""
    net.eval()
    logits, labels, embs = [], [], []
    hook = None
    if feats:
        hook = _head(net).register_forward_hook(
            lambda _m, inp, _out: embs.append(inp[0].float().cpu().numpy())
        )
    try:
        for x, y in loader:
            logits.append(net(x.to(device)).float().cpu().numpy())
            labels.append(y.numpy())
    finally:
        if hook is not None:
            hook.remove()
    out = (np.concatenate(logits), np.concatenate(labels))
    return (*out, np.concatenate(embs)) if feats else out


def evaluate(net: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    logits, labels = collect_logits(net, loader, device)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
    return multilabel_metrics(labels, probs, list(ARTIFACT_CLASSES))


def build_splits(cfg: TrainConfig) -> tuple[dict[str, CorruptionDataset], str]:
    """Corruption datasets for train/val/cal/seen_test/unseen_test, plus a split summary.

    Eval sets use fixed base seeds (2..5) and epoch 0, so they are identical across runs."""
    train_pool = [c for c in available() if c not in set(cfg.exclude)]
    common = dict(
        img_size=cfg.img_size,
        clean_fraction=cfg.clean_fraction,
        max_labels=cfg.max_labels,
        resize_first=cfg.resize_first,
    )
    if cfg.substrate == "stl10":
        splits = load_clean_splits(cfg.data_root, cfg.n_seen, cfg.n_unseen, seed=cfg.split_seed)
        clean = {k: getattr(splits, k) for k in ("train", "val", "cal", "seen_test", "unseen_test")}
        train_kw: dict = {"hflip": True}
        rep_train = rep_eval = 1
    elif cfg.substrate == "real":
        splits = load_real_frames(cfg.real_manifest, cfg.real_root, seed=cfg.split_seed)
        clean = {k: getattr(splits, k) for k in ("train", "val", "cal", "seen_test", "unseen_test")}
        train_kw = {"hflip": True, "crop_scale": (0.5, 1.0)}
        rep_train, rep_eval = cfg.real_repeats, cfg.real_eval_repeats
    else:
        raise ValueError(f"unknown substrate '{cfg.substrate}'")

    def rep(items, r):
        return items if r == 1 else [x for x in items for _ in range(r)]

    seeds = {"train": 1, "val": 2, "seen_test": 3, "unseen_test": 4, "cal": 5}
    ds = {
        "train": CorruptionDataset(
            rep(clean["train"], rep_train), base_seed=seeds["train"], corruptions=train_pool,
            **common, **train_kw,
        )
    }
    for name in ("val", "cal", "seen_test", "unseen_test"):
        ds[name] = CorruptionDataset(rep(clean[name], rep_eval), base_seed=seeds[name], **common)
    return ds, splits.summary()


def run(cfg: TrainConfig) -> dict:
    seed_everything(cfg.seed)
    device = pick_device()
    print(f"device={device}  arch={cfg.arch}  threads={torch.get_num_threads()}")

    ds, split_summary = build_splits(cfg)
    print("clean frames:", split_summary)
    if cfg.exclude:
        print("withheld from training:", ", ".join(cfg.exclude))

    def loader(d: CorruptionDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            d, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers
        )

    train_dl = loader(ds["train"], True)
    eval_dl = {k: loader(ds[k], False) for k in ("val", "cal", "seen_test", "unseen_test")}

    net = build_classifier(
        cfg.arch, num_classes=len(ARTIFACT_CLASSES), freeze_backbone=cfg.freeze_backbone
    )
    if cfg.init_ckpt:
        state = torch.load(cfg.init_ckpt, map_location="cpu", weights_only=False)["state_dict"]
        net.load_state_dict(state)
        print(f"warm start from {cfg.init_ckpt}")
    net = net.to(device)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=cfg.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    history, val_m = [], {}
    t_start, stopped_early = time.time(), False
    for epoch in range(1, cfg.epochs + 1):
        ds["train"].set_epoch(epoch)  # fresh corruption of every frame each epoch
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
        val_m = evaluate(net, eval_dl["val"], device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(1, nb),
                "val_macro_f1": val_m["macro_f1"],
                "seconds": time.time() - t0,
            }
        )
        print(
            f"epoch {epoch}/{cfg.epochs}  loss={running / max(1, nb):.4f}  "
            f"val_macroF1={val_m['macro_f1']:.3f}  ({time.time() - t0:.1f}s)",
            flush=True,
        )
        elapsed = (time.time() - t_start) / 60
        if cfg.max_train_minutes and epoch < cfg.epochs:
            per_epoch = elapsed / epoch
            if elapsed + per_epoch > cfg.max_train_minutes:
                print(f"stopping after epoch {epoch}: time budget {cfg.max_train_minutes} min")
                stopped_early = True
                break

    # Collect raw logits once per split: drives both metrics and the M5 calibration step.
    preds = {
        name: collect_logits(net, eval_dl[name], device, feats=cfg.save_feats)
        for name in ("cal", "seen_test", "unseen_test")
    }

    def metrics_from(name: str) -> dict:
        logits, labels = preds[name][:2]
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
        return multilabel_metrics(labels, probs, list(ARTIFACT_CLASSES))

    results = {
        "config": asdict(cfg),
        "device": str(device),
        "clean_split": split_summary,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "train_minutes": (time.time() - t_start) / 60,
        "history": history,
        "val": val_m,
        "seen_test": metrics_from("seen_test"),
        "unseen_test": metrics_from("unseen_test"),
    }
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{cfg.arch}_{cfg.tag}_{stamp}" if cfg.tag else f"{cfg.arch}_{stamp}"
    path = out / f"metrics_{name}.json"
    path.write_text(json.dumps(results, indent=2))

    preds_path = out / f"preds_{name}.npz"
    arrays = {}
    for split, tup in preds.items():
        arrays[f"{split}_logits"], arrays[f"{split}_labels"] = tup[0], tup[1]
        if cfg.save_feats:
            arrays[f"{split}_feats"] = tup[2].astype(np.float16)
    if cfg.save_feats:  # reference embeddings for OOD scoring: clean + corrupted train frames
        ref = torch.utils.data.Subset(ds["train"], range(min(3000, len(ds["train"]))))
        ds["train"].set_epoch(0)
        _, ref_labels, ref_feats = collect_logits(net, loader(ref, False), device, feats=True)
        arrays["train_ref_feats"] = ref_feats.astype(np.float16)
        arrays["train_ref_labels"] = ref_labels
    np.savez(preds_path, **arrays)

    print("\n=== RESULTS ===")
    for split in ("seen_test", "unseen_test"):
        m = results[split]
        print(
            f"{split:12s}  macroF1={m['macro_f1']:.3f}  "
            f"binaryF1={m['binary_f1']:.3f}  binaryRecall={m['binary_recall']:.3f}"
        )
    ckpt_path = out / f"model_{name}.pt"
    torch.save(
        {
            "arch": cfg.arch,
            "num_classes": len(ARTIFACT_CLASSES),
            "img_size": cfg.img_size,
            "state_dict": net.state_dict(),
        },
        ckpt_path,
    )

    print(f"wrote {path}")
    print(f"wrote {preds_path}  (run: python scripts/calibrate.py {preds_path})")
    print(f"wrote {ckpt_path}  (trained weights, for serving/demo)")

    if cfg.db_url:  # optional: persist run + metrics to PostgreSQL (M2)
        from sqlalchemy.orm import Session

        from gpu_corruptnet.db.sql import get_engine, init_db, log_run

        engine = get_engine(cfg.db_url)
        init_db(engine)
        with Session(engine) as s:
            run_id = log_run(s, results)
        print(f"logged run {run_id} to {cfg.db_url}")

    return results
