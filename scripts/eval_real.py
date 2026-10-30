"""Score checkpoints on real images: clean game frames (realdata/clean_frames.csv) and
hand-labelled screenshots from public bug reports (realdata/real_glitches.csv).

    python realdata/fetch.py realdata/real_glitches.csv data/real_glitches
    python scripts/eval_real.py deployed=model.pt native=runs/.../model_real-native.pt

Writes runs/real_eval_<name>.npz (logits + embeddings per image) and prints binary
detection metrics: a frame counts as flagged when any artifact probability >= 0.5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gpu_corruptnet.data.clean_sources import read_manifest
from gpu_corruptnet.data.corrupt_dataset import IMAGENET_MEAN, IMAGENET_STD
from gpu_corruptnet.models import load_classifier
from gpu_corruptnet.train import _head


def load(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
    x = (np.asarray(im, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)


@torch.no_grad()
def run(net, paths: list[Path], size: int, bs: int = 16) -> tuple[np.ndarray, np.ndarray]:
    logits, embs = [], []
    hook = _head(net).register_forward_hook(lambda _m, i, _o: embs.append(i[0].numpy()))
    for k in range(0, len(paths), bs):
        x = torch.from_numpy(np.stack([load(p, size) for p in paths[k : k + bs]]))
        logits.append(net(x).numpy())
    hook.remove()
    return np.concatenate(logits), np.concatenate(embs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="name=checkpoint.pt")
    ap.add_argument("--frames", default="realdata/clean_frames.csv")
    ap.add_argument("--frames-root", default="data/real_frames")
    ap.add_argument("--glitches", default="realdata/real_glitches.csv")
    ap.add_argument("--glitches-root", default="data/real_glitches")
    ap.add_argument("--out-dir", default="runs")
    args = ap.parse_args()
    torch.set_num_threads(3)

    frames = [r for r in read_manifest(args.frames)
              if (Path(args.frames_root) / r["file"]).exists()]
    shots = [r for r in read_manifest(args.glitches)
             if (Path(args.glitches_root) / r["file"]).exists()]
    paths = [Path(args.frames_root) / r["file"] for r in frames] + \
            [Path(args.glitches_root) / r["file"] for r in shots]
    source = np.array(["commons"] * len(frames) + ["bugreport"] * len(shots))
    is_glitch = np.array([False] * len(frames) + [r["label"] == "glitch" for r in shots])
    print(f"{len(frames)} clean Commons frames, {len(shots)} bug-report images "
          f"({int(is_glitch.sum())} glitches)")

    summary = {}
    for spec in args.models:
        name, ckpt = spec.split("=", 1)
        net, meta = load_classifier(ckpt)
        logits, embs = run(net, paths, int(meta["img_size"]))
        np.savez(Path(args.out_dir) / f"real_eval_{name}.npz", logits=logits,
                 embeddings=embs.astype(np.float16), source=source, is_glitch=is_glitch,
                 files=np.array([str(p) for p in paths]))
        flagged = (logits >= 0).any(1)
        bug = source == "bugreport"
        tp = int((flagged & is_glitch).sum())
        fp_bug = int((flagged & bug & ~is_glitch).sum())
        rec = tp / max(1, int(is_glitch.sum()))
        prec = tp / max(1, tp + fp_bug)
        summary[name] = {
            "glitch_recall": rec,
            "bugreport_precision": prec,
            "bugreport_f1": 2 * prec * rec / max(1e-9, prec + rec),
            "fpr_bugreport_clean": fp_bug / max(1, int((bug & ~is_glitch).sum())),
            "fpr_commons_clean": float(flagged[source == "commons"].mean()),
        }
        print(name, json.dumps({k: round(v, 3) for k, v in summary[name].items()}))
    Path(args.out_dir, "real_eval_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
