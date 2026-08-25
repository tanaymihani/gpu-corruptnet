"""Render a visual sanity grid: clean frame + every implemented corruption.

Usage:
    python scripts/make_sanity_grid.py [--out assets/sanity_grid.png] [--severity 3]

This is the headline visual artifact for the repo — it proves the generator
works at a glance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from gpu_corruptnet.corruptions import apply, available
from gpu_corruptnet.data import make_demo_frame

LABEL_H = 18


def _tile(img: np.ndarray, label: str) -> Image.Image:
    h, w = img.shape[:2]
    canvas = Image.new("RGB", (w, h + LABEL_H), (20, 20, 20))
    canvas.paste(Image.fromarray(img), (0, LABEL_H))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), label, fill=(240, 240, 240))
    return canvas


def build_grid(severity: int, seed: int, cols: int = 4) -> Image.Image:
    rng = np.random.default_rng(seed)
    clean = make_demo_frame(rng=rng)

    tiles = [_tile(clean, "clean")]
    for name in available():
        corrupted = apply(name, clean, severity=severity, rng=np.random.default_rng(seed))
        tiles.append(_tile(corrupted, f"{name}  (sev {severity})"))

    tw, th = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    pad = 6
    grid = Image.new(
        "RGB",
        (cols * tw + (cols + 1) * pad, rows * th + (rows + 1) * pad),
        (10, 10, 10),
    )
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        grid.paste(tile, (pad + c * (tw + pad), pad + r * (th + pad)))
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/sanity_grid.png", type=Path)
    ap.add_argument("--severity", default=3, type=int)
    ap.add_argument("--seed", default=1337, type=int)
    args = ap.parse_args()

    grid = build_grid(args.severity, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out)
    print(f"wrote {args.out}  ({grid.size[0]}x{grid.size[1]}, {len(available())} corruptions)")


if __name__ == "__main__":
    main()
