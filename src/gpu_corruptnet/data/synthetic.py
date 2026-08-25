"""A dependency-free procedural 'gameplay frame' so the generator and its sanity
grid work with zero external data. Swapped for real game frames in M2.
"""

from __future__ import annotations

import numpy as np

from gpu_corruptnet.corruptions.base import clip_uint8


def make_demo_frame(
    h: int = 270, w: int = 480, rng: np.random.Generator | int | None = 0
) -> np.ndarray:
    """Render a simple structured scene (sky gradient, sun, ground, buildings).

    Structure matters: flat images hide many artifacts, so we include edges,
    gradients and solid regions that each corruption can visibly disturb.
    """
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    img = np.zeros((h, w, 3), dtype=np.float64)
    horizon = int(h * 0.62)

    # Sky: vertical gradient from deep blue to pale near the horizon.
    t = np.linspace(0.0, 1.0, horizon)[:, None]
    top, bot = np.array([30, 45, 90]), np.array([180, 205, 235])
    img[:horizon] = (top * (1 - t) + bot * t)[:, None, :]

    # Sun.
    sy, sx, sr = int(horizon * 0.35), int(w * 0.78), int(min(h, w) * 0.09)
    yy, xx = np.mgrid[0:h, 0:w]
    sun = (yy - sy) ** 2 + (xx - sx) ** 2 <= sr**2
    img[sun] = np.array([255, 240, 180])

    # Ground: warm gradient.
    gt = np.linspace(0.0, 1.0, h - horizon)[:, None]
    gtop, gbot = np.array([80, 120, 60]), np.array([40, 70, 35])
    img[horizon:] = (gtop * (1 - gt) + gbot * gt)[:, None, :]

    # A skyline of buildings sitting on the horizon.
    x = int(w * 0.06)
    while x < w * 0.94:
        bw = int(rng.integers(w // 22, w // 10))
        bh = int(rng.integers(h // 8, h // 3))
        shade = int(rng.integers(40, 110))
        img[max(0, horizon - bh) : horizon, x : x + bw] = np.array([shade, shade, shade + 15])
        x += bw + int(rng.integers(w // 60, w // 25))

    return clip_uint8(img)
