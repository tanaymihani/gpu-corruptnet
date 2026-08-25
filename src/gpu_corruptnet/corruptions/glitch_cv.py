"""OpenCV-based artifact injectors — the polygon-rasterization batch.

Completes the 10 Glitchify classes: shader, shapes, triangulation, line_pixelation.
These only register if OpenCV is installed (``pip install -e ".[cv]"``); the base
package stays OpenCV-free, so ``available()`` honestly reflects what's importable.
"""

from __future__ import annotations

import cv2
import numpy as np

from gpu_corruptnet.corruptions.base import clip_uint8
from gpu_corruptnet.corruptions.registry import register


@register("triangulation")
def triangulation(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Low-poly effect: Delaunay-tessellate, fill each triangle with its mean color.

    Higher severity -> fewer points -> coarser triangles -> more visible.
    """
    h, w = img.shape[:2]
    out = img.copy()
    n_points = int(np.interp(severity, [1, 5], [500, 90]))
    pts = rng.integers(0, [w, h], size=(n_points, 2))
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    pts = np.vstack([pts, corners])

    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for px, py in pts:
        try:
            subdiv.insert((int(px), int(py)))
        except cv2.error:
            pass  # duplicate / out-of-rect point

    for t in subdiv.getTriangleList():
        tri = np.array([[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]], dtype=np.int32)
        if tri.min() < 0 or tri[:, 0].max() >= w or tri[:, 1].max() >= h:
            continue  # skip virtual/outside triangles
        x, y, bw, bh = cv2.boundingRect(tri)
        if bw == 0 or bh == 0:
            continue
        mask = np.zeros((bh, bw), np.uint8)
        cv2.fillConvexPoly(mask, tri - [x, y], 255)
        m = mask.astype(bool)
        if not m.any():
            continue
        mean = img[y : y + bh, x : x + bw][m].mean(axis=0)
        out[y : y + bh, x : x + bw][m] = mean.astype(np.uint8)
    return out


@register("shapes")
def shapes(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Random thin dark polygons seeded around the darkest region of the frame."""
    h, w = img.shape[:2]
    out = img.copy()
    small = cv2.resize(img.mean(axis=2), (16, 9), interpolation=cv2.INTER_AREA)
    sy, sx = np.unravel_index(int(np.argmin(small)), small.shape)
    cx, cy = int((sx + 0.5) / 16 * w), int((sy + 0.5) / 9 * h)

    for _ in range(severity * 2):
        nv = int(rng.integers(3, 7))
        cxi = cx + int(rng.integers(-w // 6, w // 6 + 1))
        cyi = cy + int(rng.integers(-h // 6, h // 6 + 1))
        r = int(rng.integers(4, max(5, min(w, h) // 10)))
        ang = np.sort(rng.uniform(0, 2 * np.pi, size=nv))
        poly = np.stack([cxi + r * np.cos(ang), cyi + r * np.sin(ang)], axis=1).astype(np.int32)
        shade = int(rng.integers(0, 40))
        cv2.polylines(out, [poly], isClosed=True, color=(shade, shade, shade),
                      thickness=int(rng.integers(1, 3)))
    return out


@register("shader")
def shader(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Random-vertex polygons filled with a color gradient fading from a seed pixel."""
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(max(1, severity)):
        sx, sy = int(rng.integers(0, w)), int(rng.integers(0, h))
        seed_color = img[sy, sx].astype(np.float32)
        target = rng.integers(0, 256, size=3).astype(np.float32)
        nv = int(rng.integers(3, 8))
        r = int(rng.integers(max(4, min(w, h) // 10), max(6, min(w, h) // 3)))
        ang = np.sort(rng.uniform(0, 2 * np.pi, size=nv))
        poly = np.stack([sx + r * np.cos(ang), sy + r * np.sin(ang)], axis=1).astype(np.int32)

        x, y, bw, bh = cv2.boundingRect(poly)
        x, y = max(0, x), max(0, y)
        bw, bh = min(bw, w - x), min(bh, h - y)
        if bw <= 0 or bh <= 0:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        region = mask[y : y + bh, x : x + bw].astype(bool)

        grad = np.clip((np.linspace(0, 1, bw)[None, :] + np.linspace(0, 1, bh)[:, None]) / 2, 0, 1)
        patch = seed_color * (1 - grad[..., None]) + target * grad[..., None]
        out[y : y + bh, x : x + bw][region] = clip_uint8(patch)[region]
    return out


@register("line_pixelation")
def line_pixelation(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Noisy stripes at a random orientation, each ringed by a brightened halo."""
    h, w = img.shape[:2]
    out = img.copy()
    yy, xx = np.mgrid[0:h, 0:w]
    theta = float(rng.uniform(0, np.pi))
    d = xx * np.cos(theta) + yy * np.sin(theta)
    dmin = float(d.min())
    extent = float(d.max() - dmin)

    for _ in range(severity + 1):
        center = float(rng.uniform(0, extent))
        thick = float(rng.uniform(extent * 0.01, extent * 0.04))
        dist = np.abs((d - dmin) - center)
        band = dist < thick
        halo = (dist < thick * 2.2) & ~band
        noise = rng.integers(0, 256, size=(int(band.sum()), 3), dtype=np.int16).astype(np.uint8)
        out[band] = noise
        out[halo] = clip_uint8(out[halo].astype(np.int16) + 80)
    return out
