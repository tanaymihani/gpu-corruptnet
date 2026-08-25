"""GPU-style artifact injectors (the "Glitchify-2" generator).

Pure NumPy/PIL so far (no OpenCV dependency): 6 of the 10 Glitchify classes.
Remaining (shader, shapes, triangulation, line_pixelation) need polygon
rasterization and land in the OpenCV batch next — see registry.available().

Each function is registered under its artifact-class name and follows the
``fn(img, severity, rng) -> img`` contract (returns a fresh uint8 array).
"""

from __future__ import annotations

import numpy as np

from gpu_corruptnet.corruptions.base import clip_uint8
from gpu_corruptnet.corruptions.registry import register


def _random_color(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 256, size=3, dtype=np.int16).astype(np.uint8)


@register("screen_tearing")
def screen_tearing(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Splice horizontal bands from a temporally-shifted 'next frame'.

    We approximate frame t+100 by a horizontal roll (camera motion), then
    replace random horizontal bands of frame t with it, producing the classic
    misaligned tear line.
    """
    h, w = img.shape[:2]
    out = img.copy()
    shift = int(rng.integers(w // 10, max(w // 10 + 1, w // 4)))
    frame2 = np.roll(img, shift, axis=1)
    n_tears = severity + 1
    for _ in range(n_tears):
        band_h = int(rng.integers(max(1, h // 12), max(2, h // 4)))
        top = int(rng.integers(0, max(1, h - band_h)))
        out[top : top + band_h] = frame2[top : top + band_h]
    return out


@register("morse_code")
def morse_code(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Stuck-memory 'morse-code' dashes: rows of stuck values (dash/gap runs)."""
    h, w = img.shape[:2]
    out = img.copy()
    n_rows = severity * 3
    for _ in range(n_rows):
        y = int(rng.integers(0, h))
        thick = int(rng.integers(1, 3))
        # Half the time a stuck black/white cell, half a stuck random color.
        if rng.random() < 0.5:
            color = np.uint8(rng.choice([0, 255]))
        else:
            color = _random_color(rng)
        x = 0
        while x < w:
            dash = int(rng.integers(3, 16))
            gap = int(rng.integers(3, 16))
            out[y : y + thick, x : x + dash] = color
            x += dash + gap
    return out


@register("discoloration")
def discoloration(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Channel-threshold shifts inside random regions -> local color casts."""
    h, w = img.shape[:2]
    out = img.astype(np.int16)
    n_regions = severity
    for _ in range(n_regions):
        c = int(rng.integers(0, 3))
        rw = int(rng.integers(w // 6, max(w // 6 + 1, w // 2)))
        rh = int(rng.integers(h // 6, max(h // 6 + 1, h // 2)))
        x0 = int(rng.integers(0, max(1, w - rw)))
        y0 = int(rng.integers(0, max(1, h - rh)))
        region = out[y0 : y0 + rh, x0 : x0 + rw, c]
        thr = int(rng.integers(60, 180))
        boost = int(rng.choice([-1, 1])) * int(rng.integers(90, 180))
        mask = region > thr if boost > 0 else region < thr
        region[mask] = region[mask] + boost
        out[y0 : y0 + rh, x0 : x0 + rw, c] = region
    return clip_uint8(out)


@register("parallel_lines")
def parallel_lines(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """N evenly-spaced parallel lines at a random angle (projection method)."""
    h, w = img.shape[:2]
    out = img.copy()
    yy, xx = np.mgrid[0:h, 0:w]
    theta = float(rng.uniform(0, np.pi))
    d = xx * np.cos(theta) + yy * np.sin(theta)
    n = int(rng.integers(60, 101))
    extent = float(d.max() - d.min())
    period = extent / n
    thickness = max(1.0, period * 0.15 * (0.5 + severity / 5.0))
    phase = (d - d.min()) % period
    out[phase < thickness] = _random_color(rng)
    return out


@register("dotted_lines")
def dotted_lines(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Like parallel_lines but each line is dotted along its direction."""
    h, w = img.shape[:2]
    out = img.copy()
    yy, xx = np.mgrid[0:h, 0:w]
    theta = float(rng.uniform(0, np.pi))
    d_perp = xx * np.cos(theta) + yy * np.sin(theta)  # across the lines
    d_par = -xx * np.sin(theta) + yy * np.cos(theta)  # along the lines
    n = int(rng.integers(60, 101))
    extent = float(d_perp.max() - d_perp.min())
    period = extent / n
    thickness = max(1.0, period * 0.18 * (0.5 + severity / 5.0))
    dot_period = float(rng.uniform(4, 10))
    phase_perp = (d_perp - d_perp.min()) % period
    phase_par = (d_par - d_par.min()) % dot_period
    mask = (phase_perp < thickness) & (phase_par < dot_period * 0.5)
    out[mask] = _random_color(rng)
    return out


def _swap_neighbour_blocks(arr: np.ndarray, axis: int, blk: int) -> np.ndarray:
    """Swap each pair of neighbouring ``blk``-thick slabs along ``axis``."""
    n = (arr.shape[axis] // (2 * blk)) * (2 * blk)
    if n == 0:
        return arr
    order = list(range(blk, 2 * blk)) + list(range(0, blk))
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(0, n)
    head = arr[tuple(sl)]
    shape = list(head.shape)
    shape[axis : axis + 1] = [n // (2 * blk), 2 * blk]
    grouped = head.reshape(shape)
    idx = [slice(None)] * grouped.ndim
    idx[axis + 1] = order
    arr[tuple(sl)] = grouped[tuple(idx)].reshape(head.shape)
    return arr


@register("screen_stuttering")
def screen_stuttering(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Swap neighbouring row/column *blocks* within random bands -> jitter."""
    h, w = img.shape[:2]
    out = img.copy()
    blk = 3
    for _ in range(severity):
        band_h = int(rng.integers(max(2 * blk, h // 8), max(2 * blk + 1, h // 3)))
        y0 = int(rng.integers(0, max(1, h - band_h)))
        out[y0 : y0 + band_h] = _swap_neighbour_blocks(out[y0 : y0 + band_h].copy(), 0, blk)

        band_w = int(rng.integers(max(2 * blk, w // 8), max(2 * blk + 1, w // 3)))
        x0 = int(rng.integers(0, max(1, w - band_w)))
        out[:, x0 : x0 + band_w] = _swap_neighbour_blocks(out[:, x0 : x0 + band_w].copy(), 1, blk)
    return out
