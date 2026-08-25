"""Shared types & helpers for the synthetic corruption generator ("Glitchify-2").

Convention used everywhere: an image is a ``np.ndarray`` of shape ``(H, W, 3)``,
dtype ``uint8``, channel order RGB. Severity is an int in ``[1, 5]``.
"""

from __future__ import annotations

import numpy as np

# The 10 software-reproducible artifact classes from the AMD/UCLA "Glitchify"
# paper (arXiv:2011.15103). This is the full multi-label target vocabulary;
# see the registry for which injectors are implemented so far.
ARTIFACT_CLASSES: tuple[str, ...] = (
    "shader",
    "shapes",
    "discoloration",
    "morse_code",
    "dotted_lines",
    "parallel_lines",
    "triangulation",
    "line_pixelation",
    "screen_stuttering",
    "screen_tearing",
)

MIN_SEVERITY = 1
MAX_SEVERITY = 5


def ensure_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Coerce ``None`` / an int seed / a Generator into a Generator."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def check_image(img: np.ndarray) -> None:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB image, got shape {img.shape}")
    if img.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got dtype {img.dtype}")


def check_severity(severity: int) -> int:
    if not (MIN_SEVERITY <= severity <= MAX_SEVERITY):
        raise ValueError(f"severity must be in [{MIN_SEVERITY}, {MAX_SEVERITY}], got {severity}")
    return severity


def clip_uint8(a: np.ndarray) -> np.ndarray:
    """Clip an arbitrary-dtype array to a valid uint8 image."""
    return np.clip(a, 0, 255).astype(np.uint8)
