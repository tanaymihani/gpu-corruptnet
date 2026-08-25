"""A tiny registry so injectors self-register and can be applied by name.

Each injector has signature ``fn(img, severity, rng) -> img`` and returns a new
uint8 (H, W, 3) array (never mutates its input).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from gpu_corruptnet.corruptions.base import check_image, check_severity, ensure_rng

Injector = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]

_REGISTRY: dict[str, Injector] = {}


def register(name: str) -> Callable[[Injector], Injector]:
    def deco(fn: Injector) -> Injector:
        if name in _REGISTRY:
            raise ValueError(f"corruption '{name}' already registered")
        _REGISTRY[name] = fn
        return fn

    return deco


def get(name: str) -> Injector:
    if name not in _REGISTRY:
        raise KeyError(f"unknown corruption '{name}'; available: {available()}")
    return _REGISTRY[name]


def available() -> list[str]:
    """Names of injectors implemented so far (subset of ARTIFACT_CLASSES)."""
    return sorted(_REGISTRY)


def apply(
    name: str,
    img: np.ndarray,
    severity: int = 3,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Apply a single named corruption. Validates inputs and returns a new image."""
    check_image(img)
    check_severity(severity)
    out = get(name)(img, severity, ensure_rng(rng))
    check_image(out)
    return out
