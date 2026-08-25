"""GPU-CorruptNet: detect & classify GPU-rendered visual corruption."""

from gpu_corruptnet.corruptions import apply, available, get  # noqa: F401
from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES  # noqa: F401

__version__ = "0.1.0"
