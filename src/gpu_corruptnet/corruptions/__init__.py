"""Corruption generator package. Importing it registers all injectors."""

from gpu_corruptnet.corruptions import glitch  # noqa: F401  (triggers @register)
from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES
from gpu_corruptnet.corruptions.registry import apply, available, get, register

__all__ = ["ARTIFACT_CLASSES", "apply", "available", "get", "register"]
