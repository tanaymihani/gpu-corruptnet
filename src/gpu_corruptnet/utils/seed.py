"""Reproducibility helpers. Everything random flows from one seed."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 1337) -> np.random.Generator:
    """Seed stdlib + NumPy (and torch if present) and return a Generator."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass
    return np.random.default_rng(seed)
