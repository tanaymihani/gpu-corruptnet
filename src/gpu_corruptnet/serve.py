"""Single-frame inference core for the demo/serving path (M10).

Wraps the trained classifier + M5 calibration + M9 OOD score behind one call.
Model-optional: without a checkpoint it still returns the OOD/drift score (so the
generator + drift story is demoable), and with one it adds detection + calibrated
confidence + conformal set + latency. Streamlit UI sits on top of this.
"""

from __future__ import annotations

import time

import numpy as np

from gpu_corruptnet.calibration import sigmoid
from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES
from gpu_corruptnet.data import make_demo_frame
from gpu_corruptnet.data.corrupt_dataset import IMAGENET_MEAN, IMAGENET_STD
from gpu_corruptnet.drift import DESCRIPTOR_NAMES, DriftMonitor, image_descriptors


class CorruptionInspector:
    def __init__(
        self,
        model_path: str | None = None,
        temperature: float = 1.0,
        conformal_tau: float | None = None,
        device: str = "cpu",
    ) -> None:
        self.class_names = list(ARTIFACT_CLASSES)
        self.temperature = temperature
        self.conformal_tau = conformal_tau
        self.device = device
        self.img_size = 224
        self.net = None
        if model_path:
            from gpu_corruptnet.models import load_classifier

            self.net, ckpt = load_classifier(model_path, map_location=device)
            self.net.to(device)
            self.img_size = int(ckpt.get("img_size", 224))

        # Small clean reference so we can report a per-frame OOD score with no model.
        ref = np.stack([make_demo_frame(rng=s) for s in range(64)])
        self.drift = DriftMonitor(feature_names=DESCRIPTOR_NAMES).fit(image_descriptors(ref))

    def _preprocess(self, img: np.ndarray):
        import torch
        from PIL import Image

        im = Image.fromarray(img).resize((self.img_size, self.img_size), Image.BILINEAR)
        x = np.asarray(im).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

    def predict(self, img: np.ndarray) -> dict:
        feats = image_descriptors(img[None])
        out: dict = {
            "ood_score": float(self.drift.mahalanobis(feats)[0]),
            "ood_flag": bool(self.drift.ood_flags(feats)[0]),
            "has_model": self.net is not None,
        }
        if self.net is None:
            return out

        import torch

        x = self._preprocess(img)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.net(x).cpu().numpy()[0]
        out["latency_ms"] = (time.perf_counter() - t0) * 1000.0

        cal = sigmoid(logits / self.temperature)
        out["probs"] = {c: float(p) for c, p in zip(self.class_names, cal, strict=True)}
        out["predicted"] = [c for c, p in zip(self.class_names, cal, strict=True) if p >= 0.5]
        out["corrupted_prob"] = float(cal.max())
        if self.conformal_tau is not None:
            out["conformal_set"] = [
                c for c, p in zip(self.class_names, cal, strict=True) if p >= self.conformal_tau
            ]
        return out
