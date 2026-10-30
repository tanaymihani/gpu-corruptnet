---
license: mit
library_name: pytorch
pipeline_tag: image-classification
tags:
  - gpu
  - rendering-artifacts
  - glitch-detection
  - multi-label-classification
  - calibration
  - conformal-prediction
metrics:
  - f1
datasets:
  - tanganke/stl10
---

# GPU-CorruptNet ResNet-50

Multi-label detector for 10 GPU rendering-artifact types (the
[Glitchify](https://arxiv.org/abs/2011.15103) taxonomy): shader, shapes, discoloration,
morse code, dotted lines, parallel lines, triangulation, line pixelation, screen stuttering,
screen tearing. An all-negative output means "clean". Code, data manifests and every
number below: [github.com/tanaymihani/gpu-corruptnet](https://github.com/tanaymihani/gpu-corruptnet).
Try it in the browser: [tanaymihani.github.io/gpu-corruptnet](https://tanaymihani.github.io/gpu-corruptnet/).

## Files

- `model.safetensors`: torchvision `resnet50` weights with `fc` replaced by a 10-way linear layer
- `config.json`: class order, input size, normalisation, temperature, conformal threshold

```python
import json, torch, torchvision
from safetensors.torch import load_file

cfg = json.load(open("config.json"))
net = torchvision.models.resnet50()
net.fc = torch.nn.Linear(2048, cfg["num_classes"])
net.load_state_dict(load_file("model.safetensors"))
net.eval()
# x: (N, 3, 224, 224) after a bilinear squash-resize and ImageNet normalisation
probs = torch.sigmoid(net(x) / cfg["temperature"])          # calibrated per-class probabilities
flagged = probs >= cfg["decision_threshold"]                # predicted artifact types
conformal_set = probs >= cfg["conformal_tau"]               # ~90% full-inclusion set
```

## Training

ImageNet-pretrained ResNet-50 fine-tuned 12 epochs at 224 px (Adam, lr 1e-4, BCE) on
STL-10 photos corrupted on the fly by a re-implementation of the Glitchify injectors
(30% clean, else 1–2 artifacts at severity 1–5). Ship and truck images are held out as
unseen content. Free Colab T4.

## Evaluation (identical held-out frames, 95% bootstrap CIs)

| split | macro-F1 | binary F1 (corrupted vs clean) |
|---|---|---|
| seen content (1,560 frames) | 0.911 [0.899, 0.922] | 0.991 |
| unseen content (2,600 frames) | 0.876 [0.866, 0.884] | 0.971 |

Calibration (T = 1.5647, fitted on a disjoint 1,560-frame split): pooled ECE 1.06% → 0.26%
(seen) and 1.42% → 0.17% (unseen); classwise ECE 1.37% → 1.18% and 1.67% → 1.39%.
Conformal threshold τ = 0.0493 (α = 0.1): full-inclusion coverage on corrupted frames
0.905 seen / 0.895 unseen. Weakest classes on unseen content: screen tearing (F1 0.63) and shapes (0.75).

## Limitations (read before use)

- **Synthetic training data only.** On real game screenshots this checkpoint flags 78% of
  clean Commons frames and 80% of clean bug-report screenshots as corrupted. It separates
  117 real glitch screenshots from 382 clean ones at AUROC 0.52, i.e. chance. Do not use it
  on real frames without fine-tuning on them; see the repository's real-frame experiments.
- Its embedding-based OOD check does not flag real frames either (they embed near the
  corrupted training frames), so it cannot warn you about this shift.
- Inputs are squashed to 224 × 224; fine artifacts at native resolution (1–2 px dashes)
  may disappear in the resize.

## Citation

If you use this model, please cite the Glitchify taxonomy (Davarmanesh et al., 2020) and
the repository above.
