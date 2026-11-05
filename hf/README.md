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
  - roc_auc
datasets:
  - tanganke/stl10
---

# GPU-CorruptNet ResNet-50 (real-frame fine-tune)

Multi-label detector for 10 GPU rendering-artifact types (the
[Glitchify](https://arxiv.org/abs/2011.15103) taxonomy): shader, shapes, discoloration,
morse code, dotted lines, parallel lines, triangulation, line pixelation, screen stuttering,
screen tearing. An all-negative output means "clean". This is the checkpoint the always-on
browser demo serves: [tanaymihani.github.io/gpu-corruptnet](https://tanaymihani.github.io/gpu-corruptnet/).
Code, data manifests and every number below: [github.com/tanaymihani/gpu-corruptnet](https://github.com/tanaymihani/gpu-corruptnet).

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
conformal_set = probs >= cfg["conformal_tau"]               # conformal set of plausible types
```

## Training

1. ImageNet-pretrained ResNet-50, fine-tuned 12 epochs at 224 px (Adam, lr 1e-4, BCE) on STL-10
   photos corrupted on the fly by a re-implementation of the Glitchify injectors (30% clean,
   else 1–2 artifacts at severity 1–5).
2. Then fine-tuned 8 epochs (lr 5e-5) on 132 free-licence screenshots of 16 open-source games
   from Wikimedia Commons, with artifacts injected at the frame's native resolution (random
   crops, 8 corruption draws per frame per epoch). Four further titles are held out.

## Evaluation

| test set | metric | this model | STL-10-only checkpoint |
|---|---|---|---|
| Real glitch screenshots vs clean screenshots from public bug reports (200 vs 176) | AUROC | **0.67 [0.62, 0.73]** | 0.53 [0.47, 0.58] |
| Clean free-licence game frames (326) | false-alarm rate | **14%** | 78% |
| Synthetic artifacts on 4 held-out game titles (344 draws) | macro-F1 | **0.694** | 0.335 |
| Synthetic artifacts on STL-10 photos, unseen content (2,600) | macro-F1 | 0.711 [0.698, 0.723] | **0.876** [0.866, 0.884] |

Fine-tuning on real frames trades synthetic STL-10 accuracy for far fewer false alarms on real
frames. The STL-10-only checkpoint (GitHub release `demo-model`) flags most clean real frames
as corrupted. The real-glitch labels come from a two-pass visual review of public bug-report
screenshots (`realdata/real_glitches.csv`, 123 disputed rows excluded); only the URLs are
distributed. At the 0.5 threshold this model catches 50% of the real glitches; it is a triage
aid, not a replacement for a tester.

Calibration: temperature T = 1.057 and conformal threshold τ = 0.0484 (α = 0.1), both fitted on a
real-frame calibration split. Coverage on corrupted held-out frames is 0.87, below the 0.90
target: the calibration split is small (36 frames × 4 draws).

## Limitations

- Most real driver faults (missing geometry, garbage textures, wrong buffers) map only loosely
  onto the ten procedural types. On real screenshots, trust the binary "corrupted" score more
  than the per-class labels.
- Inputs are squashed to 224 × 224, so artifacts 1–2 px wide at native resolution can vanish.
- Trained on 16 games; content far from those (2D sprites, video, desktop UI) is out of scope.

## Citation

If you use this model, please cite the Glitchify taxonomy (Davarmanesh et al., 2020) and the
repository above.
