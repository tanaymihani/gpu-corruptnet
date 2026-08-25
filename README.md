# GPU-CorruptNet

**An end-to-end computer-vision pipeline that detects and classifies GPU-rendered visual corruption** —
shader glitches, screen tearing, texture/block artifacts, discoloration, and stuck-memory
"morse-code" patterns — in rendered frames.

The project pairs a **supervised multi-label corruption classifier** (PyTorch) with an
**unsupervised anomaly head** for *unknown* corruption, trained on a self-built synthetic
corruption generator, and serves it behind a benchmarked inference API with **calibrated,
conformal confidence**.

> **Prior art / inspiration.** The 10 artifact classes follow the AMD/UCLA-RIPS *"Glitchify"*
> paper — *Automating Artifact Detection in Video Games* ([arXiv:2011.15103](https://arxiv.org/abs/2011.15103)) —
> which procedurally injects software-reproducible GPU artifacts because real corruption data is
> scarce. GPU-CorruptNet modernizes that 2020 classical-feature ensemble (84% seen / 69% unseen)
> with deep CNNs, an anomaly head for novel corruptions, and uncertainty quantification.

## The synthetic corruption generator ("Glitchify-2")

The crux of the project: real GPU-corruption datasets barely exist, so we *generate* labeled data
by procedurally injecting artifacts into clean frames. Below, one procedural frame corrupted by
each currently-implemented injector (`python scripts/make_sanity_grid.py`):

![Sanity grid of synthetic corruptions](assets/sanity_grid.png)

## Status

Actively building. Honest state — nothing here claims a metric it hasn't measured.

| Milestone | Scope | State |
|---|---|---|
| **M0** | Repo scaffold, config, tests, CI | ✅ done |
| **M1** | Glitchify-2 generator (10 artifact classes) + ImageNet-C wrapper | 🚧 6/10 injectors done |
| **M2** | Data pipeline + PostgreSQL/MongoDB metadata stores | ⬜ next |
| **M3** | ResNet-50 / EfficientNet-B4 classifiers + per-class metrics | ⬜ |
| **M4** | Unsupervised anomaly head (EfficientAD / PatchCore) | ⬜ |
| **M5** | Calibration (temperature scaling, ECE) + conformal sets (MAPIE) | ⬜ |
| **M6** | C++/libtorch inference path (+ optional HIP kernel) | ⬜ |
| **M7** | ONNX / FP16 export + latency-throughput benchmark harness | ⬜ |
| **M8** | AWS deploy (S3 + EC2-Spot) + FastAPI demo | ⬜ |
| **M9** | Drift / OOD monitor ("fixing deployed AI") | ⬜ |
| **M10** | Upload-a-frame dashboard demo | ⬜ |

**Implemented injectors:** `screen_tearing`, `screen_stuttering`, `morse_code`, `discoloration`,
`parallel_lines`, `dotted_lines`.
**Remaining (need polygon rasterization, OpenCV batch):** `shader`, `shapes`, `triangulation`,
`line_pixelation`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python scripts/make_sanity_grid.py     # -> assets/sanity_grid.png
pytest -q                              # generator unit tests
```

Apply a corruption programmatically:

```python
import numpy as np
from gpu_corruptnet.corruptions import apply, available
from gpu_corruptnet.data import make_demo_frame

frame = make_demo_frame()                      # or your own (H, W, 3) uint8 RGB frame
print(available())                             # implemented injectors
glitched = apply("screen_tearing", frame, severity=4, rng=0)
```

## Design notes

- **Images** are `(H, W, 3)` `uint8` RGB throughout. **Severity** is `1..5`.
- Injectors **self-register** (`@register`) and never mutate their input; each is
  **deterministic given a seed** (enforced in tests) so every generated dataset is reproducible.
- The base install is intentionally light (NumPy/Pillow). Heavier deps arrive with the milestone
  that needs them — `[cv]`, `[train]`, `[common]` extras in `pyproject.toml`.

## Layout

```
src/gpu_corruptnet/
  corruptions/   # base types, registry, injectors (the Glitchify-2 generator)
  data/          # procedural demo frame (real frames land in M2)
  utils/         # seeding / reproducibility
scripts/         # make_sanity_grid.py
tests/           # generator unit tests
configs/         # default.yaml (seeds, frame size, class vocab)
```

## License

MIT
