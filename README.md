# GPU-CorruptNet

[![Live demo](https://img.shields.io/badge/live-demo-brightgreen)](https://gpu-corruptnet-7asy5rnjkbnckvbzcuvoj8.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Tests](https://img.shields.io/badge/tests-87%20passing-brightgreen.svg)

GPU-CorruptNet detects and classifies GPU-rendered visual corruption in rendered frames:
shader glitches, screen tearing, texture and block artifacts, discoloration, and stuck-memory
"morse-code" patterns. There's a [live demo](https://gpu-corruptnet-7asy5rnjkbnckvbzcuvoj8.streamlit.app)
where you can upload a frame, inject a corruption, and see the detection with calibrated confidence.

It pairs a supervised multi-label classifier (PyTorch) with an unsupervised anomaly head for
corruption types the classifier never saw. Both train on a synthetic corruption generator I built,
and inference runs behind a benchmarked serving path with calibrated, conformal confidence.

The 10 artifact classes come from the AMD/UCLA-RIPS *Glitchify* paper,
[*Automating Artifact Detection in Video Games*](https://arxiv.org/abs/2011.15103), which
injects software-reproducible GPU artifacts procedurally because real corruption data is scarce.
That 2020 work used classical features (84% on seen games, 69% on unseen); this project uses a
deep CNN, adds an anomaly head for novel corruptions, and quantifies uncertainty.

## The synthetic corruption generator ("Glitchify-2")

Real GPU-corruption datasets barely exist, so the generator makes labeled data by injecting
artifacts into clean frames. The grid below shows one procedural frame under each injector
(`python scripts/make_sanity_grid.py`):

![Sanity grid of synthetic corruptions](assets/sanity_grid.png)

## Status

Metrics in this README come from real runs (see Results), not estimates.

| Milestone | Scope | State |
|---|---|---|
| M0 | Repo scaffold, config, tests, CI | done |
| M1 | Glitchify-2 generator, 10 artifact classes (ImageNet-C wrapper planned) | done (10/10 injectors) |
| M2 | Corruption dataset, seen/unseen splits, PostgreSQL/MongoDB stores | done |
| M3 | ResNet-50 multi-label classifier + metrics | done (see Results) |
| M4 | Unsupervised anomaly head (feature-kNN / PatchCore-lite), AUROC 0.83 | done |
| M5 | Temperature scaling, ECE, split-conformal label sets | done (see Results) |
| M6 | C++/libtorch inference path (~1.4x vs Python), ROCm/HIP notes | done |
| M7 | Latency harness, ONNX export, ORT speedup (FP16/TensorRT on GPU) | core done |
| M8 | AWS deploy (S3 + EC2-Spot) + FastAPI demo | planned |
| M9 | Drift (PSI) + OOD (Mahalanobis) monitor | done |
| M10 | Streamlit upload-a-frame demo | done |
| Extra | Source-code analysis (AST static metrics) | done |

## Results (measured)

ResNet-50, fine-tuned 12 epochs at 224px, on-the-fly Glitchify-2 corruptions over an STL-10
substrate (free Colab T4). Unseen content means whole object classes held out of training.

| Split | macro-F1 | binary F1 | binary recall | ECE (temp-scaled) | conformal cov. @90% (avg set) |
|---|---|---|---|---|---|
| seen content | 0.911 | 0.991 | 0.982 | 1.06% to 0.26% | 0.935 (1.43) |
| unseen content | 0.876 | 0.971 | 0.971 | 1.42% to 0.17% | 0.927 (1.70) |

Inference (ONNX Runtime): single-frame latency drops from 10.4 ms to 1.4 ms (~7.6x via the
CoreML execution provider vs PyTorch-eager), 642 img/s batched. You can reproduce all of this
with [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb).

The three figures below regenerate from the checkpoint alone
(`python scripts/make_report_figures.py <model.pt>`):

| Per-class F1 (seen vs unseen) | Calibration reliability |
|---|---|
| ![Per-class F1](assets/per_class_f1.png) | ![Reliability diagram](assets/reliability.png) |

Grad-CAM shows which region triggered each corruption flag:

![Grad-CAM saliency](assets/gradcam.png)

## Generator

All 10 injectors are implemented. Six are pure NumPy: `screen_tearing`, `screen_stuttering`,
`morse_code`, `discoloration`, `parallel_lines`, `dotted_lines`. The OpenCV batch
(`pip install -e ".[cv]"`) adds `shader`, `shapes`, `triangulation`, `line_pixelation`. The
OpenCV injectors register only when OpenCV is installed, so `available()` reflects what's
actually importable.

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

## Training the classifier (M3)

The classifier predicts which corruption(s) are present (multi-label) on clean frames corrupted
on the fly by Glitchify-2. The clean substrate is STL-10, with two object classes (ship, truck)
held out as unseen content so the unseen-test macro-F1 measures generalization to content the
model never trained on.

```bash
pip install -e ".[cv,train]"

python scripts/train_classifier.py --smoke                       # fast local sanity run
python scripts/train_classifier.py --arch resnet50 --epochs 10 --img-size 224
```

The first run downloads STL-10 (~2.6 GB, cached after that). For a real run, use a free GPU:
open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) in Google Colab (Runtime, T4).
Each run writes metrics to `runs/metrics_*.json`.

## Calibration and conformal prediction (M5)

Training writes raw logits for a disjoint calibration split and the test splits to
`runs/preds_*.npz`. `calibrate.py` fits temperature scaling and a split-conformal threshold on
the calibration split, then reports ECE before/after and the conformal set's empirical coverage
against its target:

```bash
python scripts/calibrate.py runs/preds_resnet50_<stamp>.npz --alpha 0.1
```

For each frame the conformal procedure returns a set of candidate artifact types with a coverage
guarantee: `P(true artifacts ⊆ predicted set) ≥ 1 − α`. APS/RAPS are single-label multiclass
methods, and corruption detection is multi-label, so I use a full-inclusion split-conformal
construction instead (see [`calibration.py`](src/gpu_corruptnet/calibration.py)).

## Latency and throughput benchmark (M7)

The harness measures latency carefully: warmup first, then device-appropriate synchronized
timing (CUDA events on GPU, `synchronize` + `perf_counter` on MPS/CPU), reported as p50/p95/p99
and throughput across a batch-size sweep.

```bash
python scripts/benchmark.py --arch resnet50 --img-size 224
```

Latency depends on the architecture and input size rather than the trained weights, so these
numbers are meaningful before training finishes. Results go to `runs/bench_*.json`.

ONNX export and optimized inference (`pip install -e ".[export]"`): export to ONNX and compare
PyTorch eager against ONNX Runtime, which uses whatever execution providers the machine offers
(CoreML on Apple; CUDA/TensorRT with FP16/INT8 on NVIDIA for a larger gain):

```bash
python scripts/export_benchmark.py --arch resnet50 --img-size 224
```

## Interactive demo (M10)

```bash
pip install -e ".[cv,train,export,demo]"
streamlit run scripts/demo_app.py
```

Pick or upload a frame, optionally inject a corruption, and see the detected artifact types,
calibrated confidence, the conformal set, an OOD score, and inference latency. It runs in
generator + drift mode with no model; drop a trained `runs/model_*.pt` in for full detection
(training saves one automatically).

The hosted demo runs on [Streamlit Community Cloud](https://share.streamlit.io), pointed at this
repo with main file `streamlit_app.py`. It pulls the trained model from the repo's GitHub Release
at startup, so there's no local setup.

## Metadata stores: PostgreSQL and MongoDB (M2)

Structured run and metric records go to PostgreSQL (SQLAlchemy, portable to SQLite for dev).
Flexible per-image artifact annotations and model-version documents go to MongoDB (pymongo).
Both are optional and import-light; the core pipeline never requires a database.

```bash
docker compose up -d          # start Postgres + Mongo locally

# log a training run's metrics to Postgres:
python scripts/train_classifier.py --db-url postgresql+psycopg2://corruptnet:corruptnet@localhost/gpu_corruptnet
```

Tests run against in-memory SQLite and `mongomock`, so CI needs no servers.

## Drift and OOD monitor (M9)

When production frames drift away from the training distribution, model quality degrades
quietly. `DriftMonitor` flags this with per-feature PSI for distribution drift and a Mahalanobis
score for per-frame novelty. It's model-free by default (cheap image descriptors), or you can
feed model embeddings once the classifier is trained.

```bash
python scripts/drift_demo.py
# clean (in-distribution)  -> max_psi=0.23  ood_rate=0%    (below the corruption signal)
# corrupted (production)   -> max_psi=6.77  ood_rate=75%   significant_drift
```

## C++ / libtorch inference path (M6)

A native C++ serving path ([`cpp/`](cpp/)) loads a TorchScript export and runs the classifier at
21.7 ms/frame versus 30.7 ms in Python (~1.4x) on CPU, mostly by dropping interpreter overhead.
It links against the libtorch bundled in the installed PyTorch, so the ABI matches and there's
no separate download. On AMD/ROCm the same code targets GPUs unchanged, since `torch::kCUDA`
maps to HIP. See [`cpp/README.md`](cpp/README.md).

## Unsupervised anomaly head (M4)

Trained on clean frames only (no corruption labels), this head flags novel corruption by kNN
distance to a coreset of pretrained-backbone features, a lightweight image-level PatchCore. On
held-out STL-10 it separates clean from corrupted at AUROC 0.83, catching corruption types the
supervised head never trained on.

```bash
python scripts/anomaly_eval.py
```

## Source-code analysis

A static analyzer that treats the project's own Python as data: AST cyclomatic complexity,
docstring coverage, and complexity hotspots.

```bash
python scripts/code_report.py --json runs/code_report.json
# 26 files, 1431 LOC, 94 functions, docstring coverage 28%, avg complexity 2.5
```

## Design notes

- Images are `(H, W, 3)` `uint8` RGB throughout. Severity is `1..5`.
- Injectors self-register (`@register`) and never mutate their input. Each is deterministic given
  a seed (enforced in tests), so every generated dataset is reproducible.
- The base install stays light (NumPy/Pillow). Heavier dependencies come with the milestone that
  needs them, as the `[cv]`, `[train]`, `[export]`, `[db]` extras in `pyproject.toml`.

## Layout

```
src/gpu_corruptnet/
  corruptions/   # base types, registry, injectors (the Glitchify-2 generator)
  data/          # clean-frame sources (STL-10) + on-the-fly corruption dataset
  models/        # ResNet-50 / EfficientNet-B4 multi-label classifier
  utils/         # seeding / reproducibility
  metrics.py     # multi-label F1 / recall, derived binary corrupted-vs-clean
  calibration.py # temperature scaling, ECE, split-conformal label sets (M5)
  bench.py       # latency/throughput harness (warmup + synced timing) (M7)
  export.py      # ONNX export + ONNX Runtime speedup benchmark (M7b)
  drift.py       # PSI distribution drift + Mahalanobis OOD monitor (M9)
  anomaly.py     # unsupervised anomaly head: kNN on deep features (M4)
  codeanalysis.py# AST static source-code analysis
  serve.py       # single-frame inference core: detection + calibration + OOD (M10)
  db/            # PostgreSQL (SQLAlchemy) + MongoDB (pymongo) metadata stores (M2)
  train.py       # training loop + seen/unseen eval, writes runs/metrics_*.json + preds_*.npz
scripts/         # sanity grid, train, calibrate, benchmark, anomaly_eval, code_report, figures, demo
cpp/             # C++ / libtorch inference path + CMake (M6)
docker-compose.yml  # local Postgres + Mongo
notebooks/       # train_colab.ipynb (free-T4 run)
tests/           # generator unit tests
configs/         # default.yaml (seeds, frame size, class vocab)
```

## License

MIT
