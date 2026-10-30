# Calibrated, conformal GPU-artifact detection under content shift

*GPU-CorruptNet technical note. Every number below is regenerated from saved logits by
`scripts/aggregate_results.py`; tables live in [`results/RESULTS.md`](../results/RESULTS.md).*

## Summary

A ResNet-50 trained on procedurally corrupted frames detects ten GPU-artifact types
(the Glitchify taxonomy) at macro-F1 0.935 ± 0.004 on seen content and 0.907 ± 0.010 on
unseen image content (5 seeds). Temperature scaling fixes the pooled calibration error
(ECE 1.06% → 0.26%) but barely moves classwise ECE (1.37% → 1.18%), and a marginal
conformal set that reports 93.5% coverage covers only 90.5% of the frames that actually
contain an artifact, with two classes near 80%. A per-class (Mondrian) conformal set lifts
every class to ≥ 0.84. The largest gap is sim-to-real: on 117 real glitch screenshots
from public bug reports the deployed model separates glitches from clean screenshots at
AUROC 0.52 [0.46, 0.57], i.e. chance, because it flags 78–80% of all real frames;
fine-tuning on 132 real game frames with injected artifacts lifts this to 0.73 [0.68, 0.78] and cuts false alarms on clean game frames from 78% to 14%, but real captured faults (as in VISAPP 2022) are still needed to close the gap.

## 1. Problem and prior work

GPU rendering faults (shader errors, corrupted textures, stuck memory, tearing) are
caught in game and driver QA mostly by people looking at frames. Learned detectors exist,
but labelled real corruption is scarce, so most work trains on injected artifacts:

- **Glitchify** (Davarmanesh et al., AMD + UCLA IPAM, [arXiv 2011.15103](https://arxiv.org/abs/2011.15103))
  defines the ten software-reproducible artifact types used here and reports 84% accuracy
  on seen games and 69% on unseen games with classical features.
- **Chan Chee, Pandit and Kiehn** (University of Toronto + AMD, VISAPP 2022,
  [doi:10.5220/0010788900003124](https://doi.org/10.5220/0010788900003124)) train an
  EfficientNet on *real* corruption captured from 50+ games: precision 0.989, recall 0.888.
  This is the closest prior work and the bar for real-frame performance.
- **Ling, Tollmar and Gisslén** (EA SEED, AIIDE 2020,
  [doi:10.1609/aiide.v16i1.7409](https://doi.org/10.1609/aiide.v16i1.7409)) detect
  rendered texture glitches with a ShuffleNetV2 trained on Unity-generated data (86.8% accuracy).
- **GLIB** (Chen et al., NetEase + UT Dallas, ESEC/FSE 2021,
  [arXiv 2106.10507](https://arxiv.org/abs/2106.10507)) uses code-based augmentation to build a
  test oracle for UI/graphics glitches in mobile games.
- **GlitchBench** (Taesiri, Feng, Bezemer and Nguyen, CVPR 2024,
  [project](https://glitchbench.github.io/)) benchmarks multimodal LLMs on 593 real game
  glitches from r/GamePhysics; the best model reaches 43.4%.

None of these report calibrated probabilities or coverage-guaranteed prediction sets,
which is what a triage queue needs: *how sure is the detector, and which artifact types
should a tester look for?* That question is the focus here.

## 2. Data

**Synthetic training data.** Ten injectors re-implement the Glitchify types (pure NumPy +
OpenCV). Each training frame gets 0 (30%), 1 or 2 artifacts at severity 1–5. The clean
substrate is STL-10 (13,000 labelled 96 px photos). Two object classes (ship, truck) are
held out as *unseen content*: the unseen-content split measures generalisation to image
content the model never saw. It is **not** a test on unseen artifact types (that is §5).
Splits are fixed by a split seed (1337) independent of the training seed, so every model
and baseline is scored on identical frames: 1,560 seen-content and 2,600 unseen-content
test frames, plus a disjoint 1,560-frame calibration split.

*A bug fixed along the way:* each frame used to receive the same corruption every epoch
(the corruption seed ignored the epoch). Corruptions are now re-drawn per epoch; eval
sets keep the original seeds, so the deployed model's numbers are unchanged.

**Real game frames.** 326 free-licence screenshots of 20 open-source games from
Wikimedia Commons (author and licence per file in `realdata/clean_frames.csv`), up to
1920 px wide. Four titles (Veloren, Speed Dreams, Neverball, Battle for Wesnoth) are held out.
Artifacts are injected at the frame's native resolution and the result is resized to
224 px, the way a GPU fault appears before any downscaling.

**Real glitches.** Screenshots attached to public issues of 19 graphics projects (DXVK,
vkd3d-proton, Proton, MoltenVK, RPCS3, PCSX2, PPSSPP, xemu, Flycast, melonDS, shadPS4,
Cemu, Godot, Luanti, SuperTuxKart, Bevy, wgpu, Vita3K, Azahar). Of 679 reviewed images,
117 show a visible rendering glitch and 382 are clean in-game frames from the same
reports (expected-vs-actual pairs, other scenes); 180 were UI, logs, photos of monitors
or unclear and were dropped. Each glitch also has a best-fit Glitchify class and a short
description. Only URLs and labels are committed (`realdata/real_glitches.csv`).

## 3. Classification

All rows use the same 1,560 seen-content and 2,600 unseen-content test frames; 95%
frame-level bootstrap CIs.

| model | epochs | seen macro-F1 | unseen macro-F1 |
|---|---|---|---|
| ResNet-50 fine-tuned, 5 seeds (mean ± std) | 7–8 | 0.935 ± 0.004 | 0.907 ± 0.010 |
| ResNet-50, deployed checkpoint (before the per-epoch fix) | 12 | 0.911 [0.899, 0.921] | 0.876 [0.866, 0.884] |
| EfficientNet-B4 fine-tuned | 10 | 0.926 [0.916, 0.934] | 0.909 [0.901, 0.917] |
| ResNet-50 linear probe (frozen backbone) | 12 | 0.742 [0.722, 0.757] | 0.671 [0.658, 0.684] |
| Small CNN from scratch (4 conv blocks) | 12 | 0.631 [0.616, 0.646] | 0.604 [0.592, 0.616] |
| DFT + HOG + colour features, logistic regression | - | 0.526 [0.509, 0.542] | 0.497 [0.484, 0.510] |

Training ran on free CPU runners with a 200-minute budget per job, which stopped the seed runs
after 7–8 of 12 epochs; the numbers are therefore conservative for the recipe. The seeds beat
the deployed checkpoint even so, which I attribute to re-drawing corruptions every epoch
(the deployed model saw each frame with the same corruption 12 times).

Per-class F1 with bootstrap CIs (unseen content, seed 1337) shows where the model is weak:
screen stuttering 0.70, shapes 0.82, screen tearing 0.83 and discoloration 0.83 are the weak classes; line, dot and triangulation artifacts are all ≥ 0.96. Screen tearing on unseen content is the weakest class.

## 4. Uncertainty

**Calibration.** One temperature T fitted on the calibration split (deployed model:
T = 1.565). Pooled ECE treats every (frame, class) pair as one event; with ten classes and
mostly-absent artifacts, about 90% of pairs are negatives sitting near p = 0, which hides
per-class miscalibration. Classwise ECE (the mean of each class's own ECE) and the Brier
score tell a less flattering story:

| deployed model | pooled ECE | classwise ECE | Brier |
|---|---|---|---|
| seen content, before → after T | 1.06% → 0.26% | 1.37% → 1.18% | 0.0146 → 0.0142 |
| unseen content, before → after T | 1.42% → 0.17% | 1.67% → 1.39% | 0.0200 → 0.0194 |

**Conformal label sets.** The set is {classes with calibrated p ≥ τ}, with τ fitted on
calibration frames so that all of a frame's true artifacts are in the set with probability
≥ 90% (full-inclusion split conformal; α = 0.1). Two corrections to how this was reported:

1. *Clean frames are covered vacuously* (no true labels), so averaging over all frames
   inflates coverage. On corrupted frames only, marginal coverage is **0.905 seen /
   0.895 unseen** (reported before as 0.935 / 0.927).
2. *Marginal coverage hides per-class failures.* Conditional on a class being present,
   the marginal set covers shapes only 0.815 (seen) / 0.782 (unseen) of the time and
   screen tearing 0.816 on unseen content.

A Mondrian (class-conditional) set fits one threshold per class on the calibration frames
where that class is present. It lifts the worst class to 0.844 seen / 0.878 unseen (all
others ≥ 0.87), at a cost: sets on corrupted frames grow from 2.07 to 2.23 classes, and
clean frames get non-empty sets more often (0.07 → 0.41 classes on average). Per-class
validity does not give joint validity for two-artifact frames: full-inclusion coverage of
the Mondrian set on corrupted frames is 0.861. That trade-off, plus finite-sample noise
(≈ 150 calibration positives per class), is why a deployment should report both numbers.

## 5. Artifact types the model never saw

To test genuinely novel artifact types, the classifier is retrained ten times, each time
without one type (6-epoch budget; 4–6 epochs completed), and scored on test frames whose only
artifact is the held-out type (127–165 frames per type). Recall = share still flagged as corrupted:

| held-out type | recall when held out | recall when trained on it | anomaly-head AUROC |
|---|---|---|---|
| triangulation | 1.000 | 1.000 | 0.997 |
| dotted lines | 0.994 | 1.000 | 0.989 |
| parallel lines | 0.970 | 1.000 | 0.990 |
| morse code | 0.852 | 0.992 | 0.911 |
| line pixelation | 0.841 | 1.000 | 0.935 |
| shader | 0.691 | 0.976 | 0.747 |
| discoloration | 0.550 | 0.954 | 0.658 |
| screen stuttering | 0.446 | 0.993 | 0.917 |
| shapes | 0.142 | 0.929 | 0.571 |
| screen tearing | 0.014 | 0.993 | 0.701 |

High-frequency structured artifacts transfer: a model that never saw triangulation still flags
all of it. Region and displacement artifacts do not: a novel tearing or shapes fault is almost
never flagged. The anomaly head is complementary for screen stuttering (0.92) and partly for
tearing (0.70), and fails on shapes in both.

The label-free anomaly head (kNN on ImageNet ResNet-50 features of clean frames) never
sees any corruption, so every type is novel to it. Overall AUROC is 0.835, but it ranges
from 0.571 (shapes) and 0.658 (discoloration) to 0.99+ (dotted lines, parallel lines,
triangulation): structured high-frequency artifacts are easy, colour and region artifacts
are not.

## 6. Out-of-distribution checks on the model's own embedding

A PCA-64 Mahalanobis score on the penultimate embedding, with the training frames as the
reference set and the 99th percentile of seen-content scores as threshold, separates
clean from corrupted seen frames at AUROC 0.844 and seen from unseen *content* at only
0.598. The model generalises across content, so the embedding barely moves. More
important, it flags **0% of real game frames** (AUROC 0.47 vs seen content) even though
the classifier false-alarms on 78% of them. The detector embeds real frames next to the
corrupted training frames, which are part of the reference. An OOD monitor whose
reference set includes the positive class cannot catch a shift that makes everything look
positive. A clean-only reference or a domain classifier on real-vs-synthetic frames is
the fix to try next.

## 7. Sim-to-real

Three detectors, three test sets of increasing realism:

| model | synthetic, STL-10 unseen content | synthetic on held-out game titles | real glitches vs clean bug-report screenshots (AUROC) | false alarms on clean Commons frames |
|---|---|---|---|---|
| deployed (STL-10 only) | 0.876 | 0.335 | 0.52 [0.46, 0.57] | 78% |
| + real frames, artifacts at native resolution | - | 0.694 | 0.73 [0.68, 0.78] | 14% |
| + real frames, artifacts painted after resize | - | 0.879* | 0.77 [0.72, 0.82] | 6% |

*Scored on its own resize-first test pipeline, so not comparable with the row above; the
real-glitch column is the fair comparison.

Fine-tuning from the STL-10 checkpoint on 132 real game frames (8 epochs, random crops, 8
corruption draws per frame) recovers 21–25 AUROC points on real glitches and removes most false
alarms. At the 0.5 threshold, recall on real glitches falls from 0.92 (the deployed model flags
almost everything) to 0.68 (native) and 0.44 (resize-first); bug-report F1 rises from 0.41 to
0.53 and 0.49. Whether artifacts are painted at native resolution or after the resize made no
significant difference in ranking (CIs overlap); native keeps more recall at the default
threshold. The remaining gap is the one VISAPP 2022 closed with real captured corruption:
procedural injectors do not look like most real driver faults (missing geometry, garbage
textures, wrong buffers), which my labels mostly map to *shapes* and *discoloration*, the two
types the synthetic model also generalises to worst.

## 8. Serving

One run on one Apple M2 (shared machine, load average ≈ 7–12), batch 1, 224 px real frames,
parity against PyTorch-CPU logits on the same frames:

| backend | p50 | max abs logit diff | label sets agree |
|---|---|---|---|
| ONNX Runtime, CoreML EP | 1.8 ms | 0.090 | 100% |
| PyTorch, MPS | 18.2 ms | 0.0001 | 100% |
| PyTorch, CPU | 33.2 ms | 0 | 100% |
| ONNX Runtime, CPU | 40.5 ms | 0 | 100% |

The always-on demo runs the same network in the browser (ONNX Runtime Web, WASM): weights
stored as fp16 (47 MB) and cast to fp32 at load, max logit difference 0.005 vs PyTorch,
about 0.4 s per frame single-threaded. A T4 / TensorRT row needs an NVIDIA GPU; the Colab
notebook has the cell.

## 9. Limitations

- The real-glitch labels come from one reviewer looking at screenshots at reduced size;
  subtle glitches (a wrong shadow) are likely labelled clean, so recall on real glitches is
  measured on the *visible* ones. Labels are in the CSV for anyone to audit.
- Bug-report screenshots are a convenience sample (emulators and translation layers
  dominate) and include UI overlays; they are not a capture campaign like VISAPP 2022's.
- The cross-generator test against AMD's original Glitchify code was not run: that
  repository has no licence.
- Training ran on CPU-only CI runners; seeds were limited to 5 and baselines to one seed.

## Reproduce

```bash
pip install -e ".[cv,train,export,test]"
python scripts/fetch_stl10.py data                      # md5-verified
python realdata/fetch.py realdata/clean_frames.csv data/real_frames
python realdata/fetch.py realdata/real_glitches.csv data/real_glitches
gh workflow run experiments.yml -f suite=core            # or loto / real; free CPU runners
python scripts/aggregate_results.py && python scripts/make_results_figures.py
```
