"""README figures from results/ (no model or data needed).

    python scripts/make_results_figures.py

* assets/reliability.png   deployed model, full seen + unseen splits, fixed T from
                           results/calibration_deployed.json (not refit)
* assets/per_class_f1.png  per-class F1 with 95% bootstrap CIs, seen vs unseen content
* assets/sim_to_real.png   STL-10 synthetic -> synthetic on real frames -> real glitches
* assets/latency.png       same-run single-frame latency per backend
* assets/seeds_baselines.png  unseen-content macro-F1 with CIs: seeds and baselines
Each figure is skipped when its inputs are missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gpu_corruptnet.calibration import expected_calibration_error, sigmoid
from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES

# Reference categorical palette (validated: CVD + normal-vision separation, light surface).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.size": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False,
})
OUT = Path("assets")


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_reliability(preds: Path, cal: Path) -> None:
    if not preds.exists() or not cal.exists():
        return
    z, t = np.load(preds), json.loads(cal.read_text())["temperature"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, split, title in zip(axes, ("seen_test", "unseen_test"),
                                ("Seen content", "Unseen content"), strict=True):
        lg, y = z[f"{split}_logits"], z[f"{split}_labels"]
        for probs, color, name in ((sigmoid(lg), ORANGE, "before"),
                                   (sigmoid(lg / t), BLUE, f"after (T = {t:.2f})")):
            ece, bins = expected_calibration_error(probs, y)
            b = [d for d in bins if d["count"] >= 20]
            ax.plot([d["confidence"] for d in b], [d["accuracy"] for d in b], "-o", color=color,
                    lw=2, ms=5, label=f"{name}: ECE {ece:.2%}")
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
        ax.set_title(f"{title} ({len(y)} frames)", fontsize=10, loc="left")
        ax.set_xlabel("predicted P(artifact present)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("observed frequency")
    fig.suptitle("Calibration of the deployed model (pooled over classes, bins with ≥ 20 pairs)",
                 x=0.01, ha="left", fontsize=11)
    save(fig, "reliability.png")


def fig_per_class(summary: dict) -> None:
    pc = summary.get("per_class_f1", {})
    tag = "s1337" if "s1337" in pc else "deployed" if "deployed" in pc else None
    if tag is None:
        return
    names = list(ARTIFACT_CLASSES)
    order = sorted(names, key=lambda c: pc[tag]["unseen_test"][c][0])
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for off, split, color, label in ((0.17, "seen_test", BLUE, "seen content"),
                                     (-0.17, "unseen_test", ORANGE, "unseen content")):
        v = np.array([pc[tag][split][c] for c in order])
        ax.errorbar(v[:, 0], y + off, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o",
                    color=color, ms=6, elinewidth=2, capsize=0, label=label)
    for i, c in enumerate(order):
        u = pc[tag]["unseen_test"][c]
        ax.text(1.045, i - 0.17, f"{u[0]:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.text(1.045, len(order) - 0.45, "unseen", fontsize=8, color=INK2)
    ax.set_yticks(y, [c.replace("_", " ") for c in order])
    ax.set_xlim(0.4, 1.1)
    ax.set_xticks(np.arange(0.4, 1.01, 0.1))
    ax.set_xlabel("F1 (95% bootstrap CI)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left")
    ax.set_title("Per-class F1, ResNet-50" + (" (seed 1337)" if tag == "s1337" else " (deployed)"),
                 loc="left", fontsize=11)
    save(fig, "per_class_f1.png")


def fig_latency(lat: Path) -> None:
    if not lat.exists():
        return
    r = json.loads(lat.read_text())
    rows = sorted(r["backends"].items(), key=lambda kv: kv[1]["p50_ms"], reverse=True)
    labels = {"pytorch_cpu": "PyTorch, CPU", "pytorch_mps": "PyTorch, MPS (GPU)",
              "onnxruntime_cpu": "ONNX Runtime, CPU", "onnxruntime_coreml": "ONNX Runtime, CoreML"}
    fig, ax = plt.subplots(figsize=(7, 2.8))
    vals = [v["p50_ms"] for _, v in rows]
    ax.barh(range(len(rows)), vals, color=BLUE, height=0.6)
    for i, (_, v) in enumerate(rows):
        note = f"{v['p50_ms']:.1f} ms  (max |Δlogit| {v['max_abs_logit_diff']:.3f})"
        ax.text(v["p50_ms"] + 0.6, i, note, va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)), [labels.get(k, k) for k, _ in rows])
    ax.set_xlim(0, max(vals) * 1.9)
    ax.set_xlabel("median single-frame latency, batch 1 (ms)")
    ax.grid(axis="y", visible=False)
    la = r["load_average_before_after"][0][0]
    ax.set_title(f"One run, one Apple M2 (shared, load avg {la:.1f}); same 224 px real frames",
                 loc="left", fontsize=10)
    save(fig, "latency.png")


def fig_seeds_baselines(summary: dict) -> None:
    rows = []
    for k, v in summary.get("seeds", {}).get("runs", {}).items():
        rows.append((f"ResNet-50 {k.removeprefix('r50-')}", v["unseen_test"], BLUE))
    names = {"deployed-eval": "ResNet-50 deployed (Colab T4)", "effb4": "EfficientNet-B4",
             "probe": "ResNet-50 linear probe", "smallcnn": "Small CNN (scratch)",
             "classical": "DFT/HOG + logistic reg."}
    for k, label in names.items():
        if k in summary.get("baselines", {}):
            rows.append((label, summary["baselines"][k]["unseen_test"],
                         BLUE if k == "deployed-eval" else ORANGE))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.2))
    for i, (_label, d, color) in enumerate(rows):
        err = [[d["macro_f1"] - d["ci"][0]], [d["ci"][1] - d["macro_f1"]]]
        ax.errorbar(d["macro_f1"], i, xerr=err, fmt="o", color=color, ms=6, elinewidth=2)
        ax.text(d["ci"][1] + 0.01, i, f"{d['macro_f1']:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("macro-F1 on unseen image content (95% CI)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Seeds (blue) and baselines (orange), identical test frames", loc="left",
                 fontsize=11)
    save(fig, "seeds_baselines.png")


def fig_sim_to_real(summary: dict) -> None:
    rg, rs = summary.get("real_glitches", {}), summary.get("real_synthetic", {})
    seeds = summary.get("baselines", {}).get("deployed-eval")
    if not rg:
        return
    models = [("deployed", "real-deployed-eval", "STL-10 model (deployed)", ORANGE),
              ("native", "real-native", "+ real frames, native-res artifacts", BLUE),
              ("resize_first", "real-resize-first", "+ real frames, artifacts after resize", AQUA)]
    models = [m for m in models if m[0] in rg]
    panels = [("Synthetic artifacts,\nSTL-10 unseen content", "macro-F1"),
              ("Synthetic artifacts,\nheld-out game titles", "macro-F1"),
              ("Real glitches vs clean\nbug-report screenshots", "AUROC"),
              ("False alarms on clean\nCommons game frames", "rate (lower is better)")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for j, (ax, (title, unit)) in enumerate(zip(axes, panels, strict=True)):
        for i, (key, rkey, _label, color) in enumerate(models):
            if j == 0:
                v = seeds["unseen_test"]["macro_f1"] if key == "deployed" and seeds else np.nan
            elif j == 1:
                v = rs.get(rkey, {}).get("unseen_test", {}).get("macro_f1", np.nan)
            elif j == 2:
                v = rg[key]["auroc_glitch_vs_bugreport_clean"]
            else:
                v = rg[key]["fpr_commons_clean"]
            if np.isnan(v):
                continue
            ax.bar(i, v, color=color, width=0.7)
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=INK)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_ylim(0, 1.08)
        ax.set_xticks([])
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=m[3]) for m in models]
    fig.legend(handles, [m[2] for m in models], loc="lower center", ncol=len(models),
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("From synthetic to real: the same detectors on progressively more real test sets",
                 x=0.01, y=1.06, ha="left", fontsize=11)
    save(fig, "sim_to_real.png")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    summary_path = Path("results/summary.json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    fig_reliability(Path("runs/ci/core/deployed-eval/preds_deployed.npz"),
                    Path("results/calibration_deployed.json"))
    fig_per_class(summary)
    fig_latency(Path("results/latency_parity.json"))
    fig_seeds_baselines(summary)
    fig_sim_to_real(summary)


if __name__ == "__main__":
    main()
