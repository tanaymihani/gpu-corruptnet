"""Aggregate experiment artifacts into results/summary.json and results/RESULTS.md.

    gh run download <core-run-id> -D runs/ci/core     # likewise loto, real
    python scripts/eval_real.py deployed=model.pt native=... resize_first=...
    python scripts/aggregate_results.py --ci runs/ci --out results

Every number is recomputed from saved logits + labels (preds_*.npz), so the tables can
be regenerated without re-training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gpu_corruptnet.calibration import ConformalLabelSets, TemperatureScaler
from gpu_corruptnet.calibration import expected_calibration_error as pooled_ece
from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES
from gpu_corruptnet.evaluation import (
    EmbeddingOOD,
    MondrianConformal,
    auroc,
    binary_f1,
    bootstrap_ci,
    brier,
    classwise_ece,
    macro_f1,
    per_class_f1_ci,
    probs_from_logits,
    set_coverage,
)

CLASSES = list(ARTIFACT_CLASSES)
SPLITS = ("seen_test", "unseen_test")


def load(d: Path) -> tuple[dict, dict] | None:
    ms, ps = sorted(d.glob("metrics_*.json")), sorted(d.glob("preds_*.npz"))
    if not ms or not ps:
        return None
    return json.loads(ms[-1].read_text()), dict(np.load(ps[-1]))


def scores(z: dict, split: str, t: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    return probs_from_logits(z[f"{split}_logits"], t), z[f"{split}_labels"]


def f1_block(z: dict, n_boot: int) -> dict:
    out = {}
    for s in SPLITS:
        p, y = scores(z, s)
        point, lo, hi = bootstrap_ci(macro_f1, y, p, n_boot=n_boot)
        out[s] = {"macro_f1": point, "ci": [lo, hi], "binary_f1": binary_f1(y, p), "n": len(y)}
    return out


def uncertainty_block(z: dict, alpha: float = 0.1) -> dict:
    ts = TemperatureScaler().fit(z["cal_logits"], z["cal_labels"])
    t = ts.temperature
    pc, yc = scores(z, "cal", t)
    marginal = ConformalLabelSets(alpha).fit(pc, yc)
    mondrian = MondrianConformal(alpha).fit(pc, yc)
    out = {"temperature": t, "alpha": alpha, "marginal_tau": marginal.tau,
           "mondrian_tau": dict(zip(CLASSES, mondrian.tau.tolist(), strict=True))}
    for s in SPLITS:
        p0, y = scores(z, s)
        p1, _ = scores(z, s, t)
        cw0, _ = classwise_ece(p0, y)
        cw1, per1 = classwise_ece(p1, y)
        out[s] = {
            "pooled_ece": [pooled_ece(p0, y)[0], pooled_ece(p1, y)[0]],
            "classwise_ece": [cw0, cw1],
            "classwise_ece_per_class": dict(zip(CLASSES, per1, strict=True)),
            "brier": [brier(p0, y), brier(p1, y)],
            "marginal_conformal": set_coverage(marginal.predict_set(p1), y),
            "mondrian_conformal": set_coverage(mondrian.predict_set(p1), y),
        }
    return out


def loto_block(ci: Path, ref: dict | None) -> dict:
    out = {}
    for k, name in enumerate(CLASSES):
        r = load(ci / "loto" / f"loto-{name}")
        if r is None:
            continue
        m, z = r
        p = np.concatenate([scores(z, s)[0] for s in SPLITS])
        y = np.concatenate([scores(z, s)[1] for s in SPLITS])
        single = y.sum(1) == 1
        held, other, clean = single & (y[:, k] == 1), single & (y[:, k] == 0), y.sum(1) == 0
        flagged = (p >= 0.5).any(1)
        row = {
            "recall_heldout_type": float(flagged[held].mean()),
            "recall_other_types": float(flagged[other].mean()),
            "false_alarm_clean": float(flagged[clean].mean()),
            "n_heldout_frames": int(held.sum()),
            "epochs": m.get("epochs_completed"),
        }
        if ref is not None:
            pr = np.concatenate([scores(ref, s)[0] for s in SPLITS])
            row["recall_when_trained_on_it"] = float((pr >= 0.5).any(1)[held].mean())
        out[name] = row
    return out


def ood_block(z: dict, real_npz: Path | None) -> dict:
    det = EmbeddingOOD(k=64).fit(z["train_ref_feats"])
    seen = det.score(z["seen_test_feats"])
    unseen = det.score(z["unseen_test_feats"])
    corrupted = z["seen_test_labels"].sum(1) > 0
    thr = float(np.percentile(seen, 99))
    out = {
        "threshold_p99_seen": thr,
        "auroc_seen_vs_unseen_content": auroc(seen, unseen),
        "auroc_clean_vs_corrupted_seen": auroc(seen[~corrupted], seen[corrupted]),
        "flag_rate": {"seen_test": float((seen > thr).mean()),
                      "unseen_test": float((unseen > thr).mean())},
    }
    if real_npz and real_npz.exists():
        r = np.load(real_npz)
        emb = r["embeddings"].astype(np.float64)
        for src in ("commons", "bugreport"):
            sc = det.score(emb[r["source"] == src])
            out[f"auroc_seen_vs_real_{src}"] = auroc(seen, sc)
            out["flag_rate"][f"real_{src}"] = float((sc > thr).mean())
    return out


def real_glitch_block(npz: Path, n_boot: int = 1000) -> dict:
    r = np.load(npz)
    score = r["logits"].max(1)
    flagged = score >= 0
    g, src = r["is_glitch"], r["source"]
    bug_clean = (src == "bugreport") & ~g
    com = src == "commons"
    rng = np.random.default_rng(0)
    gi, ci_ = np.where(g)[0], np.where(bug_clean)[0]
    boots = [auroc(score[rng.choice(ci_, len(ci_))], score[rng.choice(gi, len(gi))])
             for _ in range(n_boot)]
    return {
        "n_glitch": int(g.sum()), "n_bugreport_clean": int(bug_clean.sum()), "n_commons": int(com.sum()),
        "glitch_recall": float(flagged[g].mean()),
        "fpr_bugreport_clean": float(flagged[bug_clean].mean()),
        "fpr_commons_clean": float(flagged[com].mean()),
        "auroc_glitch_vs_bugreport_clean": auroc(score[bug_clean], score[g]),
        "auroc_ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "auroc_glitch_vs_commons": auroc(score[com], score[g]),
    }


def fmt_ci(d: dict) -> str:
    return f"{d['macro_f1']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", default="runs/ci")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    ci, out = Path(args.ci), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    S: dict = {}
    md: list[str] = ["# Results", "",
                     "Generated by `scripts/aggregate_results.py` from saved logits. "
                     "95% CIs are frame-level bootstrap (1000 resamples).", ""]

    # A. seeds
    seeds = {}
    for d in sorted((ci / "core").glob("r50-s*")):
        r = load(d)
        if r:
            seeds[d.name] = {**f1_block(r[1], args.n_boot), "epochs": r[0].get("epochs_completed"),
                             "train_minutes": r[0].get("train_minutes")}
    if seeds:
        agg = {}
        for s in SPLITS:
            v = np.array([seeds[k][s]["macro_f1"] for k in seeds])
            b = np.array([seeds[k][s]["binary_f1"] for k in seeds])
            agg[s] = {"macro_f1_mean": float(v.mean()), "macro_f1_std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                      "binary_f1_mean": float(b.mean()), "n_seeds": len(v)}
        S["seeds"] = {"runs": seeds, "aggregate": agg}
        md += ["## ResNet-50, 5 seeds (fine-tuned, 12 epochs, 224 px, CPU runners)", "",
               "| run | epochs | seen macro-F1 [95% CI] | unseen macro-F1 [95% CI] | seen bin-F1 | unseen bin-F1 |",
               "|---|---|---|---|---|---|"]
        for k, v in seeds.items():
            md.append(f"| {k} | {v['epochs']} | {fmt_ci(v['seen_test'])} | {fmt_ci(v['unseen_test'])} | "
                      f"{v['seen_test']['binary_f1']:.3f} | {v['unseen_test']['binary_f1']:.3f} |")
        a = agg
        md += [f"| **mean ± std** | | **{a['seen_test']['macro_f1_mean']:.3f} ± {a['seen_test']['macro_f1_std']:.3f}** | "
               f"**{a['unseen_test']['macro_f1_mean']:.3f} ± {a['unseen_test']['macro_f1_std']:.3f}** | "
               f"{a['seen_test']['binary_f1_mean']:.3f} | {a['unseen_test']['binary_f1_mean']:.3f} |", ""]

    # B. baselines (+ the deployed checkpoint re-scored on the same splits)
    base = {}
    for name in ("deployed-eval", "probe", "smallcnn", "effb4", "classical"):
        r = load(ci / "core" / name)
        if r:
            base[name] = {**f1_block(r[1], args.n_boot), "epochs": r[0].get("epochs_completed"),
                          "stopped_early": r[0].get("stopped_early")}
    S["baselines"] = base
    if base:
        md += ["## Baselines (same test frames)", "",
               "| model | epochs | seen macro-F1 [95% CI] | unseen macro-F1 [95% CI] | unseen bin-F1 |",
               "|---|---|---|---|---|"]
        for k, v in base.items():
            md.append(f"| {k} | {v['epochs'] or '-'} | {fmt_ci(v['seen_test'])} | {fmt_ci(v['unseen_test'])} | "
                      f"{v['unseen_test']['binary_f1']:.3f} |")
        md.append("")

    # C. per-class F1 with CIs, uncertainty
    ref = load(ci / "core" / "deployed-eval")
    s1337 = load(ci / "core" / "r50-s1337")
    for tag, r in (("deployed", ref), ("s1337", s1337)):
        if r is None:
            continue
        z = r[1]
        S.setdefault("per_class_f1", {})[tag] = {
            s: dict(zip(CLASSES, per_class_f1_ci(scores(z, s)[1], scores(z, s)[0],
                                                 n_boot=args.n_boot).tolist(), strict=True))
            for s in SPLITS}
        S.setdefault("uncertainty", {})[tag] = uncertainty_block(z)
    for tag in S.get("uncertainty", {}):
        u = S["uncertainty"][tag]
        md += [f"## Uncertainty ({tag}; T = {u['temperature']:.4f}, alpha = {u['alpha']})", "",
               "| split | pooled ECE before→after | classwise ECE before→after | Brier before→after | "
               "marginal cov. all / corrupted-only | Mondrian min-class cov. | set size (corrupted) |",
               "|---|---|---|---|---|---|---|"]
        for s in SPLITS:
            x = u[s]
            mc, mo = x["marginal_conformal"], x["mondrian_conformal"]
            md.append(f"| {s} | {x['pooled_ece'][0]:.4f}→{x['pooled_ece'][1]:.4f} | "
                      f"{x['classwise_ece'][0]:.4f}→{x['classwise_ece'][1]:.4f} | "
                      f"{x['brier'][0]:.4f}→{x['brier'][1]:.4f} | "
                      f"{mc['coverage_all_frames']:.3f} / {mc['coverage_corrupted_frames']:.3f} | "
                      f"{mo['min_class_coverage']:.3f} | {mc['avg_set_size_corrupted']:.2f} / {mo['avg_set_size_corrupted']:.2f} |")
        md.append("")
    if "per_class_f1" in S:
        tag = "s1337" if "s1337" in S["per_class_f1"] else "deployed"
        pc = S["per_class_f1"][tag]
        md += [f"## Per-class F1 ({tag}) with 95% CIs", "", "| class | seen | unseen |", "|---|---|---|"]
        for c in CLASSES:
            a, b = pc["seen_test"][c], pc["unseen_test"][c]
            md.append(f"| {c} | {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] | {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}] |")
        md.append("")

    # D. leave-one-type-out
    loto = loto_block(ci, s1337[1] if s1337 else None)
    an = ci / "core" / "anomaly-per-type" / "anomaly_per_type.json"
    anomaly = json.loads(an.read_text()) if an.exists() else {}
    S["loto"], S["anomaly"] = loto, anomaly
    if loto:
        md += ["## Leave one corruption type out", "",
               "Classifier retrained without type k (6 epochs); recall = share of frames whose only "
               "artifact is k that are still flagged as corrupted. Anomaly AUROC: clean vs "
               "type-k frames for the label-free kNN head.", "",
               "| held-out type | recall on held-out type | recall on its other types | "
               "recall when trained on it | anomaly-head AUROC |", "|---|---|---|---|---|"]
        for c, v in loto.items():
            a = anomaly.get("per_type_auroc", {}).get(c)
            ref_rec = v.get("recall_when_trained_on_it", float("nan"))
            md.append(f"| {c} | {v['recall_heldout_type']:.3f} | {v['recall_other_types']:.3f} | "
                      f"{ref_rec:.3f} | {'-' if a is None else f'{a:.3f}'} |")
        md.append("")

    # E. OOD on embeddings
    if ref:
        S["ood"] = ood_block(ref[1], Path(args.runs) / "real_eval_deployed.npz")
        o = S["ood"]
        md += ["## OOD on model embeddings (deployed model, PCA-64 Mahalanobis, reference = training frames)", "",
               "| comparison | AUROC |", "|---|---|"]
        for k, v in o.items():
            if k.startswith("auroc"):
                md.append(f"| {k.removeprefix('auroc_').replace('_', ' ')} | {v:.3f} |")
        md += ["", "Flag rate at the 99th-percentile threshold: " +
               ", ".join(f"{k} {v:.1%}" for k, v in o["flag_rate"].items()), ""]

    # F. real frames
    real = {}
    for name in ("real-deployed-eval", "real-native", "real-resize-first"):
        r = load(ci / "real" / name)
        if r:
            real[name] = f1_block(r[1], args.n_boot)
    S["real_synthetic"] = real
    rg = {}
    for p in sorted(Path(args.runs).glob("real_eval_*.npz")):
        rg[p.stem.removeprefix("real_eval_")] = real_glitch_block(p, args.n_boot)
    S["real_glitches"] = rg
    if real:
        md += ["## Synthetic artifacts on real game frames (native resolution; held-out titles = unseen)", "",
               "| model | seen-title macro-F1 [95% CI] | unseen-title macro-F1 [95% CI] | unseen bin-F1 |",
               "|---|---|---|---|"]
        for k, v in real.items():
            md.append(f"| {k} | {fmt_ci(v['seen_test'])} | {fmt_ci(v['unseen_test'])} | "
                      f"{v['unseen_test']['binary_f1']:.3f} |")
        md.append("")
    if rg:
        md += ["## Real glitches from public bug reports (hand-labelled)", "",
               "| model | glitch recall | false alarms: bug-report clean / Commons clean | "
               "AUROC glitch vs bug-report clean [95% CI] |", "|---|---|---|---|"]
        for k, v in rg.items():
            md.append(f"| {k} | {v['glitch_recall']:.3f} | {v['fpr_bugreport_clean']:.3f} / "
                      f"{v['fpr_commons_clean']:.3f} | {v['auroc_glitch_vs_bugreport_clean']:.3f} "
                      f"[{v['auroc_ci'][0]:.3f}, {v['auroc_ci'][1]:.3f}] |")
        n = next(iter(rg.values()))
        md += ["", f"{n['n_glitch']} glitch screenshots, {n['n_bugreport_clean']} clean bug-report "
               f"screenshots, {n['n_commons']} clean Commons frames.", ""]

    (out / "summary.json").write_text(json.dumps(S, indent=2))
    (out / "RESULTS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
