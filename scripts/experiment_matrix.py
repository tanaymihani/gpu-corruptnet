"""Print the GitHub Actions job matrix for one experiment suite (used by experiments.yml).

    python scripts/experiment_matrix.py core     # 5 seeds + baselines + deployed re-eval
    python scripts/experiment_matrix.py loto     # leave-one-corruption-type-out
    python scripts/experiment_matrix.py real     # real game frames, native-resolution artifacts
    python scripts/experiment_matrix.py smoke    # minutes-long pipeline check

Every entry is {"name", "cmd", "keep_model"}; the CPU runners are free for public repos.
"""

from __future__ import annotations

import json
import sys

from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES

TRAIN = "python scripts/train_classifier.py"
# The authoritative recipe (notebooks/train_colab.ipynb): fine-tune 12 epochs at 224 px.
FT = (
    f"{TRAIN} --arch resnet50 --epochs 12 --img-size 224 --batch-size 64 --lr 1e-4 "
    "--no-freeze --num-workers 2 --max-train-minutes 320"
)
REAL = (
    f"{TRAIN} --substrate real --arch resnet50 --epochs 8 --img-size 224 --batch-size 32 "
    "--lr 5e-5 --no-freeze --num-workers 3 --init-ckpt model.pt --max-train-minutes 300"
)


def suite(name: str) -> list[dict]:
    if name == "smoke":
        return [
            {"name": "smoke-r50", "keep_model": False,
             "cmd": f"{TRAIN} --smoke --no-freeze --lr 1e-4 --tag smoke --save-feats"},
            {"name": "smoke-classical", "keep_model": False,
             "cmd": "python scripts/baseline_classical.py --n-seen 400 --n-unseen 100"},
        ]
    if name == "core":
        seeds = [
            {"name": f"r50-s{s}", "keep_model": s == 1337,
             "cmd": f"{FT} --seed {s} --tag s{s}" + (" --save-feats" if s == 1337 else "")}
            for s in (1337, 1, 2, 3, 4)
        ]
        return seeds + [
            {"name": "probe", "keep_model": False,
             "cmd": f"{TRAIN} --arch resnet50 --epochs 12 --img-size 224 --batch-size 64 "
                    "--lr 1e-3 --num-workers 2 --tag probe"},
            {"name": "smallcnn", "keep_model": False,
             "cmd": f"{TRAIN} --arch small_cnn --epochs 12 --img-size 224 --batch-size 64 "
                    "--lr 1e-3 --num-workers 2 --tag smallcnn"},
            {"name": "effb4", "keep_model": False,
             "cmd": f"{TRAIN} --arch efficientnet_b4 --epochs 12 --img-size 224 --batch-size 32 "
                    "--lr 1e-4 --no-freeze --num-workers 2 --max-train-minutes 320 --tag effb4"},
            {"name": "classical", "keep_model": False,
             "cmd": "python scripts/baseline_classical.py"},
            {"name": "deployed-eval", "keep_model": False,
             "cmd": "python scripts/eval_checkpoint.py model.pt --tag deployed"},
            {"name": "anomaly-per-type", "keep_model": False,
             "cmd": "python scripts/anomaly_eval.py --per-type --out runs/anomaly_per_type.json"},
        ]
    if name == "loto":
        ep6 = FT.replace("--epochs 12", "--epochs 6")
        return [
            {"name": f"loto-{c}", "keep_model": False,
             "cmd": f"{ep6} --exclude {c} --tag loto-{c}"}
            for c in ARTIFACT_CLASSES
        ]
    if name == "real":
        return [
            {"name": "real-native", "keep_model": True, "cmd": f"{REAL} --tag real-native"},
            {"name": "real-resize-first", "keep_model": True,
             "cmd": f"{REAL} --resize-first --tag real-resize-first"},
            {"name": "real-deployed-eval", "keep_model": False,
             "cmd": "python scripts/eval_checkpoint.py model.pt --substrate real "
                    "--tag deployed-real"},
        ]
    raise SystemExit(f"unknown suite '{name}'")


if __name__ == "__main__":
    print(json.dumps(suite(sys.argv[1] if len(sys.argv) > 1 else "smoke")))
