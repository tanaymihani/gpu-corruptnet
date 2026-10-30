"""Package a checkpoint for the Hugging Face Hub: safetensors weights + config + model card.

    python scripts/publish_hf.py model.pt --out hf_export            # build locally
    huggingface-cli login                                            # once
    python scripts/publish_hf.py model.pt --out hf_export --push tanaymihani/gpu-corruptnet-resnet50

The config carries everything inference needs: class order, input size, ImageNet
normalisation, the temperature and the conformal threshold fitted on the calibration split.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES
from gpu_corruptnet.models import load_classifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="hf_export")
    ap.add_argument("--calibration", default="results/calibration_deployed.json")
    ap.add_argument("--card", default="hf/README.md")
    ap.add_argument("--push", default=None, help="repo id, e.g. user/gpu-corruptnet-resnet50")
    args = ap.parse_args()

    from safetensors.torch import save_file

    net, meta = load_classifier(args.ckpt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in net.state_dict().items()}, out / "model.safetensors")
    cal = json.loads(Path(args.calibration).read_text()) if Path(args.calibration).exists() else {}
    (out / "config.json").write_text(json.dumps({
        "architecture": meta["arch"],
        "num_classes": meta["num_classes"],
        "classes": list(ARTIFACT_CLASSES),
        "img_size": meta["img_size"],
        "resize": "bilinear squash to img_size x img_size",
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "multi_label": True,
        "decision_threshold": 0.5,
        "temperature": cal.get("temperature"),
        "conformal_tau": cal.get("conformal_tau"),
        "conformal_alpha": cal.get("alpha"),
    }, indent=2))
    shutil.copy(args.card, out / "README.md")
    print(f"wrote {out}/model.safetensors, config.json, README.md")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.push)
        print(f"pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
