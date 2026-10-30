"""Build the web demo's sample gallery and config.

    python scripts/make_web_samples.py --calibration calibration.json

Takes a few free-licence frames from realdata/clean_frames.csv, injects artifacts with the
real generator at native resolution, saves 960-px JPEGs to web/samples/ with a credits file
(title, licence, author, source page), and writes web/config.json (classes, temperature,
conformal threshold, input size, sample list).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from gpu_corruptnet.corruptions import apply
from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES
from gpu_corruptnet.data.clean_sources import read_manifest

# (title, artifacts to show on that frame)
PICKS = [
    ("SuperTuxKart", ["screen_tearing", "discoloration"]),
    ("0 A.D.", ["triangulation", "morse_code"]),
    ("FlightGear", ["parallel_lines", "shader"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="realdata/clean_frames.csv")
    ap.add_argument("--root", default="data/real_frames")
    ap.add_argument("--out", default="web")
    ap.add_argument("--calibration", default=None, help="JSON with temperature + conformal tau")
    ap.add_argument("--temperature", type=float, default=1.5647)
    ap.add_argument("--tau", type=float, default=0.0493)
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    out = Path(args.out) / "samples"
    out.mkdir(parents=True, exist_ok=True)
    samples, credits = [], []
    rng = np.random.default_rng(7)
    for title, artifacts in PICKS:
        cands = [r for r in rows if r["title"] == title and (Path(args.root) / r["file"]).exists()
                 and int(r["width"]) >= 1280]
        row = cands[len(cands) // 2]
        frame = np.asarray(Image.open(Path(args.root) / row["file"]).convert("RGB"))
        stem = title.lower().replace(" ", "").replace(".", "").replace(":", "")
        variants = [("clean", frame)] + [
            (a, apply(a, frame, severity=4, rng=rng)) for a in artifacts
        ]
        for name, img in variants:
            im = Image.fromarray(img)
            im = im.resize((960, round(im.height * 960 / im.width)), Image.BILINEAR)
            fname = f"{stem}_{name}.jpg"
            im.save(out / fname, quality=88)
            samples.append({"file": fname, "caption": f"{title}: {name.replace('_', ' ')}",
                            "artifact": None if name == "clean" else name})
        credits.append({k: row[k] for k in ("title", "license", "artist", "commons_file",
                                             "source_page")})
    (out / "credits.json").write_text(json.dumps(
        {"note": "Frames from Wikimedia Commons under the licences listed; artifacts were "
                 "added by the GPU-CorruptNet generator.", "frames": credits}, indent=2))

    t, tau = args.temperature, args.tau
    if args.calibration:
        c = json.loads(Path(args.calibration).read_text())
        t = c["temperature"]
        tau = c["splits"]["seen_test"]["conformal"]["tau"]
    order = [s for s in samples if s["artifact"] is None] + [s for s in samples if s["artifact"]]
    cfg = {"classes": list(ARTIFACT_CLASSES), "temperature": t, "conformal_tau": tau,
           "img_size": 224, "samples": order}
    (Path(args.out) / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"{len(samples)} samples, T={t:.4f}, tau={tau:.4f}")


if __name__ == "__main__":
    main()
