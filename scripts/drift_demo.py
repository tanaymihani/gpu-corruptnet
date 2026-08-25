"""Demo the M9 drift monitor: fit on clean frames, score a corrupted batch.

    python scripts/drift_demo.py

Model-free (image descriptors), so it runs with no training and no downloads.
"""

from __future__ import annotations

import numpy as np

from gpu_corruptnet.corruptions import apply, available
from gpu_corruptnet.data import make_demo_frame
from gpu_corruptnet.drift import DESCRIPTOR_NAMES, DriftMonitor, image_descriptors


def main() -> None:
    rng = np.random.default_rng(0)
    clean = np.stack([make_demo_frame(rng=s) for s in range(200)])
    mon = DriftMonitor(feature_names=DESCRIPTOR_NAMES).fit(image_descriptors(clean))

    # In-distribution reference (held-out clean frames) vs a corrupted production batch.
    clean_new = np.stack([make_demo_frame(rng=s) for s in range(200, 300)])
    corruptions = list(available())
    corrupted = np.stack(
        [apply(rng.choice(corruptions), make_demo_frame(rng=s), severity=4, rng=s)
         for s in range(300, 400)]
    )

    batches = [("clean (in-distribution)", clean_new), ("corrupted (production)", corrupted)]
    for name, batch in batches:
        rep = mon.report(image_descriptors(batch))
        print(
            f"{name:28s} -> status={rep['status']:18s} "
            f"max_psi={rep['max_psi']:.3f}  ood_rate={rep['ood_rate']:.2%}"
        )


if __name__ == "__main__":
    main()
