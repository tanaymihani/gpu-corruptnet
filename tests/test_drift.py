import numpy as np

from gpu_corruptnet.corruptions import apply
from gpu_corruptnet.data import make_demo_frame
from gpu_corruptnet.drift import DriftMonitor, image_descriptors


def test_no_drift_when_same_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=(3000, 6))
    cur = rng.normal(0, 1, size=(3000, 6))
    rep = DriftMonitor(psi_bins=10).fit(ref).report(cur)
    assert not rep["drifted"]
    assert rep["max_psi"] < 0.25
    assert rep["ood_rate"] < 0.1  # ~1% by construction (99th-pct threshold)


def test_drift_flagged_when_distribution_shifts():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, size=(3000, 6))
    cur = rng.normal(3, 1, size=(3000, 6))  # shifted 3 sigma
    rep = DriftMonitor(psi_bins=10).fit(ref).report(cur)
    assert rep["drifted"]
    assert rep["status"] == "significant_drift"
    assert rep["ood_rate"] > 0.5


def test_image_descriptors_shape():
    imgs = np.stack([make_demo_frame(h=80, w=120, rng=s) for s in range(5)])
    feats = image_descriptors(imgs)
    assert feats.shape == (5, 9)


def test_corrupted_frames_register_as_drift():
    clean = np.stack([make_demo_frame(h=100, w=160, rng=s) for s in range(150)])
    corrupted = np.stack(
        [apply("parallel_lines", make_demo_frame(h=100, w=160, rng=s), severity=4, rng=s)
         for s in range(150, 250)]
    )
    from gpu_corruptnet.drift import DESCRIPTOR_NAMES

    mon = DriftMonitor(feature_names=DESCRIPTOR_NAMES).fit(image_descriptors(clean))
    rep = mon.report(image_descriptors(corrupted))
    assert rep["drifted"]           # corruption shifts edge/color stats
    assert rep["ood_rate"] > 0.2
