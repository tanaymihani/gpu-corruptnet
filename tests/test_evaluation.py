import numpy as np

from gpu_corruptnet.evaluation import (
    EmbeddingOOD,
    MondrianConformal,
    auroc,
    bootstrap_ci,
    brier,
    classwise_ece,
    macro_f1,
    set_coverage,
)


def _data(n=4000, c=5, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(size=(n, c))
    y = (rng.uniform(size=(n, c)) < p).astype(float)  # perfectly calibrated by construction
    return p, y


def test_classwise_ece_small_when_calibrated():
    p, y = _data()
    ece, per = classwise_ece(p, y)
    assert ece < 0.03 and len(per) == 5


def test_brier_zero_for_perfect_predictions():
    y = np.eye(4)
    assert brier(y, y) == 0.0


def test_mondrian_hits_per_class_target():
    p, y = _data(8000)
    cal, test = slice(0, 4000), slice(4000, None)
    m = MondrianConformal(alpha=0.1).fit(p[cal], y[cal])
    cov = set_coverage(m.predict_set(p[test]), y[test])
    assert cov["min_class_coverage"] > 0.86
    assert 0.0 <= cov["coverage_corrupted_frames"] <= cov["coverage_all_frames"] <= 1.0


def test_bootstrap_ci_brackets_point():
    p, y = _data(500)
    point, lo, hi = bootstrap_ci(macro_f1, y, p, n_boot=200)
    assert lo <= point <= hi


def test_embedding_ood_separates_shifted_cloud():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(500, 16))
    ood = EmbeddingOOD(k=8).fit(ref)
    a = auroc(ood.score(rng.normal(size=(200, 16))), ood.score(rng.normal(3, 1, size=(200, 16))))
    assert a > 0.95
