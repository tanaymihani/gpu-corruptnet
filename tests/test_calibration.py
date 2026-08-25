import numpy as np

from gpu_corruptnet.calibration import (
    ConformalLabelSets,
    TemperatureScaler,
    expected_calibration_error,
    sigmoid,
)


def test_ece_near_zero_for_perfectly_calibrated():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, size=(20000, 10))
    y = (rng.uniform(size=p.shape) < p).astype(float)
    ece, _ = expected_calibration_error(p, y, n_bins=15)
    assert ece < 0.03


def test_temperature_scaling_recovers_scale_and_reduces_ece():
    rng = np.random.default_rng(1)
    true_logits = rng.normal(0, 1.5, size=(4000, 10))
    labels = (rng.uniform(size=true_logits.shape) < sigmoid(true_logits)).astype(float)
    overconfident = true_logits * 2.5  # a model that is too sharp

    cal, test = slice(0, 2000), slice(2000, None)
    ts = TemperatureScaler().fit(overconfident[cal], labels[cal])

    ece_before, _ = expected_calibration_error(sigmoid(overconfident[test]), labels[test])
    ece_after, _ = expected_calibration_error(ts.predict_proba(overconfident[test]), labels[test])

    assert ts.temperature > 1.5          # recovered the tempering direction (truth = 2.5)
    assert ece_after < ece_before        # calibration improved


def test_conformal_achieves_target_coverage():
    rng = np.random.default_rng(0)

    def gen(n: int) -> tuple[np.ndarray, np.ndarray]:
        p = rng.uniform(0, 1, size=(n, 10))
        y = (rng.uniform(size=(n, 10)) < p).astype(float)  # calibrated by construction
        return p, y

    p_cal, y_cal = gen(4000)
    p_test, y_test = gen(4000)
    cp = ConformalLabelSets(alpha=0.1).fit(p_cal, y_cal)
    res = cp.evaluate(p_test, y_test)

    assert res["empirical_coverage"] >= 0.85          # target is 0.90; allow finite-sample slack
    assert 0.0 <= res["avg_set_size"] <= 10.0


def test_tighter_alpha_gives_larger_sets():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, size=(3000, 10))
    y = (rng.uniform(size=p.shape) < p).astype(float)
    loose = ConformalLabelSets(alpha=0.2).fit(p, y).evaluate(p, y)
    tight = ConformalLabelSets(alpha=0.05).fit(p, y).evaluate(p, y)
    assert tight["avg_set_size"] >= loose["avg_set_size"]
    assert tight["empirical_coverage"] >= loose["empirical_coverage"] - 1e-9
