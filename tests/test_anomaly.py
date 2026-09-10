import numpy as np

from gpu_corruptnet.anomaly import NearestNeighborAnomaly, auroc


def test_nn_anomaly_separates_shifted_distribution():
    rng = np.random.default_rng(0)
    clean_fit = rng.normal(0, 1, size=(600, 32))
    clean_test = rng.normal(0, 1, size=(200, 32))
    anomalies = rng.normal(3, 1, size=(200, 32))  # far from the clean manifold

    det = NearestNeighborAnomaly(k=3).fit(clean_fit)
    scores = np.concatenate([det.score(clean_test), det.score(anomalies)])
    labels = np.concatenate([np.zeros(200), np.ones(200)])
    assert det.score(anomalies).mean() > det.score(clean_test).mean()
    assert auroc(scores, labels) > 0.95


def test_coreset_caps_memory_bank():
    rng = np.random.default_rng(1)
    det = NearestNeighborAnomaly(k=1).fit(rng.normal(0, 1, size=(5000, 8)), coreset=500)
    assert det.nn_.n_samples_fit_ == 500
