import torch

from gpu_corruptnet.bench import benchmark, model_stats
from gpu_corruptnet.models import build_classifier


def test_benchmark_returns_sane_structure_on_cpu():
    net = build_classifier("resnet50", pretrained=False, freeze_backbone=False)
    res = benchmark(net, torch.device("cpu"), img_size=32, batch_sizes=(1, 2), warmup=1, iters=3)
    assert [r.batch_size for r in res] == [1, 2]
    for r in res:
        assert r.mean_ms > 0
        assert r.throughput_ips > 0
        assert r.p99_ms >= r.p50_ms  # percentiles ordered


def test_model_stats_resnet50():
    stats = model_stats(build_classifier("resnet50", pretrained=False))
    assert 20 < stats["params_millions"] < 27  # ResNet-50 ~23.5M with a 10-way head
    assert stats["size_mb"] > 0
