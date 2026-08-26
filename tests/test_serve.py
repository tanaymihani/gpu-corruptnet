import torch

from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES
from gpu_corruptnet.data import make_demo_frame
from gpu_corruptnet.models import build_classifier
from gpu_corruptnet.serve import CorruptionInspector


def test_inspector_without_model_gives_ood_only():
    res = CorruptionInspector(None).predict(make_demo_frame(rng=1))
    assert res["has_model"] is False
    assert "ood_score" in res and "ood_flag" in res
    assert "probs" not in res  # no detection without a model


def test_inspector_with_model_gives_full_prediction(tmp_path):
    net = build_classifier("resnet50", num_classes=len(ARTIFACT_CLASSES), pretrained=False)
    ckpt = tmp_path / "m.pt"
    torch.save(
        {"arch": "resnet50", "num_classes": 10, "img_size": 64, "state_dict": net.state_dict()},
        str(ckpt),
    )
    insp = CorruptionInspector(str(ckpt), temperature=1.5, conformal_tau=0.3)
    res = insp.predict(make_demo_frame(rng=2))

    assert res["has_model"]
    assert len(res["probs"]) == 10
    assert "latency_ms" in res
    assert "conformal_set" in res
    assert 0.0 <= res["corrupted_prob"] <= 1.0
