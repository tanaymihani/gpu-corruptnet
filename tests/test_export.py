import numpy as np

from gpu_corruptnet.corruptions.base import ARTIFACT_CLASSES
from gpu_corruptnet.export import benchmark_onnx, export_onnx, make_session
from gpu_corruptnet.models import build_classifier


def test_onnx_export_and_inference(tmp_path):
    net = build_classifier("resnet50", num_classes=len(ARTIFACT_CLASSES), pretrained=False)
    path = export_onnx(net, tmp_path / "m.onnx", img_size=32)
    assert path.exists()

    sess = make_session(path, "CPUExecutionProvider")
    x = np.random.randn(2, 3, 32, 32).astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    assert out.shape == (2, len(ARTIFACT_CLASSES))  # dynamic batch works


def test_benchmark_onnx_structure(tmp_path):
    net = build_classifier("resnet50", pretrained=False)
    path = export_onnx(net, tmp_path / "m.onnx", img_size=32)
    sess = make_session(path, "CPUExecutionProvider")
    res = benchmark_onnx(sess, img_size=32, batch_sizes=(1, 2), warmup=1, iters=3)
    assert [r.batch_size for r in res] == [1, 2]
    assert all(r.throughput_ips > 0 for r in res)
