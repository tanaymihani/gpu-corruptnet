"""Export the classifier for the in-browser demo (ONNX Runtime Web) and check parity.

    python scripts/export_web.py model.pt --out web/model.onnx [--ref runs/preds_deployed.npz]

* Two outputs: ``logits`` and the 2048-d ``embedding`` (input to the final layer), so the
  page can score OOD on model embeddings.
* Weights are stored as float16 with a Cast to float32 in front of each use, which halves the
  download; ONNX Runtime folds the casts at load, so compute stays float32.
* Parity: logits from the exported model vs PyTorch on the same inputs (max |diff|, and
  whether the predicted label sets agree), written next to the model as parity.json.
* With ``--ref``, also writes web/ood_reference.json: mean + top-64 PCA basis of the training
  reference embeddings, and the 99th-percentile in-distribution score as the OOD threshold.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

from gpu_corruptnet.models import load_classifier


class WithEmbedding(torch.nn.Module):
    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.body = copy.deepcopy(net)
        self.body.fc = torch.nn.Identity()
        self.fc = net.fc

    def forward(self, x):
        e = self.body(x)
        return self.fc(e), e


def fp16_storage(model: onnx.ModelProto, min_size: int = 1024) -> onnx.ModelProto:
    g = model.graph
    casts, keep = [], []
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32 and arr.size >= min_size:
            half = numpy_helper.from_array(arr.astype(np.float16), init.name + "_fp16")
            keep.append(half)
            casts.append(
                helper.make_node("Cast", [half.name], [init.name], to=onnx.TensorProto.FLOAT)
            )
        else:
            keep.append(init)
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    return model


def write_ood_reference(ref_npz: str, out_dir: Path, k: int = 64) -> None:
    """mean + top-k PCA basis of the training reference embeddings; threshold = 99th pct of
    seen-content test scores. Same maths as evaluation.EmbeddingOOD."""
    from gpu_corruptnet.evaluation import EmbeddingOOD

    z = np.load(ref_npz)
    det = EmbeddingOOD(k=k).fit(z["train_ref_feats"])
    thr = float(np.percentile(det.score(z["seen_test_feats"]), 99))
    b64 = lambda a: base64.b64encode(np.asarray(a, np.float32).tobytes()).decode()  # noqa: E731
    (out_dir / "ood_reference.json").write_text(json.dumps({
        "dim": int(det.mean.shape[0]), "k": k, "threshold": thr,
        "mean_b64": b64(det.mean), "components_b64": b64(det.comps), "variance_b64": b64(det.var),
        "note": "PCA-Mahalanobis on penultimate embeddings; threshold = 99th pct of "
                "seen-content test scores",
    }))
    print(f"OOD threshold {thr:.2f} (k={k})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="web/model.onnx")
    ap.add_argument("--ref", default=None, help="preds npz with train_ref_feats + test feats")
    ap.add_argument("--n-parity", type=int, default=64)
    ap.add_argument("--ref-only", action="store_true", help="only write ood_reference.json")
    args = ap.parse_args()
    if args.ref_only:
        write_ood_reference(args.ref, Path(args.out).parent)
        return

    net, ckpt = load_classifier(args.ckpt)
    size = int(ckpt["img_size"])
    wrapped = WithEmbedding(net).eval()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".fp32.onnx")
    torch.onnx.export(
        wrapped, torch.randn(1, 3, size, size), str(tmp),
        input_names=["input"], output_names=["logits", "embedding"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    model = fp16_storage(onnx.load(str(tmp)))
    onnx.save(model, str(out))
    tmp.unlink()

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = rng.standard_normal((args.n_parity, 3, size, size)).astype(np.float32)
    with torch.no_grad():
        ref_logits = wrapped(torch.from_numpy(x))[0].numpy()
    ort_logits = np.concatenate([sess.run(["logits"], {"input": x[i : i + 8]})[0]
                                 for i in range(0, len(x), 8)])
    agree = float(np.mean(np.all((ref_logits > 0) == (ort_logits > 0), axis=1)))
    parity = {
        "max_abs_logit_diff": float(np.abs(ref_logits - ort_logits).max()),
        "label_set_agreement": agree,
        "n_inputs": args.n_parity,
        "model_mb": out.stat().st_size / 1e6,
    }
    (out.parent / "parity.json").write_text(json.dumps(parity, indent=2))
    print(json.dumps(parity))

    if args.ref:
        write_ood_reference(args.ref, out.parent)


if __name__ == "__main__":
    main()
