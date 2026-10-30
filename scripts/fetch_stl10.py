"""Rebuild STL-10's labelled binaries (train/test _X.bin, _y.bin) from a Hugging Face mirror.

    python scripts/fetch_stl10.py data            # -> data/stl10_binary/*.bin

The original host (ai.stanford.edu) refuses connections from GitHub's runners. The
tanganke/stl10 mirror stores the same images losslessly (PNG) in a different order;
configs/stl10_hf_order.npz maps mirror rows back to the original order, and the rebuilt
files are checked against torchvision's MD5s, so the arrays are byte-identical.
"""

from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://huggingface.co/datasets/tanganke/stl10/resolve/refs%2Fconvert%2Fparquet/default/{split}/0000.parquet"
MD5 = {
    "train_X.bin": "918c2871b30a85fa023e0c44e0bee87f",
    "train_y.bin": "5a34089d4802c674881badbb80307741",
    "test_X.bin": "7f263ba9f9e0b06b93213547f721ac82",
    "test_y.bin": "36f9794fa4beb8a2c72628de14fa638e",
}
ORDER = Path(__file__).resolve().parents[1] / "configs" / "stl10_hf_order.npz"


def decode(parquet_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq
    from PIL import Image

    t = pq.read_table(io.BytesIO(parquet_bytes))
    imgs = np.stack([np.asarray(Image.open(io.BytesIO(c["bytes"])).convert("RGB"))
                     for c in t.column("image").to_pylist()])
    return imgs, np.asarray(t.column("label").to_pylist(), dtype=np.int64)


def write_split(imgs: np.ndarray, labels: np.ndarray, base: Path, split: str) -> None:
    # Inverse of the reader in clean_sources._stl10_arrays: (N, H, W, C) -> (N, C, W, H).
    raw = np.ascontiguousarray(imgs.transpose(0, 3, 2, 1))
    (base / f"{split}_X.bin").write_bytes(raw.tobytes())
    (base / f"{split}_y.bin").write_bytes((labels + 1).astype(np.uint8).tobytes())


def main(root: str) -> None:
    base = Path(root) / "stl10_binary"
    base.mkdir(parents=True, exist_ok=True)
    order = np.load(ORDER)
    for split in ("train", "test"):
        with urllib.request.urlopen(URL.format(split=split), timeout=300) as r:
            imgs, labels = decode(r.read())
        perm = order[split]  # original index i comes from mirror row perm[i]
        write_split(imgs[perm], labels[perm], base, split)
    for name, md5 in MD5.items():
        got = hashlib.md5((base / name).read_bytes()).hexdigest()
        if got != md5:
            raise SystemExit(f"{name}: md5 {got} != {md5}")
    print(f"STL-10 labelled splits rebuilt and verified in {base}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
