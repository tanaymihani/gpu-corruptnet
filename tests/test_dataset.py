import numpy as np
import torch

from gpu_corruptnet.data.corrupt_dataset import CorruptionDataset
from gpu_corruptnet.models import build_classifier


def _frames(n=6, h=40, w=56):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)


def test_epoch_zero_matches_per_index_seed():
    ds = CorruptionDataset(_frames(), base_seed=3, img_size=32)
    x0, y0 = ds[2]
    x1, y1 = CorruptionDataset(_frames(), base_seed=3, img_size=32)[2]
    assert torch.equal(x0, x1) and torch.equal(y0, y1)


def test_set_epoch_redraws_corruptions():
    ds = CorruptionDataset(_frames(30), base_seed=1, img_size=32, clean_fraction=0.0)
    first = [ds[i][1] for i in range(30)]
    ds.set_epoch(1)
    second = [ds[i][1] for i in range(30)]
    assert any(not torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_variable_size_frames_and_paths(tmp_path):
    from PIL import Image

    frames = [_frames(1, 50, 80)[0], _frames(1, 120, 90)[0]]
    path = tmp_path / "f.png"
    Image.fromarray(frames[1]).save(path)
    ds = CorruptionDataset([frames[0], path], base_seed=0, img_size=32, crop_scale=(0.5, 1.0))
    for i in range(2):
        x, y = ds[i]
        assert x.shape == (3, 32, 32) and y.shape == (10,)


def test_resize_first_changes_where_artifacts_are_painted():
    f = _frames(4, 96, 96)
    a = CorruptionDataset(f, base_seed=5, img_size=64, clean_fraction=0.0)
    b = CorruptionDataset(f, base_seed=5, img_size=64, clean_fraction=0.0, resize_first=True)
    assert torch.equal(a[0][1], b[0][1])  # same labels drawn
    assert not torch.equal(a[0][0], b[0][0])  # different pixels


def test_excluded_type_never_labelled():
    ds = CorruptionDataset(
        _frames(40), base_seed=1, img_size=32, clean_fraction=0.0,
        corruptions=["screen_tearing", "discoloration"],
    )
    labels = torch.stack([ds[i][1] for i in range(40)])
    assert labels.sum(0).nonzero().numel() <= 2


def test_small_cnn_forward():
    net = build_classifier("small_cnn", num_classes=10)
    assert net(torch.zeros(2, 3, 64, 64)).shape == (2, 10)
