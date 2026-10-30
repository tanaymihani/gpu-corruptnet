"""Multi-label corruption classifiers via ImageNet transfer learning.

Head 1 of the system: ResNet-50 / EfficientNet-B4 with the final layer replaced by
a ``num_classes``-way linear head trained with BCE (multi-label). Backbone can be
frozen for a fast linear-probe first pass, then unfrozen to fine-tune. ``small_cnn`` is a
from-scratch baseline (four conv blocks, ~0.4M params) to show what transfer learning buys.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_classifier(
    arch: str = "resnet50",
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        head_prefix = "fc."
    elif arch == "small_cnn":
        net = SmallCNN(num_classes)
        head_prefix = "fc."
        freeze_backbone = False  # nothing pretrained to freeze
    elif arch == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b4(weights=weights)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
        head_prefix = "classifier."
    else:
        raise ValueError(
            f"unknown arch '{arch}' (expected resnet50 | efficientnet_b4 | small_cnn)"
        )

    if freeze_backbone:
        for name, p in net.named_parameters():
            if not name.startswith(head_prefix):
                p.requires_grad = False
    return net


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 32) -> None:
        super().__init__()
        layers, c_in = [], 3
        for c_out in (width, width * 2, width * 4, width * 8):
            layers += [
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(c_in, num_classes)

    def forward(self, x):
        return self.fc(self.features(x))


def unfreeze(net: nn.Module) -> None:
    for p in net.parameters():
        p.requires_grad = True


def load_classifier(ckpt_path: str, map_location: str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild a classifier from a training checkpoint (saved by train.run)."""
    import torch

    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    net = build_classifier(
        ckpt["arch"], num_classes=ckpt["num_classes"], pretrained=False, freeze_backbone=False
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt
