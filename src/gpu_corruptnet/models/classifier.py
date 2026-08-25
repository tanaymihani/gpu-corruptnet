"""Multi-label corruption classifiers via ImageNet transfer learning.

Head 1 of the system: ResNet-50 / EfficientNet-B4 with the final layer replaced by
a ``num_classes``-way linear head trained with BCE (multi-label). Backbone can be
frozen for a fast linear-probe first pass, then unfrozen to fine-tune.
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
    elif arch == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b4(weights=weights)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
        head_prefix = "classifier."
    else:
        raise ValueError(f"unknown arch '{arch}' (expected resnet50 | efficientnet_b4)")

    if freeze_backbone:
        for name, p in net.named_parameters():
            if not name.startswith(head_prefix):
                p.requires_grad = False
    return net


def unfreeze(net: nn.Module) -> None:
    for p in net.parameters():
        p.requires_grad = True
