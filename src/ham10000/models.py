"""Backbone factories: ImageNet-pretrained, 7-class head.

Each backbone keeps whatever dropout its reference implementation ships with.
MobileNetV2 has p=0.2, EfficientNet-B3 has p=0.3, ResNet-50 has none. That is
an asymmetry between the arms and it is deliberate: the comparison is between
these networks as they are actually published and deployed, not between three
artificially matched variants. It is declared as a limitation in the write-up.

Only the final Linear layer is swapped for a 7-output one. For MobileNetV2 and
EfficientNet-B3 the classifier is a Sequential(Dropout, Linear), so index [1]
is replaced and the dropout at index [0] survives untouched.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

SUPPORTED = ("mobilenet_v2", "resnet50", "efficientnet_b3")


def build_model(architecture: str, num_classes: int = 7) -> nn.Module:
    arch = architecture.lower()

    if arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if arch in ("resnet50", "resnet_50"):
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch in ("efficientnet_b3", "efficientnet-b3"):
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    raise ValueError(f"Unknown architecture: {architecture}. Expected one of {SUPPORTED}.")


def describe_model(architecture: str, num_classes: int = 7) -> dict:
    """Structural facts used in the methodology chapter and the defence slides.

    Counts modules rather than quoting the paper, so the numbers match the code
    that actually ran. Note that torchvision's ResNet-50 reports 53 Conv2d
    modules, not 49: four of them are the 1x1 projection convolutions on the
    shortcut path, which the canonical "50 layers" figure does not count.
    """
    model = build_model(architecture, num_classes)
    convs = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    linears = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    dropouts = [m.p for m in model.modules() if isinstance(m, nn.Dropout)]
    return {
        "architecture": architecture,
        "conv2d_modules": convs,
        "linear_modules": linears,
        "weight_layers": convs + linears,
        "batchnorm_modules": sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d)),
        "dropout_p": dropouts,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
