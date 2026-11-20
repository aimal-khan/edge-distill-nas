"""
Lightweight student architectures for CIFAR-100 knowledge distillation.
  - MobileNetV2
  - SqueezeNet 1.1
  - EfficientNet-Lite0 (based on EfficientNet-B0, lite variant: no SE, fixed resolution)

All expose `forward(x, return_features)` for feature-based distillation.
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ---------------------------------------------------------------------------
# MobileNetV2
# ---------------------------------------------------------------------------
class StudentMobileNetV2(nn.Module):
    """MobileNetV2 adapted for CIFAR-100."""

    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.mobilenet_v2(weights=weights)
        self.features = base.features          # outputs (B, 1280, H, W)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.dropout  = nn.Dropout(0.2)
        self.fc       = nn.Linear(1280, num_classes)

    def forward(self, x, return_features: bool = False):
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        out    = self.fc(self.dropout(pooled))
        if return_features:
            return out, feat
        return out


# ---------------------------------------------------------------------------
# SqueezeNet 1.1
# ---------------------------------------------------------------------------
class StudentSqueezeNet(nn.Module):
    """SqueezeNet 1.1 adapted for CIFAR-100."""

    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super().__init__()
        weights = models.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.squeezenet1_1(weights=weights)
        # Keep all feature layers; replace classifier
        self.features = base.features          # outputs (B, 512, H, W)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.dropout  = nn.Dropout(0.5)
        self.fc       = nn.Linear(512, num_classes)

    def forward(self, x, return_features: bool = False):
        feat   = self.features(x)
        pooled = self.pool(feat).flatten(1)
        out    = self.fc(self.dropout(pooled))
        if return_features:
            return out, feat
        return out


# ---------------------------------------------------------------------------
# EfficientNet-Lite0  (EfficientNet-B0 without SE blocks, fixed resolution)
# ---------------------------------------------------------------------------
class StudentEfficientNetLite0(nn.Module):
    """
    EfficientNet-Lite0: EfficientNet-B0 backbone with Squeeze-and-Excitation
    disabled (Lite variant) and final FC replaced for CIFAR-100.
    """

    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.efficientnet_b0(weights=weights)

        # Disable SE blocks to match Lite0 spec
        for module in base.modules():
            if isinstance(module, models.efficientnet.SqueezeExcitation):
                module.activation = nn.Identity()
                module.scale_activation = nn.Identity()
                # Zero-out scale so SE path is bypassed
                nn.init.ones_(module.fc2.weight)
                nn.init.zeros_(module.fc2.bias)

        self.features    = base.features        # (B, 1280, H, W)
        self.pool        = nn.AdaptiveAvgPool2d(1)
        self.dropout     = nn.Dropout(0.2)
        self.fc          = nn.Linear(1280, num_classes)

    def forward(self, x, return_features: bool = False):
        feat   = self.features(x)
        pooled = self.pool(feat).flatten(1)
        out    = self.fc(self.dropout(pooled))
        if return_features:
            return out, feat
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
STUDENT_REGISTRY = {
    "mobilenet":    StudentMobileNetV2,
    "squeezenet":   StudentSqueezeNet,
    "efficientnet": StudentEfficientNetLite0,
}

def get_student(name: str, num_classes: int = 100, pretrained: bool = True) -> nn.Module:
    """Return a student model by name key."""
    if name not in STUDENT_REGISTRY:
        raise ValueError(f"Unknown student '{name}'. Choose from {list(STUDENT_REGISTRY)}")
    return STUDENT_REGISTRY[name](num_classes=num_classes, pretrained=pretrained)
