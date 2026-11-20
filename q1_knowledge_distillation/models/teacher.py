"""
Teacher model: ResNet-50 adapted for CIFAR-100.
Supports intermediate feature extraction for feature-based distillation.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class TeacherResNet50(nn.Module):
    """ResNet-50 teacher for CIFAR-100 (100-class head)."""

    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet50(weights=weights)

        # Feature extractor: everything before the final pool + FC
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.pool   = nn.AdaptiveAvgPool2d(1)

        # Replace classifier head for CIFAR-100
        self.fc = nn.Linear(2048, num_classes)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, return_features: bool = False):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat3 = self.layer3(x)   # intermediate feature for distillation
        feat4 = self.layer4(feat3)
        pooled = self.pool(feat4).flatten(1)
        logits = self.fc(pooled)
        if return_features:
            return logits, feat4   # (B, 2048, H, W)
        return logits


def get_teacher(num_classes: int = 100, pretrained: bool = True) -> TeacherResNet50:
    return TeacherResNet50(num_classes=num_classes, pretrained=pretrained)
