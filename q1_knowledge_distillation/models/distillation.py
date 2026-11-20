"""
Knowledge Distillation loss module.

Supports:
  - Soft-target KD loss  (KL divergence with temperature-scaled softmax)
  - Hard-label CE loss
  - Combined hard + soft training
  - Feature-based distillation (L2 on projected feature maps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAdaptor(nn.Module):
    """
    Projects student feature maps to match teacher feature spatial statistics.
    Uses global average pooling + linear projection to match channel dim.
    """

    def __init__(self, student_channels: int, teacher_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(student_channels, teacher_channels),
            nn.ReLU(inplace=True),
        )
        self.teacher_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, student_feat, teacher_feat):
        s = self.proj(student_feat)                        # (B, teacher_ch)
        t = self.teacher_pool(teacher_feat).flatten(1)    # (B, teacher_ch)
        return s, t


class KDLoss(nn.Module):
    """
    Combined knowledge distillation loss.

    L = alpha * L_soft + (1 - alpha) * L_hard + beta * L_feature

    where:
      L_soft    = KL(softmax(s/T) || softmax(t/T)) * T^2
      L_hard    = CrossEntropy(s, y)
      L_feature = MSE(projected student feat, teacher feat)
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,     # weight on soft loss
        beta: float = 0.1,      # weight on feature loss
    ):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha
        self.beta  = beta
        self.ce    = nn.CrossEntropyLoss()

    def soft_loss(self, student_logits, teacher_logits):
        """Temperature-scaled KL divergence (soft-label KD)."""
        s_soft = F.log_softmax(student_logits / self.T, dim=-1)
        t_soft = F.softmax(teacher_logits   / self.T, dim=-1)
        # KL(t || s) scaled by T^2 as per Hinton et al.
        return F.kl_div(s_soft, t_soft, reduction="batchmean") * (self.T ** 2)

    def feature_loss(self, student_feat_proj, teacher_feat_proj):
        """L2 feature matching loss (normalised)."""
        s = F.normalize(student_feat_proj, dim=-1)
        t = F.normalize(teacher_feat_proj, dim=-1)
        return F.mse_loss(s, t)

    def forward(
        self,
        student_logits,
        teacher_logits,
        labels,
        student_feat_proj=None,
        teacher_feat_proj=None,
    ):
        L_soft = self.soft_loss(student_logits, teacher_logits)
        L_hard = self.ce(student_logits, labels)
        loss   = self.alpha * L_soft + (1.0 - self.alpha) * L_hard

        if student_feat_proj is not None and teacher_feat_proj is not None:
            L_feat = self.feature_loss(student_feat_proj, teacher_feat_proj)
            loss   = loss + self.beta * L_feat

        return loss
