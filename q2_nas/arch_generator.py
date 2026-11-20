"""
Architecture Generator: constructs an nn.Module from a sampled config dict.

Supports:
  - Standard Conv2d blocks (BN + activation)
  - Depthwise-separable convolution blocks
  - Variable depth, filters, kernel size, activation, input resolution
  - Global average pooling → classifier head (10 classes for CIFAR-10)
"""

from __future__ import annotations
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Activation factory
# ---------------------------------------------------------------------------
_ACT = {
    "relu":       nn.ReLU,
    "hardswish":  nn.Hardswish,
    "gelu":       nn.GELU,
}


def _make_act(name: str) -> nn.Module:
    return _ACT.get(name, nn.ReLU)()


# ---------------------------------------------------------------------------
# Conv blocks
# ---------------------------------------------------------------------------
class ConvBNAct(nn.Module):
    """Standard conv → BN → activation."""

    def __init__(self, in_ch, out_ch, kernel, stride=1, activation="relu"):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            _make_act(activation),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSepConv(nn.Module):
    """Depthwise separable conv: depthwise → pointwise → BN → activation."""

    def __init__(self, in_ch, out_ch, kernel, stride=1, activation="relu"):
        super().__init__()
        pad = kernel // 2
        self.dw = nn.Conv2d(in_ch, in_ch,  kernel, stride=stride,
                            padding=pad, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = _make_act(activation)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


# ---------------------------------------------------------------------------
# Architecture builder
# ---------------------------------------------------------------------------
def build_arch(config: dict[str, Any], num_classes: int = 10) -> nn.Module:
    """
    Build a CNN architecture from the given configuration dictionary.

    Returns an nn.Sequential-based model ready for training.
    """
    n_layers   = config["num_layers"]
    base_ch    = config["num_filters_base"]
    multiplier = config["filter_multiplier"]
    kernel     = config["kernel_size"]
    use_dw     = config["use_depthwise"]
    activation = config["activation"]
    dropout    = config["dropout_rate"]

    layers: list[nn.Module] = []

    in_ch = 3
    for i in range(n_layers):
        out_ch = max(16, int(base_ch * (multiplier ** i)))
        # Cap channels at 256 to stay edge-friendly
        out_ch = min(out_ch, 256)

        stride = 2 if i < 2 else 1    # downsample in first two blocks

        if use_dw and i > 0:
            # First layer is always standard conv (3-ch input)
            layers.append(DepthwiseSepConv(in_ch, out_ch, kernel,
                                           stride=stride, activation=activation))
        else:
            layers.append(ConvBNAct(in_ch, out_ch, kernel,
                                    stride=stride, activation=activation))
        in_ch = out_ch

    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(in_ch, num_classes))

    return nn.Sequential(*layers)
