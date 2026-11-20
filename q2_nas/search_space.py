"""
NAS Search Space Definition for edge-deployable CNN architectures.

Searchable dimensions:
  - num_layers     : variable convolutional depth (2–6 blocks)
  - num_filters    : variable filter counts per layer
  - kernel_sizes   : 3×3 or 5×5
  - use_depthwise  : depthwise separable convolution blocks
  - activation     : relu, hardswish, gelu
  - input_resolution: 24 or 32 (CIFAR-10 input is 32×32, optionally downsampled)
  - dropout_rate   : regularisation
"""

from __future__ import annotations
from typing import Any

import optuna

# ---------------------------------------------------------------------------
# Hard constraints for hardware-aware selection
# ---------------------------------------------------------------------------
MAX_MODEL_MB     = 10.0     # max model storage size in MB
MAX_LATENCY_MS   = 150.0    # max CPU inference latency in ms (single image)
MAX_PEAK_MEM_MB  = 200.0    # max peak inference memory in MB (best-effort)

# ---------------------------------------------------------------------------
# Search space boundaries
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    "num_layers":        {"type": "int",        "low": 2,  "high": 6},
    "num_filters_base":  {"type": "categorical", "choices": [16, 32, 64]},
    "filter_multiplier": {"type": "float",       "low": 1.0, "high": 2.0, "step": 0.5},
    "kernel_size":       {"type": "categorical", "choices": [3, 5]},
    "use_depthwise":     {"type": "categorical", "choices": [True, False]},
    "activation":        {"type": "categorical", "choices": ["relu", "hardswish", "gelu"]},
    "input_resolution":  {"type": "categorical", "choices": [24, 32]},
    "dropout_rate":      {"type": "float",       "low": 0.0, "high": 0.5},
}


def sample_config(trial: optuna.Trial) -> dict[str, Any]:
    """Draw a configuration from the search space using an Optuna trial."""
    config = {}
    for name, spec in SEARCH_SPACE.items():
        if spec["type"] == "int":
            config[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec["type"] == "float":
            if "step" in spec:
                config[name] = trial.suggest_float(name, spec["low"], spec["high"],
                                                    step=spec["step"])
            else:
                config[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif spec["type"] == "categorical":
            config[name] = trial.suggest_categorical(name, spec["choices"])
    return config


def config_to_str(config: dict) -> str:
    """Human-readable config summary."""
    return (
        f"layers={config['num_layers']} "
        f"filters={config['num_filters_base']}×{config['filter_multiplier']:.1f} "
        f"k={config['kernel_size']} "
        f"dw={config['use_depthwise']} "
        f"act={config['activation']} "
        f"res={config['input_resolution']} "
        f"drop={config['dropout_rate']:.2f}"
    )
