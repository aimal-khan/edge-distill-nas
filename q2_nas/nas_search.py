"""
NAS Search using Optuna (Bayesian optimisation / TPE sampler).

Hardware-aware objective:
  1. Sample architecture config from search space
  2. Build model; measure size → reject if > MAX_MODEL_MB
  3. Proxy-train 3 epochs on CIFAR-10 (small subset for speed)
  4. Measure CPU inference latency → reject if > MAX_LATENCY_MS
  5. Measure peak memory (tracemalloc)
  6. Report val accuracy to Optuna

Best config and search history saved to results/.

Usage:
    python nas_search.py [--n-trials 20] [--proxy-epochs 3]
                         [--batch-size 64] [--smoke-test]
"""

import argparse
import os
import sys
import time
import json
import tracemalloc
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_space   import sample_config, config_to_str, MAX_MODEL_MB, MAX_LATENCY_MS
from arch_generator import build_arch

BASE    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(BASE, "..", ".data")
RESULTS = os.path.join(BASE, "results")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def get_loaders(resolution: int, batch_size: int, proxy_n: int = 5000):
    """Fast proxy data loaders using a small subset of CIFAR-10."""
    train_tf = T.Compose([
        T.Resize(resolution),
        T.RandomCrop(resolution, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    val_tf = T.Compose([
        T.Resize(resolution),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_ds = torchvision.datasets.CIFAR10(DATA, train=True,  download=True, transform=train_tf)
    val_ds   = torchvision.datasets.CIFAR10(DATA, train=False, download=True, transform=val_tf)

    train_ds = Subset(train_ds, range(min(proxy_n, len(train_ds))))
    val_ds   = Subset(val_ds,   range(min(1000,    len(val_ds))))

    train_ldr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=1)
    val_ldr   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=1)
    return train_ldr, val_ldr


# ---------------------------------------------------------------------------
# Model size check
# ---------------------------------------------------------------------------
def model_size_mb(model: nn.Module) -> float:
    """Estimate model storage size in MB (parameter bytes)."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / 1e6


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------
def measure_cpu_latency(model: nn.Module, resolution: int, n_runs: int = 20) -> float:
    """Mean CPU inference latency in ms (single image)."""
    model.eval()
    dummy = torch.randn(1, 3, resolution, resolution)
    lats  = []
    with torch.no_grad():
        for _ in range(n_runs + 3):
            t0 = time.perf_counter()
            model(dummy)
            lats.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(lats[3:]))


# ---------------------------------------------------------------------------
# Peak memory measurement (tracemalloc)
# ---------------------------------------------------------------------------
def measure_peak_memory_mb(model: nn.Module, resolution: int) -> float:
    model.eval()
    dummy = torch.randn(1, 3, resolution, resolution)
    tracemalloc.start()
    with torch.no_grad():
        model(dummy)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


# ---------------------------------------------------------------------------
# Proxy training
# ---------------------------------------------------------------------------
def proxy_train(model: nn.Module, train_ldr, val_ldr, epochs: int, device) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)

    model.to(device)
    for _ in range(epochs):
        model.train()
        for imgs, labels in train_ldr:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

    # Validate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_ldr:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total   += imgs.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
class NASObjective:
    def __init__(self, proxy_epochs, batch_size, device, history_rows):
        self.proxy_epochs  = proxy_epochs
        self.batch_size    = batch_size
        self.device        = device
        self.history_rows  = history_rows

    def __call__(self, trial: optuna.Trial) -> float:
        config = sample_config(trial)
        resolution = config["input_resolution"]

        # Build model
        try:
            model = build_arch(config, num_classes=10)
        except Exception as e:
            trial.set_user_attr("reject_reason", f"build_failed: {e}")
            raise optuna.exceptions.TrialPruned()

        # -- Constraint 1: model size --
        size_mb = model_size_mb(model)
        if size_mb > MAX_MODEL_MB:
            trial.set_user_attr("reject_reason", f"size_mb={size_mb:.2f} > {MAX_MODEL_MB}")
            raise optuna.exceptions.TrialPruned()

        # -- Proxy train --
        try:
            train_ldr, val_ldr = get_loaders(resolution, self.batch_size, proxy_n=3000)
            val_acc = proxy_train(model, train_ldr, val_ldr, self.proxy_epochs, self.device)
        except Exception as e:
            trial.set_user_attr("reject_reason", f"train_failed: {e}")
            raise optuna.exceptions.TrialPruned()

        # -- Constraint 2: CPU latency --
        latency_ms = measure_cpu_latency(model.cpu(), resolution)
        if latency_ms > MAX_LATENCY_MS:
            trial.set_user_attr("reject_reason",
                                f"latency_ms={latency_ms:.2f} > {MAX_LATENCY_MS}")
            raise optuna.exceptions.TrialPruned()

        # -- Optional: peak memory --
        try:
            peak_mem_mb = measure_peak_memory_mb(model.cpu(), resolution)
        except Exception:
            peak_mem_mb = -1.0

        # Record history
        self.history_rows.append({
            "trial":        trial.number,
            "val_acc":      val_acc,
            "size_mb":      size_mb,
            "latency_ms":   latency_ms,
            "peak_mem_mb":  peak_mem_mb,
            "config":       config_to_str(config),
            **config,
        })
        trial.set_user_attr("val_acc",     val_acc)
        trial.set_user_attr("size_mb",     size_mb)
        trial.set_user_attr("latency_ms",  latency_ms)
        trial.set_user_attr("peak_mem_mb", peak_mem_mb)

        return val_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials",     type=int, default=20)
    parser.add_argument("--proxy-epochs", type=int, default=3)
    parser.add_argument("--batch-size",   type=int, default=64)
    parser.add_argument("--smoke-test",   action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_trials     = 3
        args.proxy_epochs = 1

    device = torch.device("cpu")   # CPU-only for fair hardware constraint measurement
    print(f"[NAS] Starting Optuna search | trials={args.n_trials} "
          f"proxy_epochs={args.proxy_epochs} | device={device}")
    print(f"[NAS] Constraints: size ≤ {MAX_MODEL_MB} MB | latency ≤ {MAX_LATENCY_MS} ms")

    os.makedirs(RESULTS, exist_ok=True)
    history_rows = []

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.NopPruner(),
        study_name="edge_nas",
    )
    objective = NASObjective(args.proxy_epochs, args.batch_size, device, history_rows)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # -- Save search history CSV --
    history_path = os.path.join(RESULTS, "search_history.csv")
    if history_rows:
        fieldnames = list(history_rows[0].keys())
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history_rows)
        print(f"[NAS] Search history → {history_path}")

    # -- Save best config --
    best_trial  = study.best_trial
    best_config = {k: v for k, v in best_trial.params.items()}
    best_config["_val_acc"]    = best_trial.value
    best_config["_trial_num"]  = best_trial.number

    best_config_path = os.path.join(RESULTS, "best_config.json")
    with open(best_config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"[NAS] Best config → {best_config_path}")
    print(f"[NAS] Best val_acc: {best_trial.value:.2f}%")
    print(f"[NAS] Config: {config_to_str(best_config)}")


if __name__ == "__main__":
    main()
