"""
Train the best NAS-discovered architecture from scratch on full CIFAR-10.

Loads best_config.json, builds architecture, trains for --epochs epochs.
Saves checkpoint to results/best_arch.pth.

Usage:
    python train_best.py [--epochs 20] [--batch-size 128] [--smoke-test]
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_generator import build_arch

BASE    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(BASE, "..", ".data")
RESULTS = os.path.join(BASE, "results")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_loaders(resolution: int, batch_size: int, smoke: bool):
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
    if smoke:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(512))
        val_ds   = Subset(val_ds,   range(128))
    train_ldr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_ldr   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    return train_ldr, val_ldr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 1

    # Load best config
    config_path = os.path.join(RESULTS, "best_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"best_config.json not found at {config_path}. "
                                "Run nas_search.py first.")
    with open(config_path) as f:
        config = json.load(f)
    # Remove metadata keys
    config = {k: v for k, v in config.items() if not k.startswith("_")}

    resolution = config["input_resolution"]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TrainBest] Device: {device} | Epochs: {args.epochs} | Resolution: {resolution}")
    print(f"[TrainBest] Config: {config}")

    model     = build_arch(config, num_classes=10).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"[TrainBest] Parameters: {n_params/1e6:.2f}M")

    train_ldr, val_ldr = get_loaders(resolution, args.batch_size, args.smoke_test)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc  = 0.0
    ckpt_path = os.path.join(RESULTS, "best_arch.pth")
    history   = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # -- train --
        model.train()
        correct, total = 0, 0
        for imgs, labels in train_ldr:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total   += imgs.size(0)
        # shortcut: reuse forward above is fine for accuracy display only
        tr_acc = 100.0 * correct / total
        scheduler.step()

        # -- validate --
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_ldr:
                imgs, labels = imgs.to(device), labels.to(device)
                v_correct += (model(imgs).argmax(1) == labels).sum().item()
                v_total   += imgs.size(0)
        va_acc = 100.0 * v_correct / v_total

        print(f"  Ep {epoch}/{args.epochs} | train {tr_acc:.1f}% | val {va_acc:.1f}% | {time.time()-t0:.1f}s")
        history.append({"epoch": epoch, "train_acc": tr_acc, "val_acc": va_acc})

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_acc": va_acc, "config": config,
            }, ckpt_path)

    import json as _json
    log_path = os.path.join(RESULTS, "best_arch_training_log.json")
    with open(log_path, "w") as f:
        _json.dump({"best_val_acc": best_acc, "history": history, "config": config}, f, indent=2)

    print(f"\n[TrainBest] Best val acc: {best_acc:.2f}% → {ckpt_path}")


if __name__ == "__main__":
    main()
