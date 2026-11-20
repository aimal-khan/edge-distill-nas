"""
Baseline student training — trains MobileNetV2, SqueezeNet, EfficientNet-Lite0
on CIFAR-100 WITHOUT knowledge distillation.

Usage:
    python train_baseline.py [--epochs 10] [--batch-size 128] [--smoke-test]
"""

import argparse
import os
import sys
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(__file__))
from models.students import STUDENT_REGISTRY, get_student

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "..", ".data")


def get_loaders(batch_size: int, smoke: bool):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_ds = torchvision.datasets.CIFAR100(DATA, train=True,  download=True, transform=train_tf)
    val_ds   = torchvision.datasets.CIFAR100(DATA, train=False, download=True, transform=val_tf)
    if smoke:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(512))
        val_ds   = Subset(val_ds,   range(128))
    train_ldr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_ldr   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    return train_ldr, val_ldr


def train_student(name, train_ldr, val_ldr, epochs, device, smoke):
    model     = get_student(name, num_classes=100, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc  = 0.0
    ckpt_path = os.path.join(BASE, "checkpoints", f"baseline_{name}.pth")
    log_path  = os.path.join(BASE, "results",     f"baseline_{name}_log.json")
    history   = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # -- train --
        model.train()
        tr_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_ldr:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total   += imgs.size(0)
        tr_acc = 100.0 * correct / total
        scheduler.step()

        # -- validate --
        model.eval()
        va_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_ldr:
                imgs, labels = imgs.to(device), labels.to(device)
                out  = model(imgs)
                loss = criterion(out, labels)
                va_loss   += loss.item() * imgs.size(0)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total   += imgs.size(0)
        va_acc = 100.0 * v_correct / v_total

        print(f"  [{name}] Ep {epoch}/{epochs} | "
              f"train {tr_acc:.1f}% | val {va_acc:.1f}% | {time.time()-t0:.1f}s")
        history.append({"epoch": epoch, "train_acc": tr_acc, "val_acc": va_acc})

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": va_acc, "model_name": name}, ckpt_path)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({"model": name, "best_val_acc": best_acc, "history": history}, f, indent=2)
    print(f"  [{name}] Best val acc: {best_acc:.2f}% → {ckpt_path}")
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Baseline] Device: {device} | Epochs: {args.epochs}")

    train_ldr, val_ldr = get_loaders(args.batch_size, args.smoke_test)

    results = {}
    for name in STUDENT_REGISTRY:
        print(f"\n── Training baseline: {name} ──")
        acc = train_student(name, train_ldr, val_ldr, args.epochs, device, args.smoke_test)
        results[name] = acc

    print("\n[Baseline] Summary:")
    for name, acc in results.items():
        print(f"  {name}: {acc:.2f}%")


if __name__ == "__main__":
    main()
