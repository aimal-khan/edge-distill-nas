"""
Train the ResNet-50 teacher on CIFAR-100.

Usage:
    python train_teacher.py [--epochs 15] [--batch-size 128] [--lr 0.01]
                            [--smoke-test]
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
from models.teacher import get_teacher

# ── paths ─────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(__file__)
CKPT   = os.path.join(BASE, "checkpoints", "teacher_resnet50.pth")
RESULT = os.path.join(BASE, "results", "teacher_training_log.json")
DATA   = os.path.join(BASE, "..", ".data")


def get_loaders(batch_size: int, smoke: bool):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_ds = torchvision.datasets.CIFAR100(DATA, train=True,  download=True, transform=train_tf)
    val_ds   = torchvision.datasets.CIFAR100(DATA, train=False, download=True, transform=val_tf)

    if smoke:
        # Use tiny subset for smoke tests
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(512))
        val_ds   = Subset(val_ds,   range(128))

    train_ldr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=2, pin_memory=True)
    val_ldr   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)
    return train_ldr, val_ldr


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=0.01)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Teacher] Device: {device} | Epochs: {args.epochs}")

    train_ldr, val_ldr = get_loaders(args.batch_size, args.smoke_test)

    model     = get_teacher(num_classes=100, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_ldr, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_ldr, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Ep {epoch:3d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.2f}% | "
              f"val loss {va_loss:.4f} acc {va_acc:.2f}% | {elapsed:.1f}s")

        row = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
               "val_loss": va_loss, "val_acc": va_acc}
        history.append(row)

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": va_acc, "args": vars(args)}, CKPT)
            print(f"  → Saved checkpoint (val_acc={va_acc:.2f}%)")

    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w") as f:
        json.dump({"best_val_acc": best_acc, "history": history}, f, indent=2)
    print(f"\n[Teacher] Best val acc: {best_acc:.2f}%  | Checkpoint: {CKPT}")


if __name__ == "__main__":
    main()
