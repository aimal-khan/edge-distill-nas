"""
Knowledge Distillation training — trains all three student architectures
using the KD framework (soft targets + feature distillation).

Requires pre-trained teacher checkpoint: checkpoints/teacher_resnet50.pth

Usage:
    python train_kd.py [--epochs 10] [--batch-size 128]
                       [--temperature 4.0] [--alpha 0.5] [--beta 0.1]
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
from models.teacher     import get_teacher
from models.students    import STUDENT_REGISTRY, get_student
from models.distillation import KDLoss, FeatureAdaptor

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "..", ".data")

# Channel counts: needed to build FeatureAdaptor per student
STUDENT_FEAT_CHANNELS = {
    "mobilenet":    1280,
    "squeezenet":   512,
    "efficientnet": 1280,
}
TEACHER_FEAT_CHANNELS = 2048


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


def train_kd_student(name, teacher, train_ldr, val_ldr, args, device):
    student  = get_student(name, num_classes=100, pretrained=True).to(device)
    adaptor  = FeatureAdaptor(STUDENT_FEAT_CHANNELS[name], TEACHER_FEAT_CHANNELS).to(device)
    kd_loss  = KDLoss(temperature=args.temperature, alpha=args.alpha, beta=args.beta)

    params    = list(student.parameters()) + list(adaptor.parameters())
    optimizer = optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc  = 0.0
    ckpt_path = os.path.join(BASE, "checkpoints", f"kd_{name}.pth")
    log_path  = os.path.join(BASE, "results",     f"kd_{name}_log.json")
    history   = []

    teacher.eval()   # teacher always in eval mode

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── train ──
        student.train()
        adaptor.train()
        tr_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_ldr:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                t_logits, t_feat = teacher(imgs, return_features=True)

            s_logits, s_feat = student(imgs, return_features=True)
            s_proj, t_proj   = adaptor(s_feat, t_feat)

            loss = kd_loss(s_logits, t_logits, labels, s_proj, t_proj)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * imgs.size(0)
            correct += (s_logits.argmax(1) == labels).sum().item()
            total   += imgs.size(0)

        tr_acc = 100.0 * correct / total
        scheduler.step()

        # ── validate ──
        student.eval()
        va_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_ldr:
                imgs, labels = imgs.to(device), labels.to(device)
                s_logits = student(imgs)
                va_loss   += nn.CrossEntropyLoss()(s_logits, labels).item() * imgs.size(0)
                v_correct += (s_logits.argmax(1) == labels).sum().item()
                v_total   += imgs.size(0)
        va_acc = 100.0 * v_correct / v_total

        print(f"  [KD-{name}] Ep {epoch}/{args.epochs} | "
              f"train {tr_acc:.1f}% | val {va_acc:.1f}% | {time.time()-t0:.1f}s", flush=True)
        history.append({"epoch": epoch, "train_acc": tr_acc, "val_acc": va_acc})

        if va_acc >= best_acc or epoch == 1:
            best_acc = max(best_acc, va_acc)
            torch.save({
                "epoch": epoch,
                "model_state":   student.state_dict(),
                "adaptor_state": adaptor.state_dict(),
                "val_acc": va_acc, "model_name": name,
            }, ckpt_path)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({"model": name, "mode": "kd", "best_val_acc": best_acc,
                   "history": history}, f, indent=2)
    print(f"  [KD-{name}] Best val acc: {best_acc:.2f}% → {ckpt_path}")
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       choices=list(STUDENT_REGISTRY) + ["all"], default="all",
                        help="Which student model to train ('all' or specific)")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch-size",  type=int,   default=128)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha",       type=float, default=0.5,
                        help="Weight on soft KD loss (0-1)")
    parser.add_argument("--beta",        type=float, default=0.1,
                        help="Weight on feature distillation loss")
    parser.add_argument("--smoke-test",  action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[KD] Device: {device} | Epochs: {args.epochs} | T={args.temperature}")

    # Load teacher
    teacher_ckpt = os.path.join(BASE, "checkpoints", "teacher_resnet50.pth")
    teacher = get_teacher(num_classes=100, pretrained=False).to(device)
    if os.path.exists(teacher_ckpt):
        ckpt = torch.load(teacher_ckpt, map_location=device)
        teacher.load_state_dict(ckpt["model_state"])
        print(f"[KD] Loaded teacher from {teacher_ckpt} (val_acc={ckpt.get('val_acc', 'n/a')})")
    else:
        print("[KD] WARNING: Teacher checkpoint not found — using pretrained ImageNet weights only.")
        teacher = get_teacher(num_classes=100, pretrained=True).to(device)

    train_ldr, val_ldr = get_loaders(args.batch_size, args.smoke_test)

    models_to_train = list(STUDENT_REGISTRY) if args.model == "all" else [args.model]
    results = {}
    for name in models_to_train:
        print(f"\n── KD training: {name} ──")
        acc = train_kd_student(name, teacher, train_ldr, val_ldr, args, device)
        results[name] = acc

    print("\n[KD] Summary:")
    for name, acc in results.items():
        print(f"  {name}: {acc:.2f}%")


if __name__ == "__main__":
    main()
