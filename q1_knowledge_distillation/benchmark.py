"""
Benchmarking script for all Q1 student models.
Measures:
  - Inference latency (mean ms over 100 runs)
  - Model size (MB, .pth and .onnx)
  - Top-1 accuracy on CIFAR-100 test set

Outputs: results/benchmark_q1.csv

Usage:
    python benchmark.py [--batch-size 32] [--n-runs 100] [--smoke-test]
"""

import argparse
import os
import sys
import time
import csv

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from models.students import get_student, STUDENT_REGISTRY

BASE     = os.path.dirname(__file__)
CKPT_DIR = os.path.join(BASE, "checkpoints")
DATA     = os.path.join(BASE, "..", ".data")
RESULT   = os.path.join(BASE, "results", "benchmark_q1.csv")

MEAN = (0.5071, 0.4867, 0.4408)
STD  = (0.2675, 0.2565, 0.2761)


def get_loader(batch_size, smoke):
    val_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    ds     = torchvision.datasets.CIFAR100(DATA, train=False, download=True, transform=val_tf)
    if smoke:
        from torch.utils.data import Subset
        ds = Subset(ds, range(256))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


def measure_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total   += imgs.size(0)
    return 100.0 * correct / total


def measure_latency(model, device, n_runs=100, input_shape=(1, 3, 32, 32)):
    model.eval()
    dummy = torch.randn(*input_shape).to(device)
    lats  = []
    with torch.no_grad():
        for _ in range(n_runs + 5):
            t0 = time.perf_counter()
            model(dummy)
            lats.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(lats[5:])), float(np.std(lats[5:]))


def measure_onnx_latency(onnx_path, n_runs=100):
    import onnxruntime as ort
    sess  = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.rand(1, 3, 32, 32).astype(np.float32)
    lats  = []
    for _ in range(n_runs + 5):
        t0 = time.perf_counter()
        sess.run(None, {"input": dummy})
        lats.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(lats[5:])), float(np.std(lats[5:]))


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-runs",     type=int, default=100)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_runs = 10

    device = torch.device("cpu")   # CPU benchmark for fair comparison
    loader = get_loader(args.batch_size, args.smoke_test)

    rows = []
    headers = [
        "model", "mode", "top1_acc_%",
        "pth_size_mb", "onnx_size_mb", "params_M",
        "pytorch_latency_ms", "pytorch_latency_std",
        "onnx_latency_ms", "onnx_latency_std",
    ]

    for name in STUDENT_REGISTRY:
        for mode in ("baseline", "kd"):
            print(f"\n── Benchmarking {mode}-{name} ──")
            model = get_student(name, num_classes=100, pretrained=False).to(device)
            ckpt  = os.path.join(CKPT_DIR, f"{mode}_{name}.pth")
            onnx  = os.path.join(CKPT_DIR, f"kd_{name}.onnx")     # only KD models exported

            if os.path.exists(ckpt):
                state = torch.load(ckpt, map_location=device)
                model.load_state_dict(state["model_state"])
            else:
                print(f"  WARNING: {ckpt} not found; using random weights.")

            model.eval()
            acc              = measure_accuracy(model, loader, device)
            pt_lat, pt_std   = measure_latency(model, device, n_runs=args.n_runs)
            pth_mb           = os.path.getsize(ckpt) / 1e6 if os.path.exists(ckpt) else 0
            params_M         = param_count(model) / 1e6

            onnx_lat, onnx_std = (0.0, 0.0)
            onnx_mb            = 0.0
            if mode == "kd" and os.path.exists(onnx):
                onnx_mb            = os.path.getsize(onnx) / 1e6
                onnx_lat, onnx_std = measure_onnx_latency(onnx, n_runs=args.n_runs)

            row = {
                "model":                name,
                "mode":                 mode,
                "top1_acc_%":           f"{acc:.2f}",
                "pth_size_mb":          f"{pth_mb:.2f}",
                "onnx_size_mb":         f"{onnx_mb:.2f}",
                "params_M":             f"{params_M:.2f}",
                "pytorch_latency_ms":   f"{pt_lat:.2f}",
                "pytorch_latency_std":  f"{pt_std:.2f}",
                "onnx_latency_ms":      f"{onnx_lat:.2f}",
                "onnx_latency_std":     f"{onnx_std:.2f}",
            }
            rows.append(row)
            print(f"  acc={acc:.2f}% | PyTorch lat={pt_lat:.2f}ms | "
                  f"ONNX lat={onnx_lat:.2f}ms | size={pth_mb:.2f}MB")

    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[Benchmark] Results saved → {RESULT}")


if __name__ == "__main__":
    main()
