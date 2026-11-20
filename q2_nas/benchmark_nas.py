"""
NAS model benchmarking script.
Measures: accuracy, latency, model size for the best NAS architecture.
Outputs: results/benchmark_nas.csv

Usage:
    python benchmark_nas.py [--n-runs 100] [--batch-size 64] [--smoke-test]
"""

import argparse
import csv
import json
import os
import sys
import time

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_generator import build_arch

BASE    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(BASE, "..", ".data")
RESULTS = os.path.join(BASE, "results")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs",     type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_runs = 10

    # Load model
    config_path = os.path.join(RESULTS, "best_config.json")
    ckpt_path   = os.path.join(RESULTS, "best_arch.pth")
    onnx_path   = os.path.join(RESULTS, "best_arch.onnx")

    with open(config_path) as f:
        config = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    resolution = config["input_resolution"]

    model = build_arch(config, num_classes=10)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Accuracy
    val_tf = T.Compose([
        T.Resize(resolution),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    val_ds = torchvision.datasets.CIFAR10(DATA, train=False, download=True, transform=val_tf)
    if args.smoke_test:
        from torch.utils.data import Subset
        val_ds = Subset(val_ds, range(256))
    val_ldr = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_ldr:
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total   += imgs.size(0)
    acc = 100.0 * correct / total
    print(f"[NAS Benchmark] Top-1 Accuracy: {acc:.2f}%")

    # PyTorch latency
    dummy = torch.randn(1, 3, resolution, resolution)
    lats  = []
    with torch.no_grad():
        for _ in range(args.n_runs + 5):
            t0 = time.perf_counter()
            model(dummy)
            lats.append((time.perf_counter() - t0) * 1000)
    pt_lat = float(np.mean(lats[5:]))
    pt_std = float(np.std(lats[5:]))
    print(f"[NAS Benchmark] PyTorch CPU latency: {pt_lat:.2f} ± {pt_std:.2f} ms")

    # ONNX latency
    onnx_lat, onnx_std = 0.0, 0.0
    if os.path.exists(onnx_path):
        import onnxruntime as ort
        sess  = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        dummy_np = np.random.rand(1, 3, resolution, resolution).astype(np.float32)
        olats    = []
        for _ in range(args.n_runs + 5):
            t0 = time.perf_counter()
            sess.run(None, {"input": dummy_np})
            olats.append((time.perf_counter() - t0) * 1000)
        onnx_lat = float(np.mean(olats[5:]))
        onnx_std = float(np.std(olats[5:]))
        print(f"[NAS Benchmark] ONNX CPU latency:    {onnx_lat:.2f} ± {onnx_std:.2f} ms")

    # Sizes
    pth_mb  = os.path.getsize(ckpt_path) / 1e6
    onnx_mb = os.path.getsize(onnx_path) / 1e6 if os.path.exists(onnx_path) else 0
    params_M = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[NAS Benchmark] Checkpoint size: {pth_mb:.2f} MB | ONNX size: {onnx_mb:.2f} MB | Params: {params_M:.2f}M")

    # Save CSV
    csv_path = os.path.join(RESULTS, "benchmark_nas.csv")
    rows = [{
        "model":               "NAS_best",
        "top1_acc_%":          f"{acc:.2f}",
        "pth_size_mb":         f"{pth_mb:.2f}",
        "onnx_size_mb":        f"{onnx_mb:.2f}",
        "params_M":            f"{params_M:.2f}",
        "pytorch_latency_ms":  f"{pt_lat:.2f}",
        "pytorch_latency_std": f"{pt_std:.2f}",
        "onnx_latency_ms":     f"{onnx_lat:.2f}",
        "onnx_latency_std":    f"{onnx_std:.2f}",
        "resolution":          resolution,
        "config":              json.dumps(config),
    }]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[NAS Benchmark] Results → {csv_path}")


if __name__ == "__main__":
    main()
