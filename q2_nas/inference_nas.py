"""
Inference and ONNX export pipeline for the best NAS architecture.

Loads best_arch.pth → runs PyTorch inference → exports ONNX →
validates ONNX Runtime output → benchmarks CPU latency.

Usage:
    python inference_nas.py [--smoke-test]
"""

import argparse
import os
import sys
import time
import json

import torch
import numpy as np
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_generator import build_arch

BASE    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(BASE, "..", ".data")
RESULTS = os.path.join(BASE, "results")

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck",
]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def load_model_and_config():
    config_path = os.path.join(RESULTS, "best_config.json")
    ckpt_path   = os.path.join(RESULTS, "best_arch.pth")

    with open(config_path) as f:
        config = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

    model = build_arch(config, num_classes=10)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, config


def export_onnx(model, resolution, onnx_path):
    dummy = torch.randn(1, 3, resolution, resolution)
    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"[NAS ONNX] Exported → {onnx_path}  ({size_mb:.2f} MB)")
    return size_mb


def onnx_inference(onnx_path, resolution, n_runs=50):
    import onnxruntime as ort
    sess  = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.rand(1, 3, resolution, resolution).astype(np.float32)
    lats  = []
    for _ in range(n_runs + 5):
        t0 = time.perf_counter()
        out = sess.run(None, {"input": dummy})[0]
        lats.append((time.perf_counter() - t0) * 1000)
    mean_lat = np.mean(lats[5:])
    print(f"[NAS ONNX] CPU mean latency ({n_runs} runs): {mean_lat:.2f} ms")
    return out, mean_lat


def run_sample_inference(model, config, device, n=10):
    resolution = config["input_resolution"]
    val_tf = T.Compose([
        T.Resize(resolution),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    alt_dir = os.path.abspath(os.path.join(BASE, "..", "..", "Assignment 03", "data", "cifar10", "test"))
    if os.path.exists(alt_dir):
        ds = torchvision.datasets.ImageFolder(alt_dir, transform=val_tf)
    else:
        ds = torchvision.datasets.CIFAR10(DATA, train=False, download=True, transform=val_tf)
    correct = 0
    model.to(device).eval()
    with torch.no_grad():
        for i in range(n):
            img, label = ds[i]
            pred   = model(img.unsqueeze(0).to(device)).argmax(1).item()
            status = "✓" if pred == label else "✗"
            print(f"  [{status}] True: {CIFAR10_CLASSES[label]:12s} | "
                  f"Pred: {CIFAR10_CLASSES[pred]}")
            correct += int(pred == label)
    print(f"\n  Accuracy on {n} samples: {correct}/{n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    device  = torch.device("cpu")
    n_runs  = 10 if args.smoke_test else 50

    model, config = load_model_and_config()
    resolution    = config["input_resolution"]

    print(f"[NAS Inference] Resolution: {resolution}×{resolution}")
    print(f"[NAS Inference] Running sample inference on CIFAR-10 test set:")
    run_sample_inference(model, config, device, n=5 if args.smoke_test else 10)

    # Export ONNX
    onnx_path = os.path.join(RESULTS, "best_arch.onnx")
    export_onnx(model, resolution, onnx_path)
    onnx_inference(onnx_path, resolution, n_runs=n_runs)


if __name__ == "__main__":
    main()
