"""
Export all KD-trained student models to ONNX and validate CPU inference
using ONNX Runtime.

Usage:
    python export_onnx.py [--smoke-test]
"""

import argparse
import os
import sys
import time

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from models.students import get_student, STUDENT_REGISTRY

BASE     = os.path.dirname(__file__)
CKPT_DIR = os.path.join(BASE, "checkpoints")


def export_model(name: str, model: torch.nn.Module, device):
    onnx_path = os.path.join(CKPT_DIR, f"kd_{name}.onnx")
    dummy     = torch.randn(1, 3, 32, 32).to(device)

    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"  [{name}] Exported → {onnx_path}  ({size_mb:.2f} MB)")
    return onnx_path


def onnx_inference(onnx_path: str, dummy_np: np.ndarray):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    t0   = time.perf_counter()
    out  = sess.run(None, {"input": dummy_np})[0]
    lat  = (time.perf_counter() - t0) * 1000
    return out, lat


def validate(name: str, model: torch.nn.Module, onnx_path: str, device):
    """Check that ONNX and PyTorch outputs agree within tolerance."""
    dummy    = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        pt_out = model(dummy.to(device)).cpu().numpy()
    onnx_out, lat = onnx_inference(onnx_path, dummy.numpy().astype(np.float32))

    max_diff = np.abs(pt_out - onnx_out).max()
    status   = "✓ OK" if max_diff < 1e-3 else f"✗ MISMATCH (max_diff={max_diff:.6f})"
    print(f"  [{name}] ONNX validation: {status} | ONNX latency: {lat:.2f} ms")
    return max_diff < 1e-3


def benchmark_onnx(name: str, onnx_path: str, n_runs: int = 50):
    """Run n_runs inferences and report mean latency."""
    import onnxruntime as ort
    sess  = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.rand(1, 3, 32, 32).astype(np.float32)
    lats  = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"input": dummy})
        lats.append((time.perf_counter() - t0) * 1000)
    mean_lat = np.mean(lats[5:])    # skip first 5 warm-up runs
    print(f"  [{name}] ONNX CPU mean latency ({n_runs} runs): {mean_lat:.2f} ms")
    return mean_lat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    device  = torch.device("cpu")   # ONNX export must be on CPU for portability
    n_runs  = 10 if args.smoke_test else 50

    print("[ONNX Export] Exporting all KD student models...")

    for name in STUDENT_REGISTRY:
        print(f"\n── {name} ──")
        model = get_student(name, num_classes=100, pretrained=False).to(device)
        ckpt_path = os.path.join(CKPT_DIR, f"kd_{name}.pth")

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            print(f"  Loaded checkpoint: {ckpt_path}")
        else:
            print(f"  WARNING: No KD checkpoint for {name}; exporting with pretrained ImageNet weights.")
            model = get_student(name, num_classes=100, pretrained=True).to(device)

        model.eval()
        onnx_path = export_model(name, model, device)
        validate(name, model, onnx_path, device)
        benchmark_onnx(name, onnx_path, n_runs=n_runs)

    print("\n[ONNX Export] Done. All ONNX models saved in:", CKPT_DIR)


if __name__ == "__main__":
    main()
