#!/usr/bin/env python3
"""
run_all.py — Master pipeline script: runs the full Assignment 4 pipeline
in the correct order, chaining Q1 and Q2 steps.

Usage:
    python run_all.py [--smoke-test] [--q1-only] [--q2-only]
                      [--teacher-epochs 15] [--student-epochs 10]
                      [--nas-trials 20] [--nas-proxy-epochs 3]
                      [--best-epochs 20]
"""

import argparse
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
Q1   = os.path.join(BASE, "q1_knowledge_distillation")
Q2   = os.path.join(BASE, "q2_nas")
PY   = sys.executable


def run(cmd: list[str], cwd: str):
    print(f"\n{'='*60}")
    print(f"RUN: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"ERROR: command failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test",         action="store_true")
    parser.add_argument("--q1-only",            action="store_true")
    parser.add_argument("--q2-only",            action="store_true")
    parser.add_argument("--teacher-epochs",     type=int, default=15)
    parser.add_argument("--student-epochs",     type=int, default=10)
    parser.add_argument("--nas-trials",         type=int, default=20)
    parser.add_argument("--nas-proxy-epochs",   type=int, default=3)
    parser.add_argument("--best-epochs",        type=int, default=20)
    args = parser.parse_args()

    smoke = ["--smoke-test"] if args.smoke_test else []

    # ── Q1 Knowledge Distillation ──────────────────────────────────────────
    if not args.q2_only:
        # 1. Train teacher
        run([PY, "train_teacher.py",
             "--epochs", str(1 if args.smoke_test else args.teacher_epochs),
             "--batch-size", "128"] + smoke, Q1)

        # 2. Train baselines
        run([PY, "train_baseline.py",
             "--epochs", str(1 if args.smoke_test else args.student_epochs)] + smoke, Q1)

        # 3. Train KD students
        run([PY, "train_kd.py",
             "--epochs", str(1 if args.smoke_test else args.student_epochs),
             "--temperature", "4.0", "--alpha", "0.5", "--beta", "0.1"] + smoke, Q1)

        # 4. Inference
        for model in ("mobilenet", "squeezenet", "efficientnet"):
            run([PY, "inference.py", "--model", model] + smoke, Q1)

        # 5. ONNX export
        run([PY, "export_onnx.py"] + smoke, Q1)

        # 6. Benchmark Q1
        run([PY, "benchmark.py"] + smoke, Q1)

    # ── Q2 NAS ────────────────────────────────────────────────────────────
    if not args.q1_only:
        # 7. NAS search
        run([PY, "nas_search.py",
             "--n-trials",     str(3 if args.smoke_test else args.nas_trials),
             "--proxy-epochs", str(1 if args.smoke_test else args.nas_proxy_epochs),
             "--batch-size", "64"] + smoke, Q2)

        # 8. Train best architecture
        run([PY, "train_best.py",
             "--epochs", str(1 if args.smoke_test else args.best_epochs)] + smoke, Q2)

        # 9. NAS inference + ONNX
        run([PY, "inference_nas.py"] + smoke, Q2)

        # 10. Benchmark NAS
        run([PY, "benchmark_nas.py"] + smoke, Q2)

    print("\n" + "="*60)
    print("ALL STEPS COMPLETE")
    print("="*60)
    print(f"Q1 results:  {Q1}/results/")
    print(f"Q2 results:  {Q2}/results/")
    print(f"Q1 models:   {Q1}/checkpoints/")
    print(f"Q2 models:   {Q2}/results/")


if __name__ == "__main__":
    main()
