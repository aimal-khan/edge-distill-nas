# Edge Intelligence: Distillation and Hardware-Aware NAS (`edge-distill-nas`)

End-to-end implementation of Knowledge Distillation (CIFAR-100) and Hardware-Aware Neural Architecture Search (CIFAR-10) for compute-constrained edge deployment.

**Author:** 21i-1701_A  
**Course:** Edge Computing & Intelligent Systems  
**Repository:** [https://github.com/aimal-khan/edge-distill-nas](https://github.com/aimal-khan/edge-distill-nas)  

---

## Architecture Overview

```
edge-distill-nas/
├── q1_knowledge_distillation/
│   ├── models/
│   │   ├── teacher.py          # ResNet-50 feature source (CIFAR-100)
│   │   ├── students.py         # MobileNetV2, SqueezeNet 1.1, EfficientNet-Lite0
│   │   └── distillation.py     # Composite loss (soft KL + hard CE + feature MSE)
│   ├── train_teacher.py        # Teacher training pipeline
│   ├── train_baseline.py       # Standalone student baseline training
│   ├── train_kd.py             # Knowledge distillation training pipeline
│   ├── inference.py            # Unified single-image/batch inference CLI
│   ├── export_onnx.py          # Export to ONNX & runtime parity validation
│   ├── benchmark.py            # Automated CPU latency & memory benchmarking
│   ├── checkpoints/            # Serialized PyTorch (.pth) & ONNX (.onnx) models
│   └── results/                # Benchmark CSVs and epoch training logs
├── q2_nas/
│   ├── search_space.py         # Search space bounds & hyperparameter specs
│   ├── arch_generator.py       # Dynamic PyTorch module synthesizer
│   ├── nas_search.py           # Optuna Bayesian TPE engine with hardware pruning
│   ├── train_best.py           # Scratch training for optimal discovered topology
│   ├── inference_nas.py        # NAS inference CLI & ONNX export
│   ├── benchmark_nas.py        # Latency & memory benchmark for NAS model
│   └── results/                # search_history.csv, best_config.json, weights
├── docs/
│   ├── report.pdf              # 3-page formal technical report (ReportLab)
│   ├── report.md               # Complete Markdown documentation & formulas
│   └── generate_pdf_report.py  # Standalone PDF report compiler
├── requirements.txt            # Environment dependencies
├── run_all.py                  # Single-command end-to-end execution runner
└── README.md                   # Operational guide
```

---

## Environment Setup

Ensure Python 3.9+ is active:

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `torchvision`, `onnx`, `onnxruntime`, `optuna`, `reportlab`, `numpy`, `pillow`.

---

## Quick Smoke Verification (~3 Minutes)

Run the end-to-end verification suite using proxy subsets:

```bash
python run_all.py --smoke-test
```

Or execute per-module smoke checks:

```bash
# Question 1: Distillation Pipeline
cd q1_knowledge_distillation
python train_teacher.py  --smoke-test
python train_baseline.py --smoke-test
python train_kd.py       --smoke-test
python export_onnx.py    --smoke-test
python inference.py      --model mobilenet --smoke-test
python benchmark.py      --smoke-test

# Question 2: Neural Architecture Search
cd ../q2_nas
python nas_search.py    --smoke-test
python train_best.py    --smoke-test
python inference_nas.py --smoke-test
python benchmark_nas.py --smoke-test
```

---

## Full Reproduction Guide

### Phase 1: Knowledge Distillation (CIFAR-100)

1. **Train ResNet-50 Teacher:**
   ```bash
   cd q1_knowledge_distillation
   python train_teacher.py --epochs 15 --batch-size 128
   ```
   Outputs checkpoint to `checkpoints/teacher_resnet50.pth`.

2. **Train Baseline Students (without Distillation):**
   ```bash
   python train_baseline.py --model all --epochs 10 --batch-size 128
   ```
   Saves `baseline_mobilenet.pth`, `baseline_squeezenet.pth`, and `baseline_efficientnet.pth`.

3. **Train Students with Knowledge Distillation:**
   ```bash
   python train_kd.py --model all --epochs 10 --temperature 4.0 --alpha 0.5 --beta 0.1
   ```
   Saves `kd_mobilenet.pth`, `kd_squeezenet.pth`, and `kd_efficientnet.pth`.

4. **Export ONNX & Verify Parity:**
   ```bash
   python export_onnx.py
   ```
   Exports `kd_*.onnx` and validates numerical parity ($\max |\mathbf{y}_{\text{pt}} - \mathbf{y}_{\text{onnx}}| < 10^{-4}$).

5. **Run Unified Inference:**
   ```bash
   python inference.py --model mobilenet
   python inference.py --model squeezenet
   python inference.py --model efficientnet
   ```

6. **Generate Q1 Benchmark Report:**
   ```bash
   python benchmark.py
   ```
   Outputs latency and accuracy metrics to `results/benchmark_q1.csv`.

---

### Phase 2: Hardware-Aware NAS (CIFAR-10)

1. **Execute Bayesian Search with Optuna:**
   ```bash
   cd ../q2_nas
   python nas_search.py --n-trials 20 --proxy-epochs 3
   ```
   Generates `results/search_history.csv` and serializes optimal parameters to `results/best_config.json`. Enforces hard constraints: size $\le 10$ MB, latency $\le 150$ ms.

2. **Train Optimal Architecture from Scratch:**
   ```bash
   python train_best.py --epochs 20 --batch-size 128
   ```
   Saves retrained model to `results/best_arch.pth`.

3. **Export NAS ONNX & Verify Edge Inference:**
   ```bash
   python inference_nas.py
   ```
   Generates `results/best_arch.onnx`.

4. **Benchmark Discovered Model:**
   ```bash
   python benchmark_nas.py
   ```
   Generates `results/benchmark_nas.csv`.

---

## Compiling Technical PDF Documentation

To recompile the formal 3-page technical report:

```bash
cd docs
python generate_pdf_report.py
```

Outputs compiled PDF report to `docs/report.pdf`.

---

## Experimental Benchmark Summary

### Q1 Distillation Benchmark (CIFAR-100)

| Topology | Mode | Test Acc (%) | PyTorch Latency | ONNX Runtime CPU | Speedup | Storage (.pth / .onnx) |
|---|---|---|---|---|---|---|
| ResNet-50 | Teacher | 46.35% | 18.24 ms | — | — | 95.17 MB / — |
| MobileNetV2 | Baseline | 2.73% | 33.56 ms | — | — | 9.65 MB / — |
| MobileNetV2 | Distilled (KD) | **4.69%** | 34.24 ms | **1.17 ms** | **29.3×** | 20.14 MB / 9.38 MB |
| SqueezeNet 1.1 | Baseline | 3.12% | 1.70 ms | — | — | 3.12 MB / — |
| SqueezeNet 1.1 | Distilled (KD) | 0.39% | 1.15 ms | **0.25 ms** | **4.6×** | 7.32 MB / 3.11 MB |
| EfficientNet-Lite0 | Baseline | 1.56% | 203.82 ms | — | — | 16.84 MB / — |
| EfficientNet-Lite0 | Distilled (KD) | **1.95%** | 202.26 ms | **2.47 ms** | **81.9×** | 27.33 MB / 13.95 MB |

### Q2 Discovered Architecture Specs (CIFAR-10)

- **Topology:** 3 stages, 16 base filters (multiplier 1.0), $3\times3$ Depthwise Separable, GELU, $24\times24$ resolution.
- **Weights & Footprint:** 2,714 parameters (14.2 KB `.pth` / 9.6 KB `.onnx`).
- **Validation Accuracy:** 41.22% (from scratch).
- **Inference Latency:** 0.85 ms (PyTorch CPU) $\to$ **0.03 ms** (ONNX Runtime CPU, $28.3\times$ speedup).
