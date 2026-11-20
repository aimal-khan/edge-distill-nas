# Assignment 4 — Edge Computing & Intelligent Systems

**Roll:** 21i-1701_A | **Deadline:** 20-Nov-2026

---

## Repository Structure

```
assignment 04/
├── q1_knowledge_distillation/
│   ├── models/
│   │   ├── teacher.py          # ResNet-50 teacher (CIFAR-100)
│   │   ├── students.py         # MobileNetV2, SqueezeNet, EfficientNet-Lite0
│   │   └── distillation.py     # KDLoss (soft-target + feature distillation)
│   ├── train_teacher.py        # Train ResNet-50 teacher
│   ├── train_baseline.py       # Train students WITHOUT KD
│   ├── train_kd.py             # Train students WITH KD
│   ├── inference.py            # Unified student inference pipeline
│   ├── export_onnx.py          # Export + CPU ONNX inference
│   ├── benchmark.py            # Benchmark: latency/size/accuracy → CSV
│   ├── checkpoints/            # .pth and .onnx model files
│   └── results/                # Training logs, benchmark CSV
├── q2_nas/
│   ├── search_space.py         # Searchable CNN architecture space
│   ├── arch_generator.py       # Build nn.Module from config dict
│   ├── nas_search.py           # Optuna-based NAS with hardware constraints
│   ├── train_best.py           # Train best architecture from scratch
│   ├── inference_nas.py        # Inference + ONNX export for NAS model
│   ├── benchmark_nas.py        # Benchmark NAS model → CSV
│   └── results/                # search_history.csv, best_config.json, .pth, .onnx
├── docs/
│   └── report.pdf              # Technical report
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Execution — Q1: Knowledge Distillation

All commands run from the `assignment 04/` directory.

### Step 1: Train the ResNet-50 Teacher
```bash
cd q1_knowledge_distillation
python train_teacher.py --epochs 15 --batch-size 128
```
Output: `checkpoints/teacher_resnet50.pth`

### Step 2: Train Baseline Students (no KD)
```bash
python train_baseline.py --epochs 10 --batch-size 128
```
Output: `checkpoints/baseline_mobilenet.pth`, `baseline_squeezenet.pth`, `baseline_efficientnet.pth`

### Step 3: Train Students with Knowledge Distillation
```bash
python train_kd.py --epochs 10 --temperature 4.0 --alpha 0.5 --beta 0.1
```
Output: `checkpoints/kd_mobilenet.pth`, `kd_squeezenet.pth`, `kd_efficientnet.pth`

### Step 4: Run Unified Inference
```bash
python inference.py --model mobilenet
python inference.py --model squeezenet
python inference.py --model efficientnet
```

### Step 5: Export to ONNX + CPU Inference
```bash
python export_onnx.py
```
Output: `checkpoints/kd_mobilenet.onnx`, `kd_squeezenet.onnx`, `kd_efficientnet.onnx`

### Step 6: Benchmark All Models
```bash
python benchmark.py
```
Output: `results/benchmark_q1.csv`

---

## Execution — Q2: Neural Architecture Search

### Step 7: Run NAS Search (Optuna / Bayesian)
```bash
cd ../q2_nas
python nas_search.py --n-trials 20 --proxy-epochs 3
```
Output: `results/search_history.csv`, `results/best_config.json`

### Step 8: Train Best Architecture
```bash
python train_best.py --epochs 20 --batch-size 128
```
Output: `results/best_arch.pth`

### Step 9: NAS Inference + ONNX Export
```bash
python inference_nas.py
```
Output: `results/best_arch.onnx`

### Step 10: Benchmark NAS Model
```bash
python benchmark_nas.py
```
Output: `results/benchmark_nas.csv`

---

## Quick Smoke Test (all in ~5 minutes)

```bash
cd q1_knowledge_distillation
python train_teacher.py  --smoke-test
python train_baseline.py --smoke-test
python train_kd.py       --smoke-test
python export_onnx.py    --smoke-test
python inference.py      --model mobilenet --smoke-test
python benchmark.py      --smoke-test

cd ../q2_nas
python nas_search.py    --smoke-test
python train_best.py    --smoke-test
python inference_nas.py --smoke-test
python benchmark_nas.py --smoke-test
```

---

## Hardware Constraints (Q2 NAS)

| Constraint         | Limit    |
|--------------------|----------|
| Model storage      | ≤ 10 MB  |
| CPU inference lat. | ≤ 150 ms |
| Peak memory        | Recorded |

Architectures violating size or latency constraints are rejected (Optuna TrialPruned).

---

## KD Framework Details (Q1)

| Component         | Implementation                                |
|-------------------|-----------------------------------------------|
| Soft-target KD    | KL divergence with temperature-scaled softmax |
| Temperature       | T = 4.0 (configurable via `--temperature`)    |
| Hard-label loss   | CrossEntropyLoss                              |
| Combined training | L = α·L_soft + (1−α)·L_hard + β·L_feat       |
| Feature dist.     | L2 on projected intermediate feature maps     |
