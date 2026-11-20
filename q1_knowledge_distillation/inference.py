"""
Unified inference pipeline for all Q1 student models.

Usage:
    python inference.py --model mobilenet [--checkpoint path/to/ckpt.pth]
                        [--image path/to/img.png] [--smoke-test]
"""

import argparse
import os
import sys
import time

import torch
import torchvision.transforms as T
from PIL import Image
import torchvision

sys.path.insert(0, os.path.dirname(__file__))
from models.students import get_student, STUDENT_REGISTRY

BASE     = os.path.dirname(__file__)
DATA     = os.path.join(BASE, "..", ".data")
CKPT_DIR = os.path.join(BASE, "checkpoints")

# CIFAR-100 class names
CIFAR100_CLASSES = [
    "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle",
    "bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar","cattle",
    "chair","chimpanzee","clock","cloud","cockroach","couch","cra","crocodile","cup","dinosaur",
    "dolphin","elephant","flatfish","forest","fox","girl","hamster","house","kangaroo","keyboard",
    "lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree","motorcycle","mountain",
    "mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear","pickup_truck","pine_tree",
    "plain","plate","poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket",
    "rose","sea","seal","shark","shrew","skunk","skyscraper","snail","snake","spider",
    "squirrel","streetcar","sunflower","sweet_pepper","table","tank","telephone","television","tiger","tractor",
    "train","trout","tulip","turtle","wardrobe","whale","willow_tree","wolf","woman","worm",
]

MEAN = (0.5071, 0.4867, 0.4408)
STD  = (0.2675, 0.2565, 0.2761)

TRANSFORM = T.Compose([
    T.Resize(32),
    T.CenterCrop(32),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


def load_model(name: str, checkpoint: str = None, device=None):
    model = get_student(name, num_classes=100, pretrained=False)
    if checkpoint is None:
        # auto-detect: prefer KD checkpoint, fall back to baseline
        for prefix in ("kd", "baseline"):
            path = os.path.join(CKPT_DIR, f"{prefix}_{name}.pth")
            if os.path.exists(path):
                checkpoint = path
                break
    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device or "cpu")
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded {name} from {checkpoint}")
    else:
        print(f"WARNING: No checkpoint found for {name}; running with random weights.")
    model.to(device).eval()
    return model


@torch.no_grad()
def predict(model, tensor: torch.Tensor, top_k: int = 5):
    logits = model(tensor.unsqueeze(0))
    probs  = torch.softmax(logits, dim=-1).squeeze()
    topk   = probs.topk(top_k)
    return [(CIFAR100_CLASSES[i], float(p)) for i, p in zip(topk.indices, topk.values)]


def run_on_cifar100_sample(model, device, n: int = 8):
    """Run inference on n random CIFAR-100 test images and print results."""
    val_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    ds     = torchvision.datasets.CIFAR100(DATA, train=False, download=True, transform=val_tf)
    correct = 0
    t_start = time.time()
    for i in range(n):
        img, label = ds[i]
        logits = model(img.unsqueeze(0).to(device))
        pred   = logits.argmax(1).item()
        status = "✓" if pred == label else "✗"
        print(f"  [{status}] True: {CIFAR100_CLASSES[label]:20s} | Pred: {CIFAR100_CLASSES[pred]}")
        correct += int(pred == label)
    elapsed = (time.time() - t_start) / n * 1000
    print(f"\n  Accuracy on {n} samples: {correct}/{n} | Avg latency: {elapsed:.1f} ms/img")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      choices=list(STUDENT_REGISTRY), default="mobilenet")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pth checkpoint (auto-detected if omitted)")
    parser.add_argument("--image",      default=None,
                        help="Path to an image file to classify")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.model, args.checkpoint, device)

    if args.image:
        img    = Image.open(args.image).convert("RGB")
        tensor = TRANSFORM(img).to(device)
        preds  = predict(model, tensor)
        print(f"\nTop-5 predictions for {args.image}:")
        for cls, prob in preds:
            print(f"  {cls:25s} {prob*100:.2f}%")
    else:
        n = 4 if args.smoke_test else 20
        print(f"\nRunning inference on {n} CIFAR-100 test images ({args.model}):")
        run_on_cifar100_sample(model, device, n=n)


if __name__ == "__main__":
    main()
