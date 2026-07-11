"""
Compare Methods — Side-by-side comparison of all explainability methods.

This example demonstrates:
- Running all methods on the same model and image
- Creating publication-quality comparison images
- Quantitative evaluation with insertion/deletion scores
- Method speed benchmarks

Run:
    python examples/compare_methods.py
"""

import time

import numpy as np
from PIL import Image
import torch
import torchvision.models as models

from torchxai import (
    EigenCAM,
    GradCAM,
    create_comparison,
    deletion_score,
    explain,
    insertion_score,
    stability_score,
)

# ── Setup ─────────────────────────────────────────────────────────────
print("Loading pretrained ResNet50...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

sample_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
tensor = torch.randn(1, 3, 224, 224)


# ══════════════════════════════════════════════════════════════════════
# Part 1: Visual Comparison
# ══════════════════════════════════════════════════════════════════════
print("\n--- Visual Comparison ---")

all_methods = ["gradcam", "eigencam", "layercam", "gradcam_pp"]
heatmaps = {}

for method in all_methods:
    hm = explain(model, tensor, method=method)
    heatmaps[method] = hm
    print(f"  {method}: mean={hm.mean():.3f}, max={hm.max():.3f}")

# Save comparison
create_comparison(sample_image, heatmaps, save_path="method_comparison.png")
print("  Saved to method_comparison.png")


# ══════════════════════════════════════════════════════════════════════
# Part 2: Quantitative Evaluation
# ══════════════════════════════════════════════════════════════════════
print("\n--- Quantitative Metrics ---")
print(f"  {'Method':<15} {'Insertion ↑':>12} {'Deletion ↓':>12}")
print("  " + "-" * 40)

for method, hm in heatmaps.items():
    ins = insertion_score(model, tensor, hm, steps=20)
    dele = deletion_score(model, tensor, hm, steps=20)
    print(f"  {method:<15} {ins:>12.4f} {dele:>12.4f}")


# ══════════════════════════════════════════════════════════════════════
# Part 3: Stability Comparison
# ══════════════════════════════════════════════════════════════════════
print("\n--- Stability Scores ---")
print("  (Higher = more consistent under input perturbations)")

explainers = {
    "GradCAM": GradCAM(model),
    "EigenCAM": EigenCAM(model),
}

for name, exp in explainers.items():
    stab = stability_score(exp, tensor, num_perturbations=5)
    print(f"  {name}: {stab:.4f}")


# ══════════════════════════════════════════════════════════════════════
# Part 4: Speed Benchmark
# ══════════════════════════════════════════════════════════════════════
print("\n--- Speed Benchmark (5 runs each) ---")

for method in all_methods:
    start = time.time()
    for _ in range(5):
        explain(model, tensor, method=method)
    elapsed = (time.time() - start) / 5
    print(f"  {method:<15} {elapsed * 1000:>8.1f} ms/run")

print("\nDone!")
