"""
Metrics Evaluation — Quantitative measurement of explanation quality.

This example demonstrates:
- Insertion score (higher = better)
- Deletion score (lower = better)
- Stability score (higher = more consistent)
- How to compare methods quantitatively

Run:
    python examples/metrics_evaluation.py
"""

import numpy as np
import torch
import torchvision.models as models

from torchxai import (
    EigenCAM,
    GradCAM,
    GradCAMPlusPlus,
    LayerCAM,
    deletion_score,
    explain,
    insertion_score,
    stability_score,
)

# ── Setup ─────────────────────────────────────────────────────────────
print("Loading pretrained ResNet50...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

tensor = torch.randn(1, 3, 224, 224)


# ══════════════════════════════════════════════════════════════════════
# What These Metrics Measure
# ══════════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════════╗
║                   What These Metrics Measure                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  INSERTION SCORE (higher = better)                               ║
║  Progressively reveals the most important pixels on a blank      ║
║  canvas. A good explanation causes confidence to rise quickly.   ║
║                                                                  ║
║  DELETION SCORE (lower = better)                                 ║
║  Progressively removes the most important pixels. A good         ║
║  explanation causes confidence to drop quickly.                  ║
║                                                                  ║
║  STABILITY SCORE (higher = more stable)                          ║
║  Adds small noise to the input and measures if the explanation   ║
║  stays similar. Stable methods are more trustworthy.             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ══════════════════════════════════════════════════════════════════════
# Evaluate All Methods
# ══════════════════════════════════════════════════════════════════════
print("Evaluating all methods (this takes a few seconds)...\n")

methods = {
    "GradCAM": ("gradcam", GradCAM(model)),
    "EigenCAM": ("eigencam", EigenCAM(model)),
    "LayerCAM": ("layercam", LayerCAM(model)),
    "GradCAM++": ("gradcam_pp", GradCAMPlusPlus(model)),
}

# Header
print(f"  {'Method':<12} {'Insertion ↑':>12} {'Deletion ↓':>12} {'Stability ↑':>12}")
print("  " + "─" * 50)

for name, (method_key, explainer) in methods.items():
    # Generate heatmap
    hm = explain(model, tensor, method=method_key)

    # Compute metrics
    ins = insertion_score(model, tensor, hm, steps=20)
    dele = deletion_score(model, tensor, hm, steps=20)
    stab = stability_score(explainer, tensor, num_perturbations=5)

    print(f"  {name:<12} {ins:>12.4f} {dele:>12.4f} {stab:>12.4f}")

print()


# ══════════════════════════════════════════════════════════════════════
# Random Baseline (for comparison)
# ══════════════════════════════════════════════════════════════════════
print("--- Random Baseline ---")
print("  (A random heatmap should score poorly on all metrics)")

random_hm = np.random.rand(224, 224).astype(np.float32)
ins_rand = insertion_score(model, tensor, random_hm, steps=20)
dele_rand = deletion_score(model, tensor, random_hm, steps=20)
print(f"  Random:      insertion={ins_rand:.4f}, deletion={dele_rand:.4f}")

print("\nDone!")
