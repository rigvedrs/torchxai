"""
Vision Transformer Explainability — Explaining ViT, DeiT, Swin, and more.

This example demonstrates:
- torchxai's automatic ViT detection and handling
- Attention Rollout vs Transformer Attribution
- How all methods (including CNN methods) adapt to ViTs
- Using timm models

Run:
    pip install timm
    python examples/vit_explainability.py
"""

import numpy as np
from PIL import Image
import torch

from torchxai import (
    AttentionRollout,
    TransformerAttribution,
    create_comparison,
    explain,
)

# ── Setup ─────────────────────────────────────────────────────────────
sample_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
tensor = torch.randn(1, 3, 224, 224)

try:
    import timm

    HAS_TIMM = True
except ImportError:
    print("Install timm for ViT models: pip install timm")
    HAS_TIMM = False

if not HAS_TIMM:
    print("This example requires timm. Install with: pip install timm")
    exit()


# ══════════════════════════════════════════════════════════════════════
# Example 1: Basic ViT Explanation
# ══════════════════════════════════════════════════════════════════════
print("\n--- Example 1: Basic ViT Explanation ---")

# Load a Vision Transformer
vit = timm.create_model("vit_tiny_patch16_224", pretrained=False)
vit.eval()

# explain() auto-detects it's a ViT and handles patch → spatial mapping
heatmap = explain(vit, tensor)
print(f"  ViT heatmap: shape={heatmap.shape}, range=[{heatmap.min():.3f}, {heatmap.max():.3f}]")


# ══════════════════════════════════════════════════════════════════════
# Example 2: All Methods on ViT
# ══════════════════════════════════════════════════════════════════════
print("\n--- Example 2: All Methods on ViT ---")

# Every method works on ViTs — torchxai handles the token→spatial conversion
methods_to_try = ["gradcam", "eigencam", "layercam", "attention_rollout"]

results = {}
for method in methods_to_try:
    hm = explain(vit, tensor, method=method)
    results[method] = hm
    print(f"  {method}: shape={hm.shape}, mean={hm.mean():.3f}")

# Compare all methods
create_comparison(sample_image, results, save_path="vit_comparison.png")
print("  Saved comparison to vit_comparison.png")


# ══════════════════════════════════════════════════════════════════════
# Example 3: Attention Rollout (ViT-Specific)
# ══════════════════════════════════════════════════════════════════════
print("\n--- Example 3: Attention Rollout ---")

# Attention Rollout traces information flow through ALL attention layers
# It's class-agnostic: shows where the model attends, not why it chose a class
rollout = AttentionRollout(vit)
print(f"  {repr(rollout)}")
print(f"  requires_grad: {rollout.requires_grad}")  # False — no backward pass needed

hm_rollout = rollout(tensor)
print(f"  Rollout heatmap: shape={hm_rollout.shape}")


# ══════════════════════════════════════════════════════════════════════
# Example 4: Transformer Attribution (Class-Specific ViT)
# ══════════════════════════════════════════════════════════════════════
print("\n--- Example 4: Transformer Attribution ---")

# Transformer Attribution uses gradients to make attention class-specific
# "Which patches matter for THIS class?"
attr = TransformerAttribution(vit)
print(f"  {repr(attr)}")

hm_attr = attr(tensor, target_class=0)
print(f"  Attribution for class 0: shape={hm_attr.shape}")


# ══════════════════════════════════════════════════════════════════════
# Example 5: Different timm Models
# ══════════════════════════════════════════════════════════════════════
print("\n--- Example 5: Different timm Models ---")

timm_models = [
    "vit_tiny_patch16_224",  # Standard ViT
    "deit_tiny_patch16_224",  # DeiT (distilled ViT)
    "efficientnet_b0",  # CNN (auto-detected)
]

for model_name in timm_models:
    try:
        m = timm.create_model(model_name, pretrained=False)
        m.eval()
        hm = explain(m, tensor)
        print(f"  {model_name}: shape={hm.shape}, detected and explained successfully")
    except Exception as e:
        print(f"  {model_name}: {e}")

print("\nDone!")
