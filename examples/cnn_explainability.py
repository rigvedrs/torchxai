"""
CNN Explainability — Explaining ResNet, VGG, EfficientNet, and DenseNet.

This example demonstrates:
- Using explain() with different CNN architectures
- Comparing all available methods on a CNN
- Class-specific explanations
- Custom target layer selection

Run:
    python examples/cnn_explainability.py
"""

import numpy as np
from PIL import Image
import torch
import torchvision.models as models

from torchxai import (
    EigenCAM,
    GradCAM,
    GradCAMPlusPlus,
    LayerCAM,
    create_comparison,
    explain,
)

# ── Setup ─────────────────────────────────────────────────────────────
print("Loading models...")

# Create a sample image (replace with your own)
sample_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
tensor = torch.randn(1, 3, 224, 224)


# ══════════════════════════════════════════════════════════════════════
# Example 1: Basic Usage with Different Architectures
# ══════════════════════════════════════════════════════════════════════
print("\n--- Example 1: Different Architectures ---")

architectures = {
    "ResNet50": models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
    "VGG16": models.vgg16(weights=models.VGG16_Weights.DEFAULT),
    # Uncomment if you have these installed:
    # "EfficientNet": models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
    # "DenseNet": models.densenet121(weights=models.DenseNet121_Weights.DEFAULT),
}

for name, model in architectures.items():
    model.eval()
    heatmap = explain(model, tensor)
    print(f"  {name}: heatmap shape = {heatmap.shape}, max = {heatmap.max():.3f}")


# ══════════════════════════════════════════════════════════════════════
# Example 2: Compare All Methods on ResNet50
# ══════════════════════════════════════════════════════════════════════
print("\n--- Example 2: Compare All Methods ---")

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

methods = {
    "GradCAM": explain(model, tensor, method="gradcam"),
    "EigenCAM": explain(model, tensor, method="eigencam"),
    "LayerCAM": explain(model, tensor, method="layercam"),
    "GradCAM++": explain(model, tensor, method="gradcam_pp"),
}

for name, hm in methods.items():
    print(f"  {name}: shape={hm.shape}, mean={hm.mean():.3f}")

# Save comparison image
create_comparison(sample_image, methods, save_path="cnn_comparison.png")
print("  Saved comparison to cnn_comparison.png")


# ══════════════════════════════════════════════════════════════════════
# Example 3: Class-Specific Explanations
# ══════════════════════════════════════════════════════════════════════
print("\n--- Example 3: Class-Specific Explanations ---")

# ImageNet class indices (see https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/)
target_classes = {
    "Predicted (auto)": None,
    "Labrador (208)": 208,
    "Tabby cat (281)": 281,
    "Tennis ball (852)": 852,
}

for label, cls_id in target_classes.items():
    hm = explain(model, tensor, target_class=cls_id)
    print(f"  {label}: shape={hm.shape}, mean={hm.mean():.3f}")


# ══════════════════════════════════════════════════════════════════════
# Example 4: Custom Target Layer
# ══════════════════════════════════════════════════════════════════════
print("\n--- Example 4: Custom Target Layers ---")

# Option A: Use a string name
cam_layer3 = GradCAM(model, target_layer="layer3")
hm_layer3 = cam_layer3(tensor)
print(f"  layer3: shape={hm_layer3.shape}, resolution before upscale = higher")

# Option B: Use a module reference
cam_layer4 = GradCAM(model, target_layer=model.layer4[-1])
hm_layer4 = cam_layer4(tensor)
print(f"  layer4[-1]: shape={hm_layer4.shape}, resolution before upscale = lower")

# List all available layers:
print("\n  Available layers for ResNet50:")
for name, module in model.named_modules():
    if name and "." not in name:  # Top-level children only
        print(f"    {name}: {type(module).__name__}")


# ══════════════════════════════════════════════════════════════════════
# Example 5: Using Explainer Classes Directly
# ══════════════════════════════════════════════════════════════════════
print("\n--- Example 5: Direct Explainer Usage ---")

# Each explainer can be used as a callable
cam = GradCAM(model)
eigen = EigenCAM(model)
layer = LayerCAM(model)
grad_pp = GradCAMPlusPlus(model)

print(f"  GradCAM:   {repr(cam)}")
print(f"  EigenCAM:  {repr(eigen)}")
print(f"  LayerCAM:  {repr(layer)}")
print(f"  GradCAM++: {repr(grad_pp)}")

# EigenCAM is fastest (no backward pass)
import time

start = time.time()
for _ in range(10):
    eigen(tensor)
eigen_time = (time.time() - start) / 10

start = time.time()
for _ in range(10):
    cam(tensor)
cam_time = (time.time() - start) / 10

print("\n  Speed comparison (10 runs):")
print(f"    EigenCAM: {eigen_time * 1000:.1f} ms/run (no gradients)")
print(f"    GradCAM:  {cam_time * 1000:.1f} ms/run (with gradients)")

print("\nDone!")
