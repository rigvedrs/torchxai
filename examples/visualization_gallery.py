"""
Visualization Gallery — Every visualization option in torchxai.

This example demonstrates:
- overlay_heatmap() with all options
- show_explanation() for 3-panel views
- create_comparison() for multi-method comparisons
- save_heatmap() for raw heatmap export
- Different colormaps and alpha values

Run:
    python examples/visualization_gallery.py
"""

import os

import numpy as np
from PIL import Image
import torch
import torchvision.models as models

from torchxai import (
    create_comparison,
    explain,
    overlay_heatmap,
    save_heatmap,
    show_explanation,
)

# ── Setup ─────────────────────────────────────────────────────────────
os.makedirs("gallery_output", exist_ok=True)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

# Create a sample image (replace with your own)
sample_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
image_np = np.array(sample_image).astype(np.float32) / 255.0
tensor = torch.randn(1, 3, 224, 224)

heatmap = explain(model, tensor)
print(f"Heatmap: shape={heatmap.shape}, range=[{heatmap.min():.3f}, {heatmap.max():.3f}]")


# ══════════════════════════════════════════════════════════════════════
# 1. overlay_heatmap — Basic Overlay
# ══════════════════════════════════════════════════════════════════════
print("\n--- overlay_heatmap ---")

# Default (jet colormap, alpha=0.5)
overlay = overlay_heatmap(image_np, heatmap)
Image.fromarray((overlay * 255).astype(np.uint8)).save("gallery_output/overlay_default.png")
print("  Saved overlay_default.png")

# Different colormaps
for cmap in ["jet", "viridis", "hot", "inferno", "coolwarm"]:
    overlay = overlay_heatmap(image_np, heatmap, colormap=cmap)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(f"gallery_output/overlay_{cmap}.png")
    print(f"  Saved overlay_{cmap}.png")

# Different alpha values
for alpha in [0.2, 0.4, 0.6, 0.8]:
    overlay = overlay_heatmap(image_np, heatmap, alpha=alpha)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(
        f"gallery_output/overlay_alpha_{alpha}.png"
    )
    print(f"  Saved overlay_alpha_{alpha}.png")

# Custom output size
overlay = overlay_heatmap(image_np, heatmap, output_size=(448, 448))
print(f"  Upscaled overlay: shape={overlay.shape}")

# Works with PIL images directly
overlay = overlay_heatmap(sample_image, heatmap)
print(f"  PIL input: shape={overlay.shape}")


# ══════════════════════════════════════════════════════════════════════
# 2. show_explanation — 3-Panel View
# ══════════════════════════════════════════════════════════════════════
print("\n--- show_explanation ---")

# Basic usage
show_explanation(
    sample_image,
    heatmap,
    title="ResNet50 GradCAM Explanation",
    save_path="gallery_output/explanation_basic.png",
)
print("  Saved explanation_basic.png")

# Custom colormap and alpha
show_explanation(
    sample_image,
    heatmap,
    title="Viridis colormap",
    colormap="viridis",
    alpha=0.7,
    save_path="gallery_output/explanation_viridis.png",
)
print("  Saved explanation_viridis.png")

# Custom figure size
show_explanation(
    sample_image,
    heatmap,
    figsize=(18, 6),
    save_path="gallery_output/explanation_large.png",
)
print("  Saved explanation_large.png")


# ══════════════════════════════════════════════════════════════════════
# 3. create_comparison — Multi-Method Side-by-Side
# ══════════════════════════════════════════════════════════════════════
print("\n--- create_comparison ---")

heatmaps = {
    "GradCAM": explain(model, tensor, method="gradcam"),
    "EigenCAM": explain(model, tensor, method="eigencam"),
    "LayerCAM": explain(model, tensor, method="layercam"),
    "GradCAM++": explain(model, tensor, method="gradcam_pp"),
}

create_comparison(
    sample_image,
    heatmaps,
    save_path="gallery_output/comparison_all.png",
)
print("  Saved comparison_all.png")

# Just 2 methods
create_comparison(
    sample_image,
    {"GradCAM": heatmaps["GradCAM"], "EigenCAM": heatmaps["EigenCAM"]},
    save_path="gallery_output/comparison_2methods.png",
)
print("  Saved comparison_2methods.png")


# ══════════════════════════════════════════════════════════════════════
# 4. save_heatmap — Raw Heatmap Export
# ══════════════════════════════════════════════════════════════════════
print("\n--- save_heatmap ---")

# Default (jet colormap)
save_heatmap(heatmap, "gallery_output/heatmap_jet.png")
print("  Saved heatmap_jet.png")

# Different colormaps
for cmap in ["viridis", "hot", "inferno"]:
    save_heatmap(heatmap, f"gallery_output/heatmap_{cmap}.png", colormap=cmap)
    print(f"  Saved heatmap_{cmap}.png")

# Save as JPEG
save_heatmap(heatmap, "gallery_output/heatmap.jpg")
print("  Saved heatmap.jpg")


print("\nAll files saved to gallery_output/")
print(f"Total files: {len(os.listdir('gallery_output'))}")
print("\nDone!")
