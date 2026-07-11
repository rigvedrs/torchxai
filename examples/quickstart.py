"""
torchxai Quickstart — Your first explanation in 60 seconds.

This script shows the simplest possible usage of torchxai:
1. Load a pretrained model
2. Explain a prediction in one line
3. Visualize the result

Run:
    python examples/quickstart.py
"""

import numpy as np
from PIL import Image
import torchvision.models as models

from torchxai import explain, show_explanation

# ── Step 1: Load any pretrained model ─────────────────────────────────
print("Loading pretrained ResNet50...")
model = models.resnet50(pretrained=True)
model.eval()  # Always set to eval mode!

# ── Step 2: Create a sample image (replace with your own) ────────────
# You can pass a file path, PIL image, numpy array, or torch tensor
sample_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
# Or use a real image:
# heatmap = explain(model, "path/to/your/image.jpg")

# ── Step 3: Explain in one line ───────────────────────────────────────
print("Generating explanation...")
heatmap = explain(model, sample_image)
print(f"Heatmap shape: {heatmap.shape}")  # (224, 224)
print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")  # [0, 1]

# ── Step 4: Visualize ─────────────────────────────────────────────────
show_explanation(
    sample_image,
    heatmap,
    title="What ResNet50 sees",
    save_path="quickstart_output.png",  # Saves to file (works on servers too)
)
print("Saved to quickstart_output.png")
