# Advanced Usage Guide

This guide covers advanced torchxai usage patterns: layer selection, quantitative evaluation, custom models, GPU acceleration, production deployment, third-party library integrations, performance optimization, and extending the library with custom methods.

---

## Table of Contents

1. [Custom Target Layers](#1-custom-target-layers)
2. [Quantitative Evaluation](#2-quantitative-evaluation)
3. [Working with Custom Models](#3-working-with-custom-models)
4. [GPU Acceleration](#4-gpu-acceleration)
5. [Deployment and Production](#5-deployment-and-production)
6. [Integration with Other Libraries](#6-integration-with-other-libraries)
7. [Performance Optimization](#7-performance-optimization)
8. [Extending torchxai](#8-extending-torchxai)

---

## 1. Custom Target Layers

### Why layer choice matters

The target layer determines the resolution and semantic level of the resulting explanation. Deeper layers (closer to the classifier head) capture high-level semantic concepts but at very coarse spatial resolution. Shallower layers retain more spatial detail but respond to lower-level features like edges and textures.

| Layer depth | Spatial resolution | Semantic level | Typical use |
|---|---|---|---|
| Early (layer1) | High | Low (edges, colors) | Fine-grained part localization |
| Middle (layer2–3) | Medium | Medium | Object parts, textures |
| Late (layer4) | Low | High (semantic objects) | Object-level explanation — default |

### Discovering layer names

Use `model.named_modules()` to enumerate all layers and their names:

```python
import torchvision.models as models

model = models.resnet50(pretrained=True)

# Print all named modules
for name, module in model.named_modules():
    print(f"{name:50s}  {module.__class__.__name__}")
```

Sample output (truncated):

```
                                                    ResNet
layer1                                              Sequential
layer1.0                                            Bottleneck
layer1.0.conv1                                      Conv2d
layer1.0.bn1                                        BatchNorm2d
...
layer4                                              Sequential
layer4.2                                            Bottleneck
layer4.2.conv3                                      Conv2d
layer4.2.bn3                                        BatchNorm2d
layer4.2.relu                                       ReLU
avgpool                                             AdaptiveAvgPool2d
fc                                                  Linear
```

### Specifying a target layer by name (string)

```python
from torchxai import explain
from PIL import Image

model = models.resnet50(pretrained=True).eval()
image = Image.open("cat.jpg")

# Use a string matching the module name from named_modules()
result = explain(model, image, method="gradcam", target_layer="layer4")
result.show()
```

### Specifying a target layer by direct reference

Pass the module object itself — useful when you already have a reference to the layer:

```python
# Get a direct reference to the layer
target_layer = model.layer4[-1].conv3

result = explain(model, image, method="gradcam", target_layer=target_layer)
result.show()
```

### Using earlier layers for finer detail

```python
# Earlier layer for higher spatial resolution
result_coarse = explain(model, image, method="layercam", target_layer="layer4")
result_fine   = explain(model, image, method="layercam", target_layer="layer2")

result_coarse.save("coarse.png")
result_fine.save("fine.png")
```

### Multi-layer explanation

Compare explanations across multiple layers simultaneously:

```python
layers = ["layer1", "layer2", "layer3", "layer4"]
results = {}

for layer_name in layers:
    results[layer_name] = explain(
        model, image,
        method="gradcam",
        target_layer=layer_name
    )
    results[layer_name].save(f"gradcam_{layer_name}.png")
```

---

## 2. Quantitative Evaluation

torchxai provides three built-in metrics for quantitatively evaluating explanation quality. These allow you to compare methods objectively rather than relying on visual inspection alone.

### Insertion Score

The insertion score measures how quickly model confidence recovers as explained regions are progressively revealed (inserted) into a masked image. A **higher insertion score is better** — it means the highlighted regions genuinely contain the most discriminative information.

**Interpretation:** If your insertion score is above 0.7, the explanation is tightly correlated with what the model uses for its decision. Scores below 0.4 suggest the highlighted regions are not strongly discriminative.

```python
from torchxai import explain
from torchxai.metrics import insertion_score
import torchvision.models as models
from PIL import Image
import torch

model = models.resnet50(pretrained=True).eval()
image = Image.open("cat.jpg")

# Get explanation
result = explain(model, image, method="gradcam")

# Compute insertion score (higher is better, range [0, 1])
score = insertion_score(
    model=model,
    image=image,
    saliency_map=result.saliency_map,    # raw numpy heatmap
    target_class=result.predicted_class,  # int class index
    steps=50,                             # number of insertion steps
    baseline="black",                     # starting image — all black
)
print(f"Insertion score: {score:.4f}")
# Example output: Insertion score: 0.7234
```

### Deletion Score

The deletion score measures how quickly model confidence drops as the most salient regions are progressively masked out. A **lower deletion score is better** — it means removing the highlighted regions rapidly destroys the model's confidence.

**Interpretation:** Deletion scores below 0.15 indicate that the explanation correctly identifies the regions the model depends on. If deletion score is high (> 0.4), the model does not depend much on the highlighted regions.

```python
from torchxai.metrics import deletion_score

score = deletion_score(
    model=model,
    image=image,
    saliency_map=result.saliency_map,
    target_class=result.predicted_class,
    steps=50,
    baseline="blur",   # replace removed regions with blurred image
)
print(f"Deletion score: {score:.4f}")
# Example output: Deletion score: 0.1102
# Good — low score means the model loses confidence quickly when
# the salient regions are removed.
```

### Stability Score

The stability score evaluates how consistent the explanation is under small input perturbations. A **higher stability score is better** — a reliable explanation should not change dramatically when the input is slightly noised.

**Interpretation:** Scores above 0.8 indicate a stable explanation that generalizes beyond the exact input pixel values. Scores below 0.5 suggest the explanation is sensitive to noise and may not be trustworthy.

```python
from torchxai.metrics import stability_score

score = stability_score(
    model=model,
    image=image,
    method="gradcam",
    n_perturbations=20,    # number of noisy copies to generate
    noise_std=0.05,        # standard deviation of Gaussian noise
)
print(f"Stability score: {score:.4f}")
# Example output: Stability score: 0.8641
```

### Comparing methods quantitatively

```python
from torchxai.metrics import insertion_score, deletion_score, stability_score
import pandas as pd

methods = ["gradcam", "gradcam++", "layercam", "eigencam"]
rows = []

for method in methods:
    result = explain(model, image, method=method)

    ins = insertion_score(model, image, result.saliency_map, result.predicted_class)
    del = deletion_score(model, image, result.saliency_map, result.predicted_class)
    sta = stability_score(model, image, method=method)

    rows.append({
        "method":    method,
        "insertion": round(ins, 4),  # higher is better
        "deletion":  round(del, 4),  # lower is better
        "stability": round(sta, 4),  # higher is better
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
```

Example output:

```
     method  insertion  deletion  stability
    gradcam     0.7234    0.1102     0.8641
  gradcam++     0.7389    0.0987     0.8512
   layercam     0.6891    0.0874     0.8203
   eigencam     0.6102    0.1543     0.9102
```

**Reading the table:** GradCAM++ achieves the best insertion/deletion trade-off, while EigenCAM has the best stability (expected — no gradient computation makes it noise-insensitive). LayerCAM has the best deletion score, confirming its fine-grained localization is genuinely discriminative.

### Score interpretation guide

| Metric | Excellent | Acceptable | Poor |
|---|---|---|---|
| Insertion | > 0.70 | 0.50–0.70 | < 0.50 |
| Deletion | < 0.15 | 0.15–0.35 | > 0.35 |
| Stability | > 0.80 | 0.60–0.80 | < 0.60 |

---

## 3. Working with Custom Models

### How auto-detection works

When you call `explain(model, image)` without specifying `target_layer`, torchxai attempts to auto-detect the appropriate layer by inspecting the model's module tree. It looks for known patterns:

1. For **CNN models**, it searches for the last `Conv2d` layer before the classifier (a `Linear` or `AdaptiveAvgPool2d` layer).
2. For **ViT models**, it detects `MultiheadAttention` modules and selects the last transformer block.
3. For **timm models**, it uses the `feature_info` attribute if available.

Auto-detection works reliably for standard torchvision and timm architectures. For custom models, it may fail or select an uninformative layer.

### What to do when auto-detection fails

If `explain()` raises a `LayerDetectionError` or produces a blank/uniform heatmap, specify the target layer manually:

```python
import torch
import torch.nn as nn
from torchxai import explain
from PIL import Image

# Example: a custom CNN
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.last_conv = nn.Conv2d(128, 256, 3, padding=1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.last_conv(x)
        return self.classifier(x)

model = MyNet().eval()
image = Image.open("sample.jpg")

# Auto-detection may not find `last_conv` — specify it explicitly
result = explain(model, image, method="gradcam", target_layer=model.last_conv)
result.show()
```

### Debugging layer selection

```python
# Enumerate the model to find candidate layers
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        print(f"Conv2d: {name}  — out_channels={module.out_channels}")
```

Then pick the last Conv2d before the classifier:

```python
result = explain(model, image, method="gradcam", target_layer="last_conv")
```

### Models with non-standard output heads

For models that return dictionaries, tuples, or other non-tensor outputs, wrap the forward method:

```python
class WrappedModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        output = self.base(x)
        # Return only the classification logits
        if isinstance(output, dict):
            return output["logits"]
        if isinstance(output, tuple):
            return output[0]
        return output

wrapped = WrappedModel(my_detection_model).eval()
result = explain(wrapped, image, method="gradcam", target_layer="base.backbone.layer4")
```

---

## 4. GPU Acceleration

### Moving the model to GPU

```python
import torch
from torchxai import explain
import torchvision.models as models
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet50(pretrained=True).to(device).eval()
image = Image.open("cat.jpg")

# torchxai automatically detects the model's device
result = explain(model, image, method="gradcam")
result.show()
```

### Explicit device parameter

```python
result = explain(model, image, method="gradcam", device="cuda:0")
```

### Multi-GPU: selecting a specific GPU

```python
# Use the second GPU (index 1)
model = model.to("cuda:1")
result = explain(model, image, method="gradcam", device="cuda:1")
```

### Memory considerations

Gradient-based methods (GradCAM, GradCAM++, LayerCAM, Transformer Attribution) retain intermediate activations and gradients in memory during the backward pass. For large models or high-resolution images:

```python
# Process in a torch.no_grad() context where possible
# (torchxai handles gradient enabling internally for methods that need it,
# but you can control memory for the surrounding pipeline)

import gc

results = []
for image_path in image_list:
    image = Image.open(image_path)
    result = explain(model, image, method="gradcam")
    results.append(result.saliency_map)  # keep only the numpy array

    # Explicit cleanup after each image
    torch.cuda.empty_cache()
    gc.collect()
```

For batch processing on GPU, prefer EigenCAM (no backward pass, lower peak memory):

```python
# EigenCAM: ~40% less VRAM than GradCAM for the same batch
result = explain(model, image, method="eigencam", device="cuda")
```

---

## 5. Deployment and Production

### Headless server usage (no display)

On servers without a display (typical in Docker, cloud VMs, CI/CD), calling `result.show()` will fail. Use `save_path` to write the overlay directly to disk:

```python
from torchxai import explain
import torchvision.models as models
from PIL import Image

model = models.resnet50(pretrained=True).eval()
image = Image.open("/data/input/cat.jpg")

result = explain(
    model, image,
    method="gradcam",
    save_path="/data/output/explanation.png",  # write to disk without displaying
)
```

### Returning a numpy array

For web API integration, skip file I/O and return the raw saliency map or overlay as a numpy array:

```python
import numpy as np

result = explain(model, image, method="gradcam")

# Raw saliency map, shape (H, W), dtype float32, range [0, 1]
saliency: np.ndarray = result.saliency_map

# Overlay (saliency composited on original image), shape (H, W, 3), dtype uint8
overlay: np.ndarray = result.overlay
```

### FastAPI integration

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import torchvision.models as models
from torchxai import explain
from PIL import Image
import io
import torch

app = FastAPI()

# Load model once at startup
model = models.resnet50(pretrained=True).eval()
if torch.cuda.is_available():
    model = model.cuda()

@app.post("/explain", response_class=Response)
async def explain_image(
    file: UploadFile = File(...),
    method: str = "gradcam",
):
    # Read uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Generate explanation
    result = explain(model, image, method=method)

    # Encode overlay as PNG and return
    overlay_pil = Image.fromarray(result.overlay)
    buf = io.BytesIO()
    overlay_pil.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.read(), media_type="image/png")
```

### Docker example

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set matplotlib/PIL to non-interactive backend
ENV MPLBACKEND=Agg

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`requirements.txt`:

```
torch>=2.0.0
torchvision>=0.15.0
torchxai-explain
fastapi
uvicorn[standard]
Pillow
numpy
```

---

## 6. Integration with Other Libraries

### timm models

[timm](https://github.com/huggingface/pytorch-image-models) models work out-of-the-box. torchxai detects the `feature_info` attribute to auto-select the best target layer:

```python
import timm
from torchxai import explain
from PIL import Image

# Standard CNN from timm
model = timm.create_model("efficientnet_b4", pretrained=True).eval()
image = Image.open("cat.jpg")

result = explain(model, image, method="gradcam")
result.show()

# timm ViT — auto-selects last transformer block
vit = timm.create_model("vit_base_patch16_224", pretrained=True).eval()
result = explain(vit, image, method="transformer_attribution", target_class=281)
result.show()
```

To manually target a specific timm layer:

```python
# Get layer names from timm's feature info
for i, info in enumerate(model.feature_info):
    print(f"Stage {i}: {info['module']}  — channels={info['num_chs']}")

# Use a specific stage
result = explain(model, image, method="gradcam", target_layer="blocks.5")
```

### HuggingFace Transformers (ViTForImageClassification)

```python
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torchxai import explain
from PIL import Image
import torch

model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
hf_model = ViTForImageClassification.from_pretrained(model_name).eval()

image = Image.open("cat.jpg")

# HuggingFace models return a ModelOutput object, not a plain tensor.
# Wrap to return logits directly.
class HFViTWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).logits

wrapped = HFViTWrapper(hf_model).eval()

# Specify the last encoder layer explicitly (HF naming convention)
result = explain(
    wrapped, image,
    method="attention_rollout",
    target_layer="model.vit.encoder.layer.11.attention.attention",
)
result.show()
```

### Ultralytics YOLO

YOLO uses a backbone + detection head. Explanations are most useful on the backbone:

```python
from ultralytics import YOLO
from torchxai import explain
from PIL import Image
import torch

yolo = YOLO("yolov8n.pt")
backbone = yolo.model.model  # access the underlying nn.Module

# Find the last conv layer in the backbone (before detection head)
# In YOLOv8, the backbone ends around layer 9
class YOLOBackboneWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Run through backbone layers only and return a classification-style output
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i == 9:  # last backbone stage
                break
        # Global average pool to get class logits
        return x.mean(dim=[2, 3])

wrapper = YOLOBackboneWrapper(backbone).eval()
image = Image.open("dog.jpg")

result = explain(wrapper, image, method="gradcam", target_layer="model.9")
result.show()
```

### CLIP

CLIP's visual encoder is a ViT (ViT-B/32, ViT-L/14). Use attention-based methods:

```python
import clip
import torch
from torchxai import explain
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Extract the visual encoder
visual_encoder = clip_model.visual.eval()

# CLIP's visual encoder outputs a CLS embedding (not class logits).
# For explanation, we project onto a text embedding direction to get a scalar.
text_tokens = clip.tokenize(["a photo of a cat"]).to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

class CLIPWrapper(torch.nn.Module):
    def __init__(self, visual, text_feat):
        super().__init__()
        self.visual = visual
        self.text_feat = text_feat

    def forward(self, x):
        image_features = self.visual(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return (image_features @ self.text_feat.T)

wrapper = CLIPWrapper(visual_encoder, text_features).eval()

image = Image.open("cat.jpg")
result = explain(
    wrapper, image,
    method="attention_rollout",
    target_layer="visual.transformer.resblocks.11.attn",
)
result.show()
```

---

## 7. Performance Optimization

### Choose EigenCAM for maximum speed

EigenCAM requires only a forward pass and SVD — no backward pass. For throughput-sensitive pipelines, it is typically 3–5× faster than gradient-based methods:

```python
import time
from torchxai import explain

# Benchmark
image = Image.open("cat.jpg")

start = time.perf_counter()
for _ in range(100):
    explain(model, image, method="gradcam")
gradcam_time = (time.perf_counter() - start) / 100

start = time.perf_counter()
for _ in range(100):
    explain(model, image, method="eigencam")
eigencam_time = (time.perf_counter() - start) / 100

print(f"GradCAM:  {gradcam_time*1000:.1f} ms/image")
print(f"EigenCAM: {eigencam_time*1000:.1f} ms/image")
# Example: GradCAM 42.3 ms, EigenCAM 11.8 ms
```

### Reduce input image size

The computational cost of most methods scales with the number of spatial positions in the feature map. Reducing the input size is the single most impactful optimization:

```python
from torchxai import explain
from PIL import Image

image = Image.open("cat.jpg")

# Resize to 224×224 before processing (standard ImageNet size — no quality loss for most models)
image_small = image.resize((224, 224), Image.LANCZOS)
result = explain(model, image_small, method="gradcam")
```

### torch.no_grad() for the surrounding pipeline

torchxai internally enables gradients only for the layers it needs. Wrapping the outer pipeline in `torch.no_grad()` prevents accidental gradient accumulation from other operations:

```python
import torch

with torch.no_grad():
    # Pre-processing, data loading, etc. run here without gradient tracking
    preprocessed = preprocess_pipeline(raw_images)

# torchxai temporarily enables gradients internally as needed
result = explain(model, preprocessed_image, method="gradcam")

with torch.no_grad():
    # Post-processing continues without gradient tracking
    overlay = postprocess(result.overlay)
```

### requires_grad flag behavior

torchxai respects the `requires_grad` state of model parameters. For inference-only pipelines, you can freeze the model to reduce memory and skip gradient bookkeeping for parameters:

```python
# Freeze all parameters — torchxai will still compute input gradients
# via hooks, but won't accumulate parameter gradients
for param in model.parameters():
    param.requires_grad_(False)

result = explain(model, image, method="gradcam")
# Works correctly — GradCAM only needs gradients w.r.t. feature maps,
# not w.r.t. model parameters.
```

> **Note:** Do NOT set `requires_grad=False` on the input tensor itself — torchxai needs to differentiate through the feature maps. The library handles this automatically, but custom wrappers that freeze input tensors will break gradient-based methods.

### Batch processing

Process multiple images in a single call when possible to leverage GPU parallelism:

```python
from PIL import Image
import os

image_dir = "/data/images"
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

# For EigenCAM (batch-friendly, no backward pass)
results = []
for path in image_paths:
    image = Image.open(path)
    result = explain(model, image, method="eigencam", device="cuda")
    results.append(result.saliency_map)
```

---

## 8. Extending torchxai

You can add a custom explainability method by subclassing `BaseExplainer`. The abstract class defines the interface; you implement `_compute_saliency()`.

### BaseExplainer interface

```python
from torchxai.explainers import BaseExplainer
import torch
import numpy as np
from typing import Optional

class BaseExplainer:
    """
    Abstract base class for all torchxai explainability methods.

    Subclasses must implement:
        _compute_saliency(model, input_tensor, target_layer, target_class)
            -> np.ndarray of shape (H, W), dtype float32, range [0, 1]
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.target_layer = target_layer
        self._hooks = []

    def explain(
        self,
        input_tensor: torch.Tensor,   # (1, C, H, W), normalized
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Entry point called by torchxai.explain().
        Resolves target_class if not provided, then delegates to _compute_saliency().
        """
        raise NotImplementedError

    def _compute_saliency(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        target_class: int,
    ) -> np.ndarray:
        raise NotImplementedError

    def _register_hook(self, layer: torch.nn.Module, hook_fn):
        handle = layer.register_forward_hook(hook_fn)
        self._hooks.append(handle)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
```

### Example: implementing ScoreCAM

ScoreCAM is a gradient-free method that weights each activation channel by the increase in class score it produces when used as a mask. Here is a minimal implementation:

```python
import torch
import torch.nn.functional as F
import numpy as np
from torchxai.explainers import BaseExplainer

class ScoreCAM(BaseExplainer):
    """
    ScoreCAM: Score-Based Visual Explanations for Convolutional Neural Networks.
    Wang et al., CVPR 2020 Workshop.

    Gradient-free: weights feature maps by the class score they produce when
    used as input masks.
    """

    def _compute_saliency(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,  # (1, 3, H, W)
        target_layer: torch.nn.Module,
        target_class: int,
    ) -> np.ndarray:
        H, W = input_tensor.shape[2], input_tensor.shape[3]

        # Step 1: Forward pass, capture activations at target layer
        activations = {}

        def save_activation(module, input, output):
            activations["feat"] = output.detach()

        hook = target_layer.register_forward_hook(save_activation)
        with torch.no_grad():
            logits = model(input_tensor)
        hook.remove()

        feat = activations["feat"]  # (1, C, h, w)
        C = feat.shape[1]

        # Step 2: For each channel, upsample the activation map and use it
        # as a soft mask on the input image. Measure the resulting class score.
        channel_scores = torch.zeros(C, device=input_tensor.device)

        with torch.no_grad():
            for c in range(C):
                # Upsample single channel to input resolution
                mask = feat[:, c:c+1, :, :]  # (1, 1, h, w)
                mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)

                # Normalize mask to [0, 1]
                m_min, m_max = mask.min(), mask.max()
                if m_max - m_min > 1e-8:
                    mask = (mask - m_min) / (m_max - m_min)
                else:
                    mask = torch.zeros_like(mask)

                # Apply mask to input image
                masked_input = input_tensor * mask

                # Get softmax probability for target class
                score = torch.softmax(model(masked_input), dim=1)[0, target_class]
                channel_scores[c] = score

        # Step 3: Normalize scores to use as channel weights
        channel_scores = F.softmax(channel_scores, dim=0)  # (C,)

        # Step 4: Weighted sum of activation maps
        feat = feat.squeeze(0)  # (C, h, w)
        weighted = (channel_scores[:, None, None] * feat).sum(dim=0)  # (h, w)

        # Step 5: ReLU and normalize
        weighted = F.relu(weighted)
        w_min, w_max = weighted.min(), weighted.max()
        if w_max - w_min > 1e-8:
            weighted = (weighted - w_min) / (w_max - w_min)
        else:
            weighted = torch.zeros_like(weighted)

        # Step 6: Upsample to input size
        cam = F.interpolate(
            weighted.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        return cam.astype(np.float32)
```

### Registering the custom method

```python
from torchxai import register_method

register_method("scorecam", ScoreCAM)
```

After registration, use it through the standard `explain()` API:

```python
from torchxai import explain
import torchvision.models as models
from PIL import Image

model = models.resnet50(pretrained=True).eval()
image = Image.open("cat.jpg")

result = explain(model, image, method="scorecam")
result.show()
```

### Testing your custom method

```python
import torch
import numpy as np
from torchxai import explain
import torchvision.models as models
from PIL import Image

def test_custom_method():
    model = models.resnet18(pretrained=False).eval()
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    result = explain(model, dummy_image, method="scorecam")

    # Basic sanity checks
    assert result.saliency_map.ndim == 2, "Saliency map must be 2D"
    assert result.saliency_map.shape == (224, 224), "Saliency map must match input size"
    assert result.saliency_map.dtype == np.float32, "Saliency map must be float32"
    assert 0.0 <= result.saliency_map.min(), "Saliency values must be non-negative"
    assert result.saliency_map.max() <= 1.0, "Saliency values must be <= 1"

    print("All tests passed.")

test_custom_method()
```

### Checklist for custom explainer implementations

- `_compute_saliency()` must return a `numpy.ndarray` of shape `(H, W)` where H, W match the input image dimensions.
- Values must be in `[0, 1]` (float32).
- Always call `self._remove_hooks()` in a `finally` block if you register forward/backward hooks, to prevent hook accumulation across calls.
- Handle the case where `m_max == m_min` (uniform activations) to avoid division by zero.
- Test with both CPU and GPU (`device="cuda"`) if your method uses device-specific operations.
- Document whether your method is class-specific and whether it requires a backward pass, so users can make informed choices.
