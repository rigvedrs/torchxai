<p align="center">
  <img src="https://raw.githubusercontent.com/rigvedrs/torchxai/main/docs/assets/hero_explanation.png" alt="torchxai — One line to explain any vision model" width="100%">
</p>

<h1 align="center">torchxai</h1>

<p align="center">
  <strong>Explain any PyTorch vision model in one line of code.</strong><br>
  10 methods. 24 models verified end-to-end. CNNs, ViTs, YOLO26, RF-DETR, DINOv2, and more.
</p>

<p align="center">
  <a href="https://pypi.org/project/torchxai-explain/"><img src="https://img.shields.io/pypi/v/torchxai-explain?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/torchxai-explain/"><img src="https://img.shields.io/pypi/pyversions/torchxai-explain" alt="Python"></a>
  <a href="https://github.com/rigvedrs/torchxai/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://github.com/rigvedrs/torchxai/stargazers"><img src="https://img.shields.io/github/stars/rigvedrs/torchxai?style=social" alt="Stars"></a>
</p>

<p align="center">
  <a href="https://rigvedrs.github.io/torchxai/">Documentation</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-methods">Methods</a> •
  <a href="#-examples">Examples</a> •
  <a href="https://rigvedrs.github.io/torchxai/api-reference/">API Reference</a> •
  <a href="https://rigvedrs.github.io/torchxai/troubleshooting/">Troubleshooting</a>
</p>

---

## Why torchxai?

Most explainability libraries make you write 20+ lines of boilerplate, only work with specific model architectures, and crash on Vision Transformers. **torchxai** does it in one line:

```python
from torchxai import explain
heatmap = explain(model, "photo.jpg")
```

That's it. It works with **any** PyTorch vision model — ResNet, ViT, EfficientNet, CLIP, YOLO, Swin, DeiT, DINO, and more. No config files. No architecture-specific code. No boilerplate.

<p align="center">
  <img src="https://raw.githubusercontent.com/rigvedrs/torchxai/main/docs/assets/quickstart_workflow.png" alt="Quick-start workflow" width="100%">
</p>

### What makes it different

| Feature | torchxai | pytorch-grad-cam | captum | tf-explain |
|---------|----------|-------------------|--------|------------|
| One-line API | ✅ `explain(model, img)` | ❌ 15+ lines | ❌ 20+ lines | ❌ TF only |
| Auto-detects architecture | ✅ CNN + ViT + YOLO | ❌ Manual config | ❌ Manual config | ❌ |
| Vision Transformer support | ✅ Native | ⚠️ Partial | ⚠️ Partial | ❌ |
| String path input | ✅ `explain(m, "dog.jpg")` | ❌ | ❌ | ❌ |
| Headless server support | ✅ Auto Agg backend | ❌ Crashes | ❌ | ❌ |
| Built-in metrics | ✅ Insertion/Deletion/Stability | ❌ | ⚠️ Separate | ❌ |
| Graceful fallbacks | ✅ Always produces output | ❌ Crashes | ❌ Crashes | ❌ |

---

## ⚡ Quickstart

### 3 lines to your first explanation

```python
from torchxai import explain
import torchvision.models as models

model = models.resnet50(pretrained=True)
heatmap = explain(model, "dog.jpg")  # Returns (224, 224) numpy array
```

### Visualize it

```python
from torchxai import explain, show_explanation
from PIL import Image

model = models.resnet50(pretrained=True)
heatmap = explain(model, "dog.jpg")
show_explanation(Image.open("dog.jpg"), heatmap, title="What the model sees")
```

<p align="center">
  <img src="https://raw.githubusercontent.com/rigvedrs/torchxai/main/docs/assets/hero_explanation.png" alt="Explanation output" width="85%">
</p>

### Compare methods side by side

```python
from torchxai import explain, create_comparison
from PIL import Image

heatmaps = {
    "GradCAM": explain(model, img, method="gradcam"),
    "EigenCAM": explain(model, img, method="eigencam"),
    "LayerCAM": explain(model, img, method="layercam"),
    "GradCAM++": explain(model, img, method="gradcam_pp"),
}
create_comparison(Image.open("dog.jpg"), heatmaps, save_path="comparison.png")
```

<p align="center">
  <img src="https://raw.githubusercontent.com/rigvedrs/torchxai/main/docs/assets/method_comparison.png" alt="Method comparison" width="100%">
</p>

---

## 📦 Installation

```bash
pip install torchxai-explain
```

**Requirements:** Python 3.8+ and PyTorch 1.9+. That's all.

For development:
```bash
git clone https://github.com/rigvedrs/torchxai.git
cd torchxai
pip install -e ".[dev]"
```

---

## 🔬 Methods

torchxai includes 10 explainability methods. Each produces a saliency heatmap showing which image regions matter most for the model's prediction.

| Method | Type | Needs Gradients | Best For |
|--------|------|-----------------|----------|
| **GradCAM** | Gradient-weighted | ✅ | General-purpose CNN explanation |
| **GradCAM++** | Gradient-weighted | ✅ | Multiple objects of same class |
| **LayerCAM** | Gradient-weighted | ✅ | Fine-grained spatial detail |
| **EigenCAM** | Activation-based | ❌ | Fast, gradient-free explanation |
| **ScoreCAM** | Perturbation-based | ❌ | Most faithful, no gradients |
| **SmoothGrad** | Input gradients | ✅ | Pixel-level attribution |
| **Integrated Gradients** | Axiomatic | ✅ | Theoretically grounded |
| **RISE** | Black-box sampling | ❌ | Model-agnostic, any architecture |
| **Attention Rollout** | Attention-based | ❌ | Vision Transformers (ViT) |
| **Transformer Attribution** | Gradient + Attention | ✅ | Class-specific ViT explanation |

<p align="center">
  <img src="https://raw.githubusercontent.com/rigvedrs/torchxai/main/docs/assets/method_comparison.png" alt="All methods compared" width="100%">
</p>

**Don't know which to pick?** Just use `explain(model, image)` — it auto-selects the best method for your model architecture.

→ [Detailed method guide with math and intuition](https://rigvedrs.github.io/torchxai/methods/)

---

## 🎯 Features

### Accept any input format

```python
# Tensor (already preprocessed)
heatmap = explain(model, tensor)

# PIL Image
heatmap = explain(model, Image.open("dog.jpg"))

# File path (string or Path)
heatmap = explain(model, "dog.jpg")
heatmap = explain(model, Path("dog.jpg"))

# 3D tensor (auto-adds batch dimension)
heatmap = explain(model, torch.randn(3, 224, 224))
```

### Class-specific explanations

Ask "why did the model predict THIS class?" instead of just "where is it looking?"

```python
heatmap_dog = explain(model, img, target_class=208)      # Labrador
heatmap_ball = explain(model, img, target_class=852)      # Tennis ball
heatmap_collar = explain(model, img, target_class=457)    # Bow tie
```

<p align="center">
  <img src="https://raw.githubusercontent.com/rigvedrs/torchxai/main/docs/assets/class_specific.png" alt="Class-specific explanations" width="100%">
</p>

### Customizable visualization

```python
from torchxai import overlay_heatmap, show_explanation, save_heatmap

# Overlay on image
blended = overlay_heatmap(image, heatmap, colormap="jet", alpha=0.5)

# Side-by-side visualization
show_explanation(image, heatmap, save_path="output.png")

# Save raw heatmap
save_heatmap(heatmap, "heatmap.png", colormap="viridis")
```

**Colormap options:**

<p align="center">
  <img src="https://raw.githubusercontent.com/rigvedrs/torchxai/main/docs/assets/colormap_options.png" alt="Colormap options" width="100%">
</p>

**Transparency control:**

<p align="center">
  <img src="https://raw.githubusercontent.com/rigvedrs/torchxai/main/docs/assets/alpha_options.png" alt="Transparency options" width="100%">
</p>

### Quantitative metrics

Don't just look at heatmaps — measure how good they are:

```python
from torchxai import insertion_score, deletion_score, stability_score

# Does removing highlighted pixels drop confidence? (lower = better)
d_score = deletion_score(model, tensor, heatmap)

# Does revealing highlighted pixels restore confidence? (higher = better)
i_score = insertion_score(model, tensor, heatmap)

# Are explanations consistent under small perturbations? (higher = more stable)
s_score = stability_score(GradCAM(model), tensor)
```

### Works on headless servers

No more `TclError: no display name` crashes. torchxai auto-detects headless environments (SSH, Docker, CI) and switches to the Agg matplotlib backend. Visualization functions save to files or return numpy arrays — no display required.

---

## 📖 Examples

### CNN Explainability (ResNet, EfficientNet, VGG)

```python
import torchvision.models as models
from torchxai import explain, show_explanation
from PIL import Image

# Any torchvision model
model = models.resnet50(pretrained=True)

# Explain
heatmap = explain(model, "cat.jpg")

# Visualize
show_explanation(Image.open("cat.jpg"), heatmap, save_path="resnet_explanation.png")
```

### Vision Transformer (ViT, DeiT, Swin)

```python
import timm
from torchxai import explain

# Any timm model — auto-detected
model = timm.create_model("vit_base_patch16_224", pretrained=True)
heatmap = explain(model, "photo.jpg")  # Just works
```

### Custom target layer

```python
from torchxai import GradCAM

# By name
cam = GradCAM(model, target_layer="layer3")

# By module reference
cam = GradCAM(model, target_layer=model.layer3[-1])

heatmap = cam("photo.jpg")
```

### Batch comparison script

```python
from torchxai import explain, create_comparison
from PIL import Image

model = ...  # Your model
image = Image.open("test.jpg")

results = {}
for method in ["gradcam", "eigencam", "layercam", "gradcam_pp"]:
    results[method] = explain(model, image, method=method)

create_comparison(image, results, save_path="all_methods.png")
```

→ [More examples](https://github.com/rigvedrs/torchxai/tree/main/examples)

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](https://rigvedrs.github.io/torchxai/getting-started/) | Installation, first explanation in 60 seconds |
| [API Reference](https://rigvedrs.github.io/torchxai/api-reference/) | Every function, every parameter, return types |
| [Methods Guide](https://rigvedrs.github.io/torchxai/methods/) | How each method works, when to use which |
| [Advanced Usage](https://rigvedrs.github.io/torchxai/advanced/) | Custom layers, metrics, deployment, optimization |
| [Troubleshooting](https://rigvedrs.github.io/torchxai/troubleshooting/) | Common errors with exact fixes |
| [Model Compatibility](https://rigvedrs.github.io/torchxai/model-compatibility/) | Tested models with proof images |
| [Contributing](https://github.com/rigvedrs/torchxai/blob/main/CONTRIBUTING.md) | How to contribute |

---

## 🤝 Contributing

Contributions are welcome. See [CONTRIBUTING.md](https://github.com/rigvedrs/torchxai/blob/main/CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/rigvedrs/torchxai.git
cd torchxai
pip install -e ".[dev]"
pytest tests/ -v
```

---

## 📄 License

MIT License. See [LICENSE](https://github.com/rigvedrs/torchxai/blob/main/LICENSE) for details.

---

## ⭐ Citation

If you use torchxai in your research, please cite:

```bibtex
@software{torchxai2026,
  title={torchxai: Universal Explainability for PyTorch Vision Models},
  author={Rigved Sandeep Shirvalkar},
  year={2026},
  url={https://github.com/rigvedrs/torchxai}
}
```

---

<p align="center">
  <strong>If torchxai helps your work, consider giving it a ⭐</strong>
</p>
