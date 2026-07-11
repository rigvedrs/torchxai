# torchxai API Reference

> Comprehensive reference for all public classes and functions in the `torchxai` library.

---

## Table of Contents

1. [Core API](#1-core-api)
   - [`explain`](#explain)
2. [Explainer Classes](#2-explainer-classes)
   - [`BaseExplainer`](#baseexplainer)
   - [`GradCAM`](#gradcam)
   - [`EigenCAM`](#eigencam)
   - [`LayerCAM`](#layercam)
   - [`GradCAMPlusPlus`](#gradcamplusplus)
   - [`AttentionRollout`](#attentionrollout)
   - [`TransformerAttribution`](#transformerattribution)
3. [Visualization Functions](#3-visualization-functions)
   - [`overlay_heatmap`](#overlay_heatmap)
   - [`show_explanation`](#show_explanation)
   - [`create_comparison`](#create_comparison)
   - [`save_heatmap`](#save_heatmap)
4. [Metrics](#4-metrics)
   - [`insertion_score`](#insertion_score)
   - [`deletion_score`](#deletion_score)
   - [`stability_score`](#stability_score)
5. [Utility Functions](#5-utility-functions)
   - [`preprocess_image`](#preprocess_image)
   - [`load_image`](#load_image)
   - [`tensor_to_numpy`](#tensor_to_numpy)
   - [`denormalize`](#denormalize)
   - [`normalize_heatmap`](#normalize_heatmap)
6. [Model Registry](#6-model-registry)
   - [`detect_architecture`](#detect_architecture)
   - [`resolve_target_layer`](#resolve_target_layer)
   - [`find_attention_layers`](#find_attention_layers)
7. [Hook Classes](#7-hook-classes)
   - [`ActivationHook`](#activationhook)
   - [`GradientHook`](#gradienthook)
   - [`MultiHook`](#multihook)
   - [`AttentionHook`](#attentionhook)

---

## 1. Core API

### `explain`

```python
torchxai.explain(
    model: nn.Module,
    image: Union[torch.Tensor, Image.Image, np.ndarray, str, Path],
    method: str = "auto",
    target_class: Optional[int] = None,
    target_layer: Optional[Union[str, nn.Module]] = None,
    image_size: tuple = (224, 224),
) -> np.ndarray
```

The top-level entry point for generating saliency maps. Automatically selects the best explanation method for the given model architecture when `method="auto"`, handles all supported input formats, and returns a normalized heatmap.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | Any PyTorch vision model. Must be in eval mode (`model.eval()`) before calling. |
| `image` | `torch.Tensor \| Image.Image \| np.ndarray \| str \| Path` | — | Input image. Accepted formats: tensor of shape `(1, 3, H, W)` or `(3, H, W)`, PIL `Image`, NumPy array of shape `(H, W, 3)`, file path as `str`, or `pathlib.Path`. |
| `method` | `str` | `"auto"` | Explanation method to use. One of: `"auto"`, `"gradcam"`, `"eigencam"`, `"layercam"`, `"gradcam_pp"`, `"attention_rollout"`, `"transformer_attribution"`. When `"auto"`, the method is selected based on the detected model architecture. |
| `target_class` | `Optional[int]` | `None` | Class index to explain. When `None`, the class with the highest predicted probability is used. |
| `target_layer` | `Optional[Union[str, nn.Module]]` | `None` | Layer to use for explanation. Accepts either the string name of a named module (e.g., `"layer4"`) or a direct module reference. When `None`, the layer is auto-detected via `resolve_target_layer`. |
| `image_size` | `tuple` | `(224, 224)` | Target `(height, width)` to resize non-tensor inputs before processing. Has no effect when `image` is already a tensor. |

**Returns**

`np.ndarray` — A 2-D float array of shape `(H, W)` with values in `[0, 1]`, where higher values indicate greater relevance to the predicted or specified class.

**Example**

```python
import torch
import torchvision.models as models
import torchxai

# Load a pretrained model
model = models.resnet50(pretrained=True).eval()

# Explain from a file path — auto-detects architecture and layer
heatmap = torchxai.explain(model, "cat.jpg")

# Specify a method and target class explicitly
heatmap = torchxai.explain(
    model,
    "cat.jpg",
    method="gradcam",
    target_class=281,         # ImageNet class: tabby cat
    target_layer="layer4",
)

# Use a raw tensor input
import torch
tensor = torch.randn(1, 3, 224, 224)
heatmap = torchxai.explain(model, tensor, method="eigencam")

print(heatmap.shape)   # (224, 224)
print(heatmap.min(), heatmap.max())  # 0.0, 1.0
```

---

## 2. Explainer Classes

All explainer classes share a common base interface defined by `BaseExplainer`. They are callable objects that encapsulate a model and explanation strategy, and can be reused across multiple images without re-initializing.

---

### `BaseExplainer`

```python
torchxai.explainers.BaseExplainer(
    model: nn.Module,
    target_layer: Optional[Union[str, nn.Module]] = None,
    device: Optional[Union[str, torch.device]] = None,
)
```

Abstract base class for all explainer implementations. Not intended to be instantiated directly — use one of the concrete subclasses. Defines the shared interface and lifecycle for explainers.

**Class Attributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `requires_grad` | `bool` | Whether the explainer requires gradient computation. Subclasses that need gradients set this to `True`; gradient-free methods set it to `False`. |

**Constructor Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | The PyTorch model to explain. Should be in eval mode. |
| `target_layer` | `Optional[Union[str, nn.Module]]` | `None` | The layer from which activations (and optionally gradients) are captured. Accepts a module name string or a direct reference. If `None`, the layer is resolved automatically. |
| `device` | `Optional[Union[str, torch.device]]` | `None` | Device to run computations on (e.g., `"cpu"`, `"cuda"`, `torch.device("mps")`). Defaults to the device of the model's parameters. |

**Methods**

#### `__call__`

```python
explainer(
    image: Union[torch.Tensor, Image.Image, np.ndarray, str, Path],
    target_class: Optional[int] = None,
    image_size: tuple = (224, 224),
) -> np.ndarray
```

Runs the explanation on the given image.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `torch.Tensor \| Image.Image \| np.ndarray \| str \| Path` | — | Input image in any supported format. |
| `target_class` | `Optional[int]` | `None` | Target class index. Uses the top-1 predicted class when `None`. |
| `image_size` | `tuple` | `(224, 224)` | Resize target for non-tensor inputs. |

Returns `np.ndarray` of shape `(H, W)` in `[0, 1]`.

#### `__repr__`

```python
explainer.__repr__() -> str
```

Returns a human-readable string summarizing the explainer's configuration, including the model type, target layer name, and device.

---

### `GradCAM`

```python
torchxai.GradCAM(
    model: nn.Module,
    target_layer: Optional[Union[str, nn.Module]] = None,
    device: Optional[Union[str, torch.device]] = None,
)
```

Gradient-weighted Class Activation Mapping. Computes a coarse localization map by weighting each feature map channel with the gradient of the class score with respect to that channel, then applies a ReLU. Works well with most CNN architectures and produces class-discriminative maps.

**Class Attributes**

| Attribute | Value |
|-----------|-------|
| `requires_grad` | `True` |

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | CNN model to explain. |
| `target_layer` | `Optional[Union[str, nn.Module]]` | `None` | Convolutional layer from which to extract feature maps. Defaults to the last convolutional layer. |
| `device` | `Optional[Union[str, torch.device]]` | `None` | Compute device. Inferred from model if `None`. |

**Example**

```python
import torchvision.models as models
from torchxai import GradCAM

model = models.resnet50(pretrained=True).eval()
explainer = GradCAM(model, target_layer="layer4")

heatmap = explainer("dog.jpg", target_class=207)  # golden retriever
print(heatmap.shape)   # (224, 224)
print(repr(explainer)) # GradCAM(model=ResNet, layer=layer4, device=cpu)
```

---

### `EigenCAM`

```python
torchxai.EigenCAM(
    model: nn.Module,
    target_layer: Optional[Union[str, nn.Module]] = None,
    device: Optional[Union[str, torch.device]] = None,
)
```

Eigen-decomposition Class Activation Mapping. Computes the saliency map from the first principal component of the target layer's activations using SVD. Does not require backpropagation, making it faster and more memory-efficient than gradient-based methods. Particularly effective for object detection models where class-discriminativeness is less critical.

**Class Attributes**

| Attribute | Value |
|-----------|-------|
| `requires_grad` | `False` |

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | CNN model to explain. |
| `target_layer` | `Optional[Union[str, nn.Module]]` | `None` | Layer from which to extract activations for PCA. Defaults to the last convolutional layer. |
| `device` | `Optional[Union[str, torch.device]]` | `None` | Compute device. |

**Example**

```python
import torchvision.models as models
from torchxai import EigenCAM

model = models.efficientnet_b0(pretrained=True).eval()

# No gradient context needed — EigenCAM is gradient-free
explainer = EigenCAM(model)

with torch.no_grad():
    heatmap = explainer("bird.jpg")

print(heatmap.shape)  # (224, 224)
```

---

### `LayerCAM`

```python
torchxai.LayerCAM(
    model: nn.Module,
    target_layer: Optional[Union[str, nn.Module]] = None,
    device: Optional[Union[str, torch.device]] = None,
)
```

Layer Class Activation Mapping. An extension of GradCAM that fuses activations element-wise with their corresponding gradients (before global averaging), preserving fine spatial detail. Produces higher-resolution and more precise saliency maps than standard GradCAM, especially when applied to shallow layers.

**Class Attributes**

| Attribute | Value |
|-----------|-------|
| `requires_grad` | `True` |

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | CNN model to explain. |
| `target_layer` | `Optional[Union[str, nn.Module]]` | `None` | Layer to capture. Shallow layers (e.g., `"layer2"`) yield finer details. |
| `device` | `Optional[Union[str, torch.device]]` | `None` | Compute device. |

**Example**

```python
import torchvision.models as models
from torchxai import LayerCAM

model = models.resnet50(pretrained=True).eval()

# Use an earlier layer for fine-grained localization
explainer = LayerCAM(model, target_layer="layer2")
heatmap = explainer("car.jpg", target_class=817)

print(heatmap.shape)  # (224, 224)
```

---

### `GradCAMPlusPlus`

```python
torchxai.GradCAMPlusPlus(
    model: nn.Module,
    target_layer: Optional[Union[str, nn.Module]] = None,
    device: Optional[Union[str, torch.device]] = None,
)
```

Gradient-weighted Class Activation Mapping++ (Grad-CAM++). An improved variant of GradCAM that uses a weighted combination of partial derivatives of the class activation score, yielding better localization of multiple object instances in the same image and cleaner boundary delineation.

**Class Attributes**

| Attribute | Value |
|-----------|-------|
| `requires_grad` | `True` |

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | CNN model to explain. |
| `target_layer` | `Optional[Union[str, nn.Module]]` | `None` | Target convolutional layer. Defaults to the last convolutional layer. |
| `device` | `Optional[Union[str, torch.device]]` | `None` | Compute device. |

**Example**

```python
import torchvision.models as models
from torchxai import GradCAMPlusPlus

model = models.vgg16(pretrained=True).eval()
explainer = GradCAMPlusPlus(model, target_layer="features.28")

# Useful when multiple instances of the same class appear in the image
heatmap = explainer("two_cats.jpg", target_class=281)
print(heatmap.shape)  # (224, 224)
```

---

### `AttentionRollout`

```python
torchxai.AttentionRollout(
    model: nn.Module,
    device: Optional[Union[str, torch.device]] = None,
)
```

Attention Rollout for Vision Transformers (ViT). Computes saliency by recursively multiplying attention weight matrices across all transformer layers to propagate raw attention from output tokens back to input patches. Does not use gradients — works by accumulating attention flow. Best suited for ViT-family models with explicit multi-head self-attention.

**Class Attributes**

| Attribute | Value |
|-----------|-------|
| `requires_grad` | `False` |

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | Vision Transformer model (e.g., ViT, DeiT). |
| `device` | `Optional[Union[str, torch.device]]` | `None` | Compute device. |

> **Note:** `AttentionRollout` does not accept a `target_layer` parameter. Attention hooks are applied automatically across all detected attention layers via `find_attention_layers`.

**Example**

```python
import timm
from torchxai import AttentionRollout

model = timm.create_model("vit_base_patch16_224", pretrained=True).eval()
explainer = AttentionRollout(model)

with torch.no_grad():
    heatmap = explainer("dog.jpg")

print(heatmap.shape)  # (224, 224)
```

---

### `TransformerAttribution`

```python
torchxai.TransformerAttribution(
    model: nn.Module,
    device: Optional[Union[str, torch.device]] = None,
)
```

Gradient-based attribution for Vision Transformers. Combines attention weights with their corresponding gradients (with respect to the target class score) and aggregates across layers and heads. Provides class-discriminative saliency maps for transformer-based models where GradCAM-style methods are inapplicable.

**Class Attributes**

| Attribute | Value |
|-----------|-------|
| `requires_grad` | `True` |

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | Vision Transformer model. |
| `device` | `Optional[Union[str, torch.device]]` | `None` | Compute device. |

> **Note:** Like `AttentionRollout`, `TransformerAttribution` does not expose a `target_layer` parameter. It automatically instruments all attention layers.

**Example**

```python
import timm
from torchxai import TransformerAttribution

model = timm.create_model("swin_base_patch4_window7_224", pretrained=True).eval()
explainer = TransformerAttribution(model)

heatmap = explainer("parrot.jpg", target_class=88)
print(heatmap.shape)  # (224, 224)
```

---

## 3. Visualization Functions

### `overlay_heatmap`

```python
torchxai.visualization.overlay_heatmap(
    image: Union[np.ndarray, Image.Image, torch.Tensor],
    heatmap: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.5,
    output_size: Optional[tuple] = None,
) -> np.ndarray
```

Blends a saliency heatmap onto the original image using a colormap and alpha compositing. Returns an RGB image array suitable for display or saving.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `np.ndarray \| Image.Image \| torch.Tensor` | — | Original image. NumPy arrays should be `(H, W, 3)` in `[0, 255]`. PIL Images and tensors are converted automatically. |
| `heatmap` | `np.ndarray` | — | 2-D float array `(H, W)` in `[0, 1]`. Typically the output of `explain()` or an explainer's `__call__`. |
| `colormap` | `str` | `"jet"` | Matplotlib colormap name applied to the heatmap before overlaying (e.g., `"jet"`, `"viridis"`, `"plasma"`, `"hot"`). |
| `alpha` | `float` | `0.5` | Blending weight of the heatmap in `[0, 1]`. `0.0` returns the original image; `1.0` returns only the colorized heatmap. |
| `output_size` | `Optional[tuple]` | `None` | Resize the output to `(height, width)`. If `None`, the output retains the original image dimensions. |

**Returns**

`np.ndarray` — An `(H, W, 3)` uint8 array representing the blended image in RGB.

**Example**

```python
import numpy as np
from PIL import Image
from torchxai.visualization import overlay_heatmap

image = np.array(Image.open("cat.jpg"))         # (H, W, 3)
heatmap = torchxai.explain(model, "cat.jpg")    # (H, W) in [0,1]

blended = overlay_heatmap(image, heatmap, colormap="viridis", alpha=0.6)
Image.fromarray(blended).save("overlay.png")
```

---

### `show_explanation`

```python
torchxai.visualization.show_explanation(
    image: Union[np.ndarray, Image.Image, torch.Tensor],
    heatmap: np.ndarray,
    title: str = "",
    colormap: str = "jet",
    alpha: float = 0.5,
    figsize: tuple = (12, 4),
    save_path: Optional[Union[str, Path]] = None,
) -> Optional[np.ndarray]
```

Renders a three-panel matplotlib figure showing the original image, the raw heatmap, and the blended overlay side by side. Optionally saves the figure to disk and returns it as a NumPy array for programmatic use.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `np.ndarray \| Image.Image \| torch.Tensor` | — | Original image. |
| `heatmap` | `np.ndarray` | — | 2-D saliency map `(H, W)` in `[0, 1]`. |
| `title` | `str` | `""` | Figure title displayed above all three panels. |
| `colormap` | `str` | `"jet"` | Colormap applied to the heatmap panel and the overlay. |
| `alpha` | `float` | `0.5` | Overlay blending weight `[0, 1]`. |
| `figsize` | `tuple` | `(12, 4)` | Matplotlib figure size `(width_inches, height_inches)`. |
| `save_path` | `Optional[str \| Path]` | `None` | If provided, saves the figure as a PNG/JPG/PDF at this path before displaying. |

**Returns**

`Optional[np.ndarray]` — The rendered figure as an `(H, W, 3)` RGB array when `save_path` is provided; otherwise `None` (figure is shown interactively).

**Example**

```python
from torchxai.visualization import show_explanation

heatmap = torchxai.explain(model, "cat.jpg", method="gradcam")

# Interactive display in a Jupyter notebook
show_explanation("cat.jpg", heatmap, title="GradCAM — ResNet50")

# Save to disk without displaying
show_explanation(
    "cat.jpg",
    heatmap,
    title="GradCAM — ResNet50",
    colormap="plasma",
    figsize=(15, 5),
    save_path="explanation.png",
)
```

---

### `create_comparison`

```python
torchxai.visualization.create_comparison(
    image: Union[np.ndarray, Image.Image, torch.Tensor],
    heatmaps: dict,
    colormap: str = "jet",
    alpha: float = 0.5,
    figsize: Optional[tuple] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Optional[np.ndarray]
```

Generates a multi-panel comparison figure showing the original image alongside overlays for multiple explanation methods. Each entry in `heatmaps` produces one panel. Useful for visually comparing different explainers on the same input.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `np.ndarray \| Image.Image \| torch.Tensor` | — | Original image shown as the first panel. |
| `heatmaps` | `dict` | — | Mapping of `{label: heatmap}` where each key is a panel title (`str`) and each value is a 2-D array `(H, W)` in `[0, 1]`. Panel order follows dictionary insertion order. |
| `colormap` | `str` | `"jet"` | Colormap applied to all heatmap overlays. |
| `alpha` | `float` | `0.5` | Overlay blending weight `[0, 1]` applied to all panels. |
| `figsize` | `Optional[tuple]` | `None` | Figure size `(width, height)`. Defaults to `(4 * (1 + len(heatmaps)), 4)`. |
| `save_path` | `Optional[str \| Path]` | `None` | Save path for the output figure. |

**Returns**

`Optional[np.ndarray]` — Rendered figure as `(H, W, 3)` array if `save_path` is provided; otherwise `None`.

**Example**

```python
from torchxai import GradCAM, EigenCAM, GradCAMPlusPlus
from torchxai.visualization import create_comparison

model = models.resnet50(pretrained=True).eval()
image_path = "dog.jpg"

heatmaps = {
    "GradCAM":      GradCAM(model)(image_path),
    "EigenCAM":     EigenCAM(model)(image_path),
    "GradCAM++":    GradCAMPlusPlus(model)(image_path),
}

create_comparison(
    image_path,
    heatmaps,
    colormap="viridis",
    alpha=0.55,
    save_path="comparison.png",
)
```

---

### `save_heatmap`

```python
torchxai.visualization.save_heatmap(
    heatmap: np.ndarray,
    path: Union[str, Path],
    colormap: str = "jet",
) -> None
```

Applies a colormap to a raw saliency map and writes the resulting image to disk. The output is saved as a standard image file (format determined by the file extension).

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `heatmap` | `np.ndarray` | — | 2-D float array `(H, W)` in `[0, 1]`. |
| `path` | `str \| Path` | — | Output file path. Extension determines format: `.png`, `.jpg`, `.tiff`, etc. |
| `colormap` | `str` | `"jet"` | Matplotlib colormap name used to colorize the heatmap. |

**Returns**

`None`

**Example**

```python
from torchxai.visualization import save_heatmap

heatmap = torchxai.explain(model, "cat.jpg")
save_heatmap(heatmap, "heatmap_jet.png", colormap="jet")
save_heatmap(heatmap, "heatmap_viridis.png", colormap="viridis")
```

---

## 4. Metrics

### `insertion_score`

```python
torchxai.metrics.insertion_score(
    model: nn.Module,
    input_tensor: torch.Tensor,
    heatmap: np.ndarray,
    target_class: Optional[int] = None,
    steps: int = 50,
    baseline: str = "blur",
) -> float
```

Computes the **Insertion** metric, which measures how quickly model confidence rises as image regions identified as important by the heatmap are progressively revealed. Pixels are inserted in descending order of saliency onto a baseline image. A higher score indicates a better-quality explanation.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | Evaluation model in eval mode. Need not be the same model used to generate the heatmap. |
| `input_tensor` | `torch.Tensor` | — | Preprocessed input tensor of shape `(1, 3, H, W)`. |
| `heatmap` | `np.ndarray` | — | 2-D saliency map `(H, W)` in `[0, 1]`. |
| `target_class` | `Optional[int]` | `None` | Class index to evaluate confidence against. Defaults to the top-1 predicted class. |
| `steps` | `int` | `50` | Number of insertion steps. More steps increase accuracy but also computation time. |
| `baseline` | `str` | `"blur"` | Starting image from which pixels are revealed. `"blur"` uses a Gaussian-blurred version of the input; `"black"` uses a zero tensor; `"white"` uses an all-ones tensor. |

**Returns**

`float` — Area under the insertion curve (AUC), normalized to `[0, 1]`. Higher is better.

**Example**

```python
from torchxai.metrics import insertion_score
from torchxai.utils import preprocess_image

model = models.resnet50(pretrained=True).eval()
tensor = preprocess_image("cat.jpg", device="cuda")
heatmap = torchxai.explain(model, tensor)

score = insertion_score(model, tensor, heatmap, steps=100)
print(f"Insertion AUC: {score:.4f}")  # e.g., 0.8213
```

---

### `deletion_score`

```python
torchxai.metrics.deletion_score(
    model: nn.Module,
    input_tensor: torch.Tensor,
    heatmap: np.ndarray,
    target_class: Optional[int] = None,
    steps: int = 50,
    baseline: str = "black",
) -> float
```

Computes the **Deletion** metric, which measures how quickly model confidence drops as important image regions (ranked by the heatmap) are progressively removed. A lower score indicates a better explanation — the model loses confidence faster when truly important regions are deleted.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | Evaluation model in eval mode. |
| `input_tensor` | `torch.Tensor` | — | Preprocessed input tensor of shape `(1, 3, H, W)`. |
| `heatmap` | `np.ndarray` | — | 2-D saliency map `(H, W)` in `[0, 1]`. |
| `target_class` | `Optional[int]` | `None` | Class index to evaluate against. Defaults to top-1 predicted class. |
| `steps` | `int` | `50` | Number of deletion steps. |
| `baseline` | `str` | `"black"` | Replacement value for deleted pixels. `"black"` replaces with zeros; `"blur"` replaces with a blurred region; `"white"` replaces with ones. |

**Returns**

`float` — Area under the deletion curve (AUC), normalized to `[0, 1]`. Lower is better.

**Example**

```python
from torchxai.metrics import deletion_score

score = deletion_score(model, tensor, heatmap, steps=100, baseline="blur")
print(f"Deletion AUC: {score:.4f}")  # e.g., 0.1892
```

---

### `stability_score`

```python
torchxai.metrics.stability_score(
    explain_fn: Callable[[torch.Tensor], np.ndarray],
    input_tensor: torch.Tensor,
    num_perturbations: int = 10,
    noise_scale: float = 0.02,
    seed: Optional[int] = None,
) -> float
```

Measures the **sensitivity** (inverse stability) of an explanation method. Applies multiple small Gaussian perturbations to the input and measures the average pairwise difference between the resulting heatmaps. Lower values indicate more stable, robust explanations.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `explain_fn` | `Callable[[torch.Tensor], np.ndarray]` | — | A callable that accepts a `(1, 3, H, W)` tensor and returns a `(H, W)` heatmap. Typically `lambda x: explainer(x)` or `lambda x: torchxai.explain(model, x)`. |
| `input_tensor` | `torch.Tensor` | — | Clean input image tensor of shape `(1, 3, H, W)`. |
| `num_perturbations` | `int` | `10` | Number of noisy variants to generate and evaluate. |
| `noise_scale` | `float` | `0.02` | Standard deviation of Gaussian noise added to the input. Relative to the input's value range. |
| `seed` | `Optional[int]` | `None` | Random seed for reproducibility. When `None`, results vary across calls. |

**Returns**

`float` — Mean absolute difference between perturbed heatmaps and the baseline heatmap, averaged over all perturbations. Lower is more stable.

**Example**

```python
from torchxai.metrics import stability_score
from torchxai import GradCAM

model = models.resnet50(pretrained=True).eval()
explainer = GradCAM(model)

score = stability_score(
    explain_fn=lambda x: explainer(x),
    input_tensor=tensor,
    num_perturbations=20,
    noise_scale=0.01,
    seed=42,
)
print(f"Stability score: {score:.6f}")  # lower = more stable
```

---

## 5. Utility Functions

### `preprocess_image`

```python
torchxai.utils.preprocess_image(
    image: Union[np.ndarray, Image.Image, str, Path],
    size: tuple = (224, 224),
    mean: Optional[list] = None,
    std: Optional[list] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor
```

Converts an image from any supported format into a normalized `(1, 3, H, W)` float tensor ready for inference. Applies ImageNet normalization by default.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `np.ndarray \| Image.Image \| str \| Path` | — | Input image as a NumPy array `(H, W, 3)`, PIL Image, or file path. |
| `size` | `tuple` | `(224, 224)` | Target `(height, width)` after resizing. |
| `mean` | `Optional[list]` | `None` | Per-channel normalization mean `[R, G, B]`. Defaults to ImageNet mean `[0.485, 0.456, 0.406]`. |
| `std` | `Optional[list]` | `None` | Per-channel normalization standard deviation `[R, G, B]`. Defaults to ImageNet std `[0.229, 0.224, 0.225]`. |
| `device` | `Optional[Union[str, torch.device]]` | `None` | Device to place the output tensor on. Defaults to `"cpu"`. |

**Returns**

`torch.Tensor` — Float tensor of shape `(1, 3, H, W)` normalized and ready for model inference.

**Example**

```python
from torchxai.utils import preprocess_image

# Default ImageNet normalization
tensor = preprocess_image("cat.jpg", size=(224, 224), device="cuda")
print(tensor.shape)   # torch.Size([1, 3, 224, 224])
print(tensor.device)  # cuda:0

# Custom normalization for a model trained on a different dataset
tensor = preprocess_image(
    "cat.jpg",
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
)
```

---

### `load_image`

```python
torchxai.utils.load_image(
    path: Union[str, Path],
    size: tuple = (224, 224),
) -> Image.Image
```

Loads an image from disk, converts it to RGB, and resizes it to the target dimensions. Returns a PIL Image for use with visualization functions or further processing.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path` | — | Path to the image file. Supports any format readable by PIL (JPEG, PNG, BMP, TIFF, WebP, etc.). |
| `size` | `tuple` | `(224, 224)` | Target `(width, height)` passed to `Image.resize`. Note: PIL uses `(width, height)` order. |

**Returns**

`Image.Image` — RGB PIL Image resized to `size`.

**Example**

```python
from torchxai.utils import load_image

img = load_image("photo.jpg", size=(256, 256))
print(img.size)   # (256, 256)
print(img.mode)   # 'RGB'

# Compatible with visualization functions
from torchxai.visualization import overlay_heatmap
import numpy as np
blended = overlay_heatmap(np.array(img), heatmap)
```

---

### `tensor_to_numpy`

```python
torchxai.utils.tensor_to_numpy(
    tensor: torch.Tensor,
) -> np.ndarray
```

Converts a PyTorch image tensor to a NumPy array, handling device transfer, gradient detachment, and channel-order conversion from `(C, H, W)` or `(1, C, H, W)` to `(H, W, C)`.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor` | `torch.Tensor` | — | Image tensor of shape `(C, H, W)` or `(1, C, H, W)`. Values may be in any range; no normalization is applied. |

**Returns**

`np.ndarray` — Float32 array of shape `(H, W, C)`.

**Example**

```python
from torchxai.utils import tensor_to_numpy

# Remove batch dimension and reorder channels
arr = tensor_to_numpy(tensor)    # tensor: (1, 3, 224, 224)
print(arr.shape)                 # (224, 224, 3)
print(arr.dtype)                 # float32
```

---

### `denormalize`

```python
torchxai.utils.denormalize(
    tensor: torch.Tensor,
    mean: Optional[list] = None,
    std: Optional[list] = None,
) -> torch.Tensor
```

Reverses ImageNet-style normalization, converting a normalized tensor back to the `[0, 1]` pixel value range. Useful for visualization of preprocessed inputs.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor` | `torch.Tensor` | — | Normalized image tensor of shape `(C, H, W)` or `(1, C, H, W)`. |
| `mean` | `Optional[list]` | `None` | Mean used during normalization. Defaults to ImageNet mean `[0.485, 0.456, 0.406]`. |
| `std` | `Optional[list]` | `None` | Standard deviation used during normalization. Defaults to ImageNet std `[0.229, 0.224, 0.225]`. |

**Returns**

`torch.Tensor` — Tensor of the same shape as `tensor` with values approximately in `[0, 1]`.

**Example**

```python
from torchxai.utils import preprocess_image, denormalize

tensor = preprocess_image("cat.jpg")          # normalized, (1, 3, 224, 224)
original = denormalize(tensor)                # back to [0, 1]

import numpy as np
arr = (original.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
from PIL import Image
Image.fromarray(arr).save("recovered.png")
```

---

### `normalize_heatmap`

```python
torchxai.utils.normalize_heatmap(
    heatmap: np.ndarray,
) -> np.ndarray
```

Normalizes a raw saliency map to the `[0, 1]` range using min-max scaling. Handles edge cases where the heatmap is constant (returns a zero array).

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `heatmap` | `np.ndarray` | — | Raw 2-D saliency map of any shape and value range. |

**Returns**

`np.ndarray` — Float32 array of the same shape as `heatmap` with values in `[0, 1]`.

**Example**

```python
from torchxai.utils import normalize_heatmap
import numpy as np

raw = np.random.randn(224, 224)   # arbitrary range
normalized = normalize_heatmap(raw)
print(normalized.min(), normalized.max())  # 0.0, 1.0

# Safe on constant arrays
flat = np.ones((224, 224)) * 3.5
result = normalize_heatmap(flat)
print(result.max())  # 0.0
```

---

## 6. Model Registry

### `detect_architecture`

```python
torchxai.registry.detect_architecture(
    model: nn.Module,
) -> ArchType
```

Inspects a model's class hierarchy, layer names, and structural properties to determine its architectural family. The result is used internally by `explain()` and `resolve_target_layer()` to choose an appropriate explanation method and target layer.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | The model to inspect. Does not need to be in eval mode. |

**Returns**

`ArchType` — An enum value representing the detected architecture family. Possible values:

| Value | Description |
|-------|-------------|
| `ArchType.CNN` | Standard convolutional networks (ResNet, VGG, EfficientNet, MobileNet, etc.) |
| `ArchType.VIT` | Plain Vision Transformers (ViT, DeiT) |
| `ArchType.SWIN` | Swin Transformer and variants |
| `ArchType.CLIP` | CLIP visual encoders |
| `ArchType.UNKNOWN` | Architecture could not be classified |

**Example**

```python
from torchxai.registry import detect_architecture, ArchType
import torchvision.models as models
import timm

resnet = models.resnet50(pretrained=True)
print(detect_architecture(resnet))           # ArchType.CNN

vit = timm.create_model("vit_base_patch16_224", pretrained=True)
print(detect_architecture(vit))             # ArchType.VIT

swin = timm.create_model("swin_base_patch4_window7_224", pretrained=True)
print(detect_architecture(swin))            # ArchType.SWIN

# Use the result in branching logic
arch = detect_architecture(model)
if arch == ArchType.CNN:
    explainer = GradCAM(model)
elif arch in (ArchType.VIT, ArchType.SWIN):
    explainer = AttentionRollout(model)
```

---

### `resolve_target_layer`

```python
torchxai.registry.resolve_target_layer(
    model: nn.Module,
    arch_type: ArchType,
) -> nn.Module
```

Returns the best default target layer for saliency-map extraction given a model and its detected architecture type. This is the heuristic used by `explain()` and all explainer classes when `target_layer=None`.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | The model whose layers are being inspected. |
| `arch_type` | `ArchType` | — | Architecture family as returned by `detect_architecture`. |

**Returns**

`nn.Module` — The resolved target layer module. For CNNs, this is typically the last convolutional block (e.g., `model.layer4[-1]` for ResNet). For transformers, this is the final attention or normalization layer.

**Example**

```python
from torchxai.registry import detect_architecture, resolve_target_layer
import torchvision.models as models

model = models.resnet50(pretrained=True)
arch = detect_architecture(model)
layer = resolve_target_layer(model, arch)

print(type(layer))  # <class 'torchvision.models.resnet.Bottleneck'>
print(layer)        # Bottleneck(...)

# Equivalent to passing target_layer=None in GradCAM
from torchxai import GradCAM
explainer = GradCAM(model, target_layer=layer)
```

---

### `find_attention_layers`

```python
torchxai.registry.find_attention_layers(
    model: nn.Module,
) -> list[nn.Module]
```

Traverses the model's named modules and returns all layers identified as attention mechanisms (multi-head self-attention blocks, attention projection layers, etc.). Used internally by `AttentionRollout` and `TransformerAttribution` to register hooks across all attention layers.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | — | The transformer model to inspect. |

**Returns**

`list[nn.Module]` — Ordered list of attention layer modules from first to last in the forward pass. Returns an empty list if no attention layers are found (e.g., for plain CNNs).

**Example**

```python
from torchxai.registry import find_attention_layers
import timm

vit = timm.create_model("vit_base_patch16_224", pretrained=True)
attn_layers = find_attention_layers(vit)

print(len(attn_layers))     # 12 (one per transformer block)
print(type(attn_layers[0])) # <class 'timm.models.vision_transformer.Attention'>

# Inspect the final attention layer
last_attn = attn_layers[-1]
print(last_attn)
```

---

## 7. Hook Classes

Hook classes are context managers that temporarily register PyTorch forward (and optionally backward) hooks on a target layer, storing activations and/or gradients as attributes. All hook classes follow the same pattern: use them with `with` statements to ensure hooks are removed after the forward/backward pass.

---

### `ActivationHook`

```python
torchxai.hooks.ActivationHook(
    layer: nn.Module,
    detach: bool = True,
)
```

Registers a forward hook on `layer` to capture its output activation during a forward pass. Useful for extracting feature maps from intermediate layers without modifying the model.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layer` | `nn.Module` | — | The layer whose output activation will be captured. |
| `detach` | `bool` | `True` | If `True`, the captured activation is detached from the computation graph, preventing unintended gradient flow. Set to `False` if you need gradients through the captured tensor. |

**Attributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `activation` | `torch.Tensor` | The layer's output tensor captured during the most recent forward pass. `None` before any forward pass has occurred. |

**Example**

```python
import torch
import torchvision.models as models
from torchxai.hooks import ActivationHook

model = models.resnet50(pretrained=True).eval()
target_layer = model.layer4[-1]

with ActivationHook(target_layer, detach=True) as hook:
    with torch.no_grad():
        output = model(tensor)
    activation = hook.activation

print(activation.shape)  # e.g., torch.Size([1, 2048, 7, 7])
```

---

### `GradientHook`

```python
torchxai.hooks.GradientHook(
    layer: nn.Module,
)
```

Registers a backward hook on `layer` to capture the gradient of the loss (or any scalar) with respect to the layer's output. The gradient is stored after `.backward()` is called.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layer` | `nn.Module` | — | The layer whose output gradient will be captured during backpropagation. |

**Attributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `gradient` | `torch.Tensor` | The gradient tensor captured during the most recent backward pass. `None` before `.backward()` is called. |

**Example**

```python
import torch
import torchvision.models as models
from torchxai.hooks import GradientHook

model = models.resnet50(pretrained=True).eval()
target_layer = model.layer4[-1]

with GradientHook(target_layer) as hook:
    output = model(tensor)
    class_score = output[0, 281]   # target class logit
    class_score.backward()
    gradient = hook.gradient

print(gradient.shape)  # e.g., torch.Size([1, 2048, 7, 7])
```

---

### `MultiHook`

```python
torchxai.hooks.MultiHook(
    layer: nn.Module,
)
```

Registers both a forward hook and a backward hook on `layer` simultaneously, capturing both the activation and the gradient in a single context manager. Equivalent to using `ActivationHook` and `GradientHook` together, but with less boilerplate.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layer` | `nn.Module` | — | The layer to instrument with both activation and gradient capture. |

**Attributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `activation` | `torch.Tensor` | Forward-pass output of `layer`. |
| `gradient` | `torch.Tensor` | Backward-pass gradient at `layer`'s output. |

**Example**

```python
import torch
import torchvision.models as models
from torchxai.hooks import MultiHook

model = models.resnet50(pretrained=True).eval()
target_layer = model.layer4[-1]

with MultiHook(target_layer) as hook:
    output = model(tensor)
    output[0, 281].backward()

    activation = hook.activation   # (1, 2048, 7, 7)
    gradient = hook.gradient       # (1, 2048, 7, 7)

# Manual GradCAM computation
import torch.nn.functional as F
weights = gradient.mean(dim=(2, 3), keepdim=True)   # (1, 2048, 1, 1)
cam = F.relu((weights * activation).sum(dim=1))      # (1, 7, 7)
```

---

### `AttentionHook`

```python
torchxai.hooks.AttentionHook(
    layer: nn.Module,
)
```

Registers a forward hook on a transformer attention layer to capture the raw attention weight matrix produced by the softmax in multi-head self-attention. Designed for use with ViT-style attention modules where the attention map is an intermediate tensor.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layer` | `nn.Module` | — | A transformer attention module (e.g., `nn.MultiheadAttention` or a `timm` attention block). The hook intercepts the attention weight output. |

**Attributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `attention` | `torch.Tensor` | Attention weight tensor of shape `(batch, heads, seq_len, seq_len)` captured during the most recent forward pass. `None` before any forward pass. |

**Example**

```python
import timm
import torch
from torchxai.hooks import AttentionHook
from torchxai.registry import find_attention_layers

model = timm.create_model("vit_base_patch16_224", pretrained=True).eval()
attn_layers = find_attention_layers(model)

# Capture attention from the last transformer block
with AttentionHook(attn_layers[-1]) as hook:
    with torch.no_grad():
        _ = model(tensor)
    attention = hook.attention

# attention shape: (1, 12, 197, 197) for ViT-B/16 with 12 heads, 196 patches + CLS
print(attention.shape)

# Extract CLS token attention to patch tokens
cls_attention = attention[0, :, 0, 1:]   # (12, 196)
mean_attention = cls_attention.mean(0)    # (196,) — average over heads
```
