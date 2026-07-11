# Troubleshooting

This guide covers every common error, mistake, and question that comes up when using torchxai. Each entry includes the exact error message, its cause, and a complete fix.

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Runtime Errors](#runtime-errors)
3. [Common Mistakes](#common-mistakes)
4. [FAQ](#faq)

---

## Installation Issues

### `ModuleNotFoundError: No module named 'torchxai'`

**Full error:**

```
ModuleNotFoundError: No module named 'torchxai'
```

**Cause:** The PyPI package is named `torchxai-explain`, not `torchxai`. Running `pip install torchxai` installs a different, unrelated package.

**Fix:**

```bash
# Uninstall the wrong package if you installed it
pip uninstall torchxai -y

# Install the correct package
pip install torchxai-explain
```

Then verify:

```bash
python -c "import torchxai; print('OK')"
```

---

### PyTorch Version Conflicts

**Symptoms:** Import errors mentioning `torch`, `torchvision`, or C extension ABI mismatches after installing torchxai.

**Cause:** torchxai requires PyTorch 1.10 or later. Older PyTorch versions are not supported. Mixing PyTorch and torchvision versions that were not built together also causes conflicts.

**Fix:**

Check your current versions:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
```

Install a compatible set. For CPU only:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torchxai-explain
```

For CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torchxai-explain
```

For CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torchxai-explain
```

See the [official PyTorch installation page](https://pytorch.org/get-started/locally/) for the current recommended commands.

---

### GPU vs CPU Installation

**Symptom:** Your GPU is not being used even though it is available.

**Cause:** You installed the CPU-only build of PyTorch.

**Check:**

```python
import torch
print(torch.cuda.is_available())  # Should print True if GPU is present and CUDA is installed
```

**Fix:** If this prints `False` but you have an NVIDIA GPU, reinstall PyTorch with CUDA support (see the version conflict fix above).

torchxai uses the device your model is already on. Move your model to GPU before calling `explain()`:

```python
import torch
import torchvision.models as models
from torchxai import explain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights="IMAGENET1K_V1")
model = model.to(device)
model.eval()

explanation = explain(model, "image.jpg")
```

---

## Runtime Errors

### `_tkinter.TclError: no display name and no $DISPLAY`

**Full error:**

```
_tkinter.TclError: no display name and no $DISPLAY environment variable
```

**Cause:** You called `show_explanation()` on a headless server (no graphical display), or inside a Docker container or CI environment without a display. torchxai handles this automatically for its own rendering, but if you are using matplotlib directly alongside torchxai you may see this error.

**Fix — Option 1 (recommended): Save to a file instead of displaying**

```python
from torchxai import explain, save_heatmap, overlay_heatmap, create_comparison

explanation = explain(model, "image.jpg")

# None of these require a display
save_heatmap(explanation, save_path="outputs/heatmap.png")

overlaid = overlay_heatmap(explanation, alpha=0.5)
overlaid.save("outputs/overlaid.png")

create_comparison(explanation, save_path="outputs/comparison.png")
```

**Fix — Option 2: Set the Agg backend before importing matplotlib**

If you are using matplotlib directly in the same script:

```python
import matplotlib
matplotlib.use("Agg")  # Must be set before importing pyplot
import matplotlib.pyplot as plt

from torchxai import explain

explanation = explain(model, "image.jpg")
# Now use plt.savefig() instead of plt.show()
```

---

### `FileNotFoundError: Image not found: /path/to/img.jpg`

**Full error:**

```
FileNotFoundError: Image not found: /path/to/img.jpg
```

**Cause:** The path you passed to `explain()` does not exist or is misspelled.

**Fix:**

```python
import os
from torchxai import explain

image_path = "path/to/your/image.jpg"

# Check before calling explain
if not os.path.exists(image_path):
    print(f"File not found: {os.path.abspath(image_path)}")
else:
    explanation = explain(model, image_path)
```

Common causes:
- Relative paths resolving differently depending on where you run the script. Use `os.path.abspath()` to see the resolved path.
- The file extension is wrong (e.g., the file is `.jpeg` but you wrote `.jpg`).
- A typo in the directory name.

---

### `TypeError: Unsupported image type`

**Full error:**

```
TypeError: Unsupported image type: <class 'your_type'>. Expected one of: str, pathlib.Path, PIL.Image.Image, numpy.ndarray, torch.Tensor
```

**Cause:** You passed a type that torchxai does not know how to convert.

**Supported input types:**

| Type | Example |
|---|---|
| `str` | `"image.jpg"` |
| `pathlib.Path` | `Path("image.jpg")` |
| `PIL.Image.Image` | `Image.open("image.jpg")` |
| `numpy.ndarray` | `np.array(...)`, shape `(H, W, C)` |
| `torch.Tensor` | shape `(C, H, W)` or `(1, C, H, W)` |

**Fix:** Convert your data to one of the supported types first:

```python
import numpy as np
from PIL import Image
from torchxai import explain

# If you have raw bytes
from io import BytesIO
image_bytes = b"..."  # your bytes
pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
explanation = explain(model, pil_image)

# If you have an OpenCV image (BGR, shape HWC)
import cv2
bgr_image = cv2.imread("image.jpg")
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
explanation = explain(model, rgb_image)  # numpy arrays are accepted
```

---

### `AttributeError: 'ResNet' has no attribute 'layer5'. Available layers: layer1, layer2, layer3, layer4, fc`

**Full error:**

```
AttributeError: 'ResNet' has no attribute 'layer5'. Available layers: layer1, layer2, layer3, layer4, avgpool, fc
```

**Cause:** You specified a `target_layer` that does not exist on your model.

**Fix:** Print all named modules to find valid layer names:

```python
import torchvision.models as models

model = models.resnet50(weights="IMAGENET1K_V1")

# List every named module
for name, module in model.named_modules():
    print(name, "->", type(module).__name__)
```

Then use a name from that list:

```python
from torchxai import GradCAM

# Use a layer that actually exists
explainer = GradCAM(model, target_layer="layer4")
explanation = explainer.explain("image.jpg")
```

For most CNNs, the last convolutional block (`layer4` for ResNet, `features[-1]` for VGG/EfficientNet) gives the best heatmaps. torchxai selects this automatically when you do not specify `target_layer`.

---

### `ValueError: Unknown method 'gradcam2'. Available: gradcam, eigencam, layercam, gradcam++, attention_rollout, transformer_attribution`

**Full error:**

```
ValueError: Unknown method 'gradcam2'. Available: gradcam, eigencam, layercam, gradcam++, attention_rollout, transformer_attribution
```

**Cause:** You passed a method name that torchxai does not recognize.

**Fix:** Use one of the exact strings listed in the error. All valid method names:

```python
from torchxai import explain

# All valid method strings
explain(model, image, method="gradcam")               # default
explain(model, image, method="eigencam")
explain(model, image, method="layercam")
explain(model, image, method="gradcam++")
explain(model, image, method="attention_rollout")
explain(model, image, method="transformer_attribution")
```

---

### `RuntimeError: Can't call numpy() on Tensor that requires grad`

**Full error:**

```
RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
```

**Cause:** You tried to convert a tensor that still has an active gradient tape to a NumPy array.

**Fix:** Call `.detach()` before `.numpy()`:

```python
import torch

tensor = torch.randn(3, 224, 224, requires_grad=True)

# Wrong
array = tensor.numpy()  # raises RuntimeError

# Correct
array = tensor.detach().numpy()

# If the tensor is on GPU, also move it to CPU first
array = tensor.detach().cpu().numpy()
```

If this error occurs inside your own code that processes the explanation output:

```python
from torchxai import explain
import numpy as np

explanation = explain(model, "image.jpg")

# Access the raw heatmap tensor safely
heatmap_array = explanation.heatmap.detach().cpu().numpy()
```

---

### `CUDA out of memory`

**Full error:**

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X MiB (GPU X; X GiB total capacity; X GiB already allocated)
```

**Cause:** The explanation method requires gradient computation through the full model, and your GPU does not have enough memory for the image size or batch size you are using.

**Fix — Option 1: Use EigenCAM (no gradients required)**

```python
from torchxai import explain

# EigenCAM uses PCA on activations — no backward pass, far lower memory usage
explanation = explain(model, "image.jpg", method="eigencam")
```

**Fix — Option 2: Reduce image size**

```python
from PIL import Image
from torchxai import explain

# Resize before passing in
image = Image.open("image.jpg").resize((224, 224))
explanation = explain(model, image)
```

**Fix — Option 3: Use `torch.no_grad()` where possible**

torchxai manages its own gradient context, but if you are running other GPU operations before calling `explain()`, free memory first:

```python
import torch
from torchxai import explain

# Clear the cache before explaining
torch.cuda.empty_cache()

explanation = explain(model, "image.jpg")
```

**Fix — Option 4: Process on CPU**

```python
import torchvision.models as models
from torchxai import explain

model = models.resnet50(weights="IMAGENET1K_V1")
model = model.cpu()  # Keep on CPU
model.eval()

explanation = explain(model, "image.jpg")
```

---

## Common Mistakes

### Forgetting `model.eval()`

**Symptom:** Heatmaps look noisy or inconsistent between runs, even on the same image.

**Cause:** In training mode, dropout layers are active and batch normalization uses batch statistics, making outputs non-deterministic.

**Fix:** Always call `model.eval()` before generating explanations:

```python
import torchvision.models as models
from torchxai import explain

model = models.resnet50(weights="IMAGENET1K_V1")
model.eval()  # <-- required

explanation = explain(model, "image.jpg")
```

---

### Passing Unnormalized Images to Models Expecting ImageNet Normalization

**Symptom:** The model makes wrong predictions; the heatmap highlights irrelevant regions.

**Cause:** Most pretrained torchvision models expect inputs normalized with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. torchxai applies this normalization automatically when you pass a file path, PIL Image, or NumPy array. However, if you pass a raw `torch.Tensor`, torchxai assumes you have already normalized it.

**Fix — Option 1: Let torchxai handle it**

Pass a file path, PIL Image, or NumPy array and torchxai normalizes for you:

```python
from torchxai import explain

# torchxai preprocesses and normalizes automatically
explanation = explain(model, "image.jpg")
explanation = explain(model, pil_image)
explanation = explain(model, numpy_array)  # shape (H, W, C), values 0-255
```

**Fix — Option 2: Normalize your tensor manually**

If you must pass a tensor, normalize it first:

```python
import torch
import torchvision.transforms as T
from torchxai import explain

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Raw tensor with values in [0, 1]
raw_tensor = your_image_tensor  # shape (1, 3, H, W)
normalized_tensor = normalize(raw_tensor.squeeze(0)).unsqueeze(0)

explanation = explain(model, normalized_tensor)
```

---

### Wrong Tensor Shape (HWC vs CHW, Missing Batch Dimension)

**Symptom:** `RuntimeError` about tensor dimensions, or explanations that look completely wrong.

**Cause:** PyTorch uses channel-first layout `(C, H, W)` or `(N, C, H, W)`. NumPy and PIL use channel-last `(H, W, C)`. Mixing these up causes silent errors.

**Fix:**

```python
import torch
import numpy as np
from torchxai import explain

# NumPy arrays should be (H, W, C) — torchxai converts internally
numpy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
explanation = explain(model, numpy_image)  # correct

# Tensors should be (C, H, W) or (1, C, H, W)
tensor_3d = torch.randn(3, 224, 224)    # (C, H, W) — accepted
tensor_4d = torch.randn(1, 3, 224, 224) # (N, C, H, W) — accepted

# Wrong: (H, W, C) tensor
wrong_tensor = torch.randn(224, 224, 3)  # do not pass this

# Fix: permute to (C, H, W) first
correct_tensor = wrong_tensor.permute(2, 0, 1)
explanation = explain(model, correct_tensor)
```

---

### Using Attention Methods on CNNs (and Vice Versa)

**Symptom:** You get a warning message and the explanation method silently falls back to GradCAM.

**Cause:** `attention_rollout` and `transformer_attribution` require attention layers and do not work on CNNs. GradCAM, EigenCAM, LayerCAM, and GradCAM++ require convolutional layers and do not work on Vision Transformers.

**Fix:** Match the method to your architecture:

```python
import torchvision.models as models
from torchxai import explain

# CNN — use gradient-based or CAM methods
cnn = models.resnet50(weights="IMAGENET1K_V1")
cnn.eval()
explanation = explain(cnn, "image.jpg", method="gradcam")   # correct
explanation = explain(cnn, "image.jpg", method="eigencam")  # correct

# Vision Transformer — use attention-based methods
vit = models.vit_b_16(weights="IMAGENET1K_V1")
vit.eval()
explanation = explain(vit, "image.jpg", method="attention_rollout")      # correct
explanation = explain(vit, "image.jpg", method="transformer_attribution") # correct
```

---

## FAQ

### Which method should I use?

| Situation | Recommended Method |
|---|---|
| CNN, general use | `gradcam` |
| CNN, need speed | `eigencam` |
| CNN, multiple objects | `gradcam++` |
| CNN, need fine detail | `layercam` |
| Vision Transformer, general use | `attention_rollout` |
| Vision Transformer, highest quality | `transformer_attribution` |

When in doubt, start with `gradcam`. It is the most well-studied method, works reliably across CNN architectures, and produces high-quality heatmaps.

```python
from torchxai import explain

# Sensible default for any CNN
explanation = explain(model, "image.jpg", method="gradcam")
```

---

### Does it work with my custom model?

Yes, as long as your model:

1. Is a `torch.nn.Module` subclass (all PyTorch models are).
2. Has at least one convolutional layer (for CAM methods) or attention layer (for attention methods).
3. Accepts a batched image tensor `(1, C, H, W)` as input and returns a class score tensor.

```python
import torch
import torch.nn as nn
from torchxai import GradCAM, show_explanation

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

model = MyModel()
model.eval()

# Specify the target layer explicitly for custom models
explainer = GradCAM(model, target_layer="features.0")
explanation = explainer.explain("image.jpg")
show_explanation(explanation)
```

If you are unsure which layer to target, print all named modules as shown in the [AttributeError section](#attributeerror-x-has-no-attribute-y-available-layers-) above.

---

### Can I use it on a GPU?

Yes. Move your model to the GPU before calling `explain()`. torchxai detects the device automatically:

```python
import torch
import torchvision.models as models
from torchxai import explain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights="IMAGENET1K_V1")
model = model.to(device)
model.eval()

explanation = explain(model, "image.jpg")
print(explanation.heatmap.device)  # cuda:0
```

If you run out of GPU memory, see the [CUDA out of memory](#cuda-out-of-memory) section above.

---

### Why does my heatmap look uniform or blank?

A blank or nearly-uniform heatmap (all one color) usually has one of these causes:

**1. Model is not in eval mode**

```python
model.eval()  # Always do this
```

**2. The model's prediction is very low-confidence**

If the model is not confident about any class, the gradients with respect to any class score will be near zero. Check the prediction first:

```python
import torch
import torchvision.transforms as T
from PIL import Image

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("image.jpg").convert("RGB")
tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    top_prob, top_class = probs.max(dim=1)
    print(f"Top class: {top_class.item()}, confidence: {top_prob.item():.4f}")
```

If confidence is below ~0.1, the model may not recognize the image (wrong normalization, out-of-distribution image, or wrong model).

**3. You are targeting the wrong layer**

Very early layers in a CNN contain low-level features (edges) and produce diffuse heatmaps. Target a later convolutional block:

```python
from torchxai import GradCAM

# Use the last convolutional block for the most class-specific heatmap
explainer = GradCAM(model, target_layer="layer4")  # for ResNet
explanation = explainer.explain("image.jpg")
```

**4. Image is not normalized correctly**

See [Passing unnormalized images](#passing-unnormalized-images-to-models-expecting-imagenet-normalization) above.

---

### How do I use this on a server / Docker / CI?

torchxai works on headless environments without any configuration. The only thing to avoid is calling `show_explanation()` without a display. Use file-based output instead:

```python
from torchxai import explain, save_heatmap, overlay_heatmap, create_comparison

explanation = explain(model, "image.jpg")

# All of these work on headless servers
save_heatmap(explanation, save_path="outputs/heatmap.png")

overlaid = overlay_heatmap(explanation, alpha=0.5)
overlaid.save("outputs/overlaid.png")

create_comparison(explanation, save_path="outputs/comparison.png")
```

**Docker:** No special configuration needed. Install torchxai as normal in your `Dockerfile`:

```dockerfile
FROM python:3.11-slim
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install torchxai-explain
```

**GitHub Actions / CI:** Same as above. Use `save_path` arguments everywhere instead of `show_explanation()`.

---

### Can I explain object detection models?

torchxai is primarily designed for image classification models. It can be applied to object detectors, but requires some adaptation since detection models do not output a single class score vector.

The general approach is to wrap your detector to expose a classifiable output for a specific detected box:

```python
import torch
import torch.nn as nn
from torchxai import GradCAM, show_explanation

class DetectorWrapper(nn.Module):
    """Wraps an object detector to expose the score of a specific detection."""
    def __init__(self, detector, box_index=0, class_index=0):
        super().__init__()
        self.detector = detector
        self.box_index = box_index
        self.class_index = class_index

    def forward(self, x):
        outputs = self.detector(x)
        # Return the score for the target box and class as a (1, 1) tensor
        # Adjust this indexing to match your detector's output format
        score = outputs[0]["scores"][self.box_index].unsqueeze(0).unsqueeze(0)
        return score

# Example usage (pseudo-code — adjust for your detector)
detector = your_object_detector
detector.eval()

wrapped = DetectorWrapper(detector, box_index=0, class_index=0)

explainer = GradCAM(wrapped, target_layer="backbone.layer4")
explanation = explainer.explain("image.jpg")
show_explanation(explanation)
```

Results will highlight the image region responsible for the selected detection.
