---
name: Bug Report
about: Report a problem with torchxai explainability methods
title: "[Bug] "
labels: bug
assignees: ''
---

## Description

<!-- A clear and concise description of what the bug is. -->

## System Information

<!-- Please fill in ALL fields. This is essential for reproducing your issue. -->

| Field              | Value |
| ------------------ | ----- |
| **OS**             | <!-- e.g., Ubuntu 22.04, macOS 14.4, Windows 11 --> |
| **Python version** | <!-- e.g., 3.11.7 — run: python --version --> |
| **PyTorch version**| <!-- e.g., 2.3.0+cu121 — run: python -c "import torch; print(torch.__version__)" --> |
| **torchxai version**| <!-- e.g., 0.1.0 — run: python -c "import torchxai; print(torchxai.__version__)" --> |
| **CUDA version**   | <!-- e.g., 12.1 or N/A — run: nvcc --version --> |
| **GPU**            | <!-- e.g., NVIDIA RTX 4090 or CPU only --> |
| **torchvision**    | <!-- run: python -c "import torchvision; print(torchvision.__version__)" --> |
| **timm (if used)** | <!-- run: python -c "import timm; print(timm.__version__)" or N/A --> |

## Model and Method

- **Model architecture**: <!-- e.g., ResNet50, ViT-B/16, CLIP ViT-L/14, YOLOv8n -->
- **Explainability method**: <!-- e.g., GradCAM, EigenCAM, AttentionRollout -->
- **Using `explain()` API or direct class?**: <!-- e.g., explain() / GradCAM() -->

## Minimal Reproducible Example

<!--
Please provide the SMALLEST possible code snippet that reproduces the bug.
Do NOT paste entire notebooks — trim it down to the essentials.
If the error requires a specific image or checkpoint, describe it briefly.
-->

```python
import torch
import torchxai

# Example:
# model = ...
# saliency = torchxai.explain(model, image)
```

## Expected Behavior

<!-- What did you expect to happen? -->

## Actual Behavior

<!-- What actually happened? Paste the full error traceback below. -->

```
Paste full traceback here
```

## Additional Context

<!-- Any other information that might be relevant: custom layers, unusual input sizes, multi-GPU setup, etc. -->
