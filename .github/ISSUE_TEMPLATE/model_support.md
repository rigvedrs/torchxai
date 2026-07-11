---
name: Model Support Request
about: Request support for a specific model architecture
title: "[Model] "
labels: model-support
assignees: ''
---

**Model name and source**
E.g., "Florence-2 from Microsoft" or "YOLOv12 from Ultralytics"

**Model loading code**
```python
# How to load the model
import timm
model = timm.create_model("model_name", pretrained=True)
```

**Current behavior**
What happens when you try `explain(model, image)` with this model?

**Expected behavior**
What output you expect (saliency map, specific visualization, etc.)
