"""
High-level one-line API for torchxai.

This is the primary interface for most users. The `explain()` function
auto-detects the model architecture, selects the best explainability
method, and returns a publication-ready saliency map.

Usage:
    from torchxai import explain

    # Simplest — everything auto-detected:
    heatmap = explain(model, image)

    # Works with file paths, PIL images, numpy arrays, and tensors:
    heatmap = explain(model, "photo.jpg")
    heatmap = explain(model, pil_image)
    heatmap = explain(model, numpy_array)
    heatmap = explain(model, tensor)

    # Specify method:
    heatmap = explain(model, image, method="eigencam")

    # With visualization:
    heatmap = explain(model, image, show=True)

    # Compare multiple methods:
    heatmaps = explain(model, image, method="all")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import warnings

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from torchxai.models.registry import ArchType, detect_architecture

# Map of method names to their classes (lazy imports for fast startup)
METHOD_MAP = {
    "gradcam": "torchxai.methods.gradcam.GradCAM",
    "eigencam": "torchxai.methods.eigencam.EigenCAM",
    "layercam": "torchxai.methods.layercam.LayerCAM",
    "gradcam++": "torchxai.methods.gradcam_pp.GradCAMPlusPlus",
    "gradcampp": "torchxai.methods.gradcam_pp.GradCAMPlusPlus",
    "gradcam_pp": "torchxai.methods.gradcam_pp.GradCAMPlusPlus",
    "attention_rollout": "torchxai.methods.attention_rollout.AttentionRollout",
    "rollout": "torchxai.methods.attention_rollout.AttentionRollout",
    "transformer_attribution": "torchxai.methods.transformer_attribution.TransformerAttribution",
    "attribution": "torchxai.methods.transformer_attribution.TransformerAttribution",
    "scorecam": "torchxai.methods.scorecam.ScoreCAM",
    "smoothgrad": "torchxai.methods.smoothgrad.SmoothGrad",
    "integrated_gradients": "torchxai.methods.integrated_gradients.IntegratedGradients",
    "rise": "torchxai.methods.rise.RISE",
}

# Auto-select best method based on architecture
AUTO_METHOD_MAP = {
    ArchType.CNN: "gradcam",
    ArchType.VIT: "attention_rollout",
    ArchType.SWIN: "gradcam",
    ArchType.CLIP: "attention_rollout",
    ArchType.DETR: "attention_rollout",
    ArchType.YOLO: "eigencam",
    # DINOv2 without register tokens has high-norm artifact tokens that act
    # as attention sinks, which hollow out rollout maps. EigenCAM on the
    # patch tokens is the reliable default for DINO-family backbones.
    ArchType.DINO: "eigencam",
    ArchType.GENERIC: "eigencam",
}


def _import_method(dotted_path: str):
    """Lazily import a method class from its dotted path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _load_image(image: Union[str, Path]) -> Image.Image:
    """Load an image from a file path string.

    Supports common formats: JPEG, PNG, BMP, TIFF, WebP.
    Converts to RGB automatically (handles RGBA, grayscale, palette).
    """
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: '{path}'. Check the file path and ensure the file exists."
        )
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        raise ValueError(
            f"Could not load image '{path}': {e}. Supported formats: JPEG, PNG, BMP, TIFF, WebP."
        ) from e


def explain(
    model: nn.Module,
    image: Union[torch.Tensor, Image.Image, np.ndarray, str, Path],
    method: Optional[str] = None,
    target_class: Optional[int] = None,
    target_layer: Optional[Union[nn.Module, str]] = None,
    image_size: Optional[tuple[int, int]] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    colormap: str = "jet",
    alpha: float = 0.5,
    device: Optional[torch.device] = None,
) -> Union[np.ndarray, dict[str, np.ndarray]]:
    """Generate a saliency map explaining the model's prediction.

    This is the primary entry point for torchxai. It auto-detects the
    model architecture, selects the optimal explainability method, and
    produces a publication-ready saliency map.

    Args:
        model: Any PyTorch vision model (CNN, ViT, CLIP, YOLO, etc.)
        image: Input image — accepts ANY of:
            - File path (str or Path): "photo.jpg", Path("./images/dog.png")
            - PIL Image: Image.open("photo.jpg")
            - Numpy array: (H, W, 3) uint8 or float
            - Torch tensor: (1,3,H,W) or (3,H,W)
        method: Explainability method to use. Options:
            - None (auto-select based on architecture)
            - "gradcam", "eigencam", "layercam", "gradcam++"
            - "attention_rollout" / "rollout"
            - "transformer_attribution" / "attribution"
            - "all" (returns dict of all applicable methods)
        target_class: Class index to explain. None = predicted class.
        target_layer: Specific layer to target. None = auto-detect.
        image_size: Size for image preprocessing (H, W). None = resolved
            from the model's expected input size (falls back to 224x224).
        show: If True, display a visualization.
        save_path: If provided, save the visualization to this path.
        colormap: Matplotlib colormap for visualization.
        alpha: Overlay transparency for visualization.
        device: Computation device. Auto-detected if None.

    Returns:
        If method != "all": numpy array (H, W) in [0, 1].
        If method == "all": dict mapping method names to heatmaps.

    Examples:
        >>> from torchxai import explain
        >>> from torchvision.models import resnet50
        >>> model = resnet50(pretrained=True)
        >>> heatmap = explain(model, "dog.jpg")
        >>> heatmap.shape
        (224, 224)
    """
    # Load image from path if string/Path provided
    original_image = image
    if isinstance(image, (str, Path)):
        original_image = _load_image(image)
        image = original_image

    # Detect architecture
    arch_type = detect_architecture(model)

    if method == "all":
        return _explain_all(
            model,
            image,
            original_image,
            arch_type,
            target_class,
            target_layer,
            image_size,
            show,
            save_path,
            colormap,
            alpha,
            device,
        )

    # Auto-select method
    method_name = method
    if method_name is None or method_name == "auto":
        method_name = AUTO_METHOD_MAP.get(arch_type, "eigencam")

    # Import and instantiate the method
    if method_name not in METHOD_MAP:
        available = sorted(set(METHOD_MAP.keys()))
        raise ValueError(f"Unknown method '{method_name}'. Available methods: {available}")

    MethodClass = _import_method(METHOD_MAP[method_name])
    explainer = MethodClass(model=model, target_layer=target_layer, device=device)

    # Generate saliency map
    heatmap = explainer(image, target_class=target_class, image_size=image_size)

    # Visualization
    if show or save_path:
        from torchxai.viz.visualize import show_explanation

        # Use original image for visualization (not preprocessed tensor)
        viz_image = original_image if isinstance(original_image, Image.Image) else image
        show_explanation(
            viz_image,
            heatmap,
            title=method_name.replace("_", " ").title(),
            colormap=colormap,
            alpha=alpha,
            save_path=save_path,
        )

    return heatmap


def _explain_all(
    model,
    image,
    original_image,
    arch_type,
    target_class,
    target_layer,
    image_size,
    show,
    save_path,
    colormap,
    alpha,
    device,
) -> dict[str, np.ndarray]:
    """Run all applicable methods and return results as a dict."""
    # Select methods based on architecture
    if arch_type in (ArchType.VIT, ArchType.DINO, ArchType.CLIP):
        methods = ["eigencam", "gradcam", "attention_rollout"]
    elif arch_type == ArchType.CNN:
        methods = ["gradcam", "eigencam", "layercam", "gradcam++"]
    else:
        methods = ["gradcam", "eigencam"]

    results = {}
    for method_name in methods:
        try:
            MethodClass = _import_method(METHOD_MAP[method_name])
            explainer = MethodClass(model=model, target_layer=target_layer, device=device)
            heatmap = explainer(image, target_class=target_class, image_size=image_size)
            results[method_name] = heatmap
        except Exception as e:
            # Skip methods that fail for this architecture, but say so —
            # silent skips hide real breakage.
            warnings.warn(
                f"Method '{method_name}' failed for this model and was skipped: {e}",
                UserWarning,
                stacklevel=2,
            )

    if not results:
        raise RuntimeError(
            f"All explainability methods failed for this model "
            f"(detected architecture: {arch_type.name}). "
            f"Try specifying a target_layer manually."
        )

    if show or save_path:
        from torchxai.viz.visualize import create_comparison

        viz_image = original_image if isinstance(original_image, Image.Image) else image
        create_comparison(
            viz_image,
            results,
            colormap=colormap,
            alpha=alpha,
            save_path=save_path,
        )

    return results
