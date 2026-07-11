"""
Image preprocessing and utility functions for torchxai.

Handles loading, preprocessing, and converting images between
different formats (PIL, numpy, torch tensors) used across the library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image
import torch

# Standard ImageNet normalization used by most vision models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image(
    path: Union[str, Path],
    size: tuple[int, int] = (224, 224),
) -> Image.Image:
    """Load an image from disk and resize it.

    Args:
        path: Path to image file.
        size: Target (width, height). Defaults to (224, 224).

    Returns:
        PIL.Image in RGB mode.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    return img


def preprocess_image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    size: tuple[int, int] = (224, 224),
    mean: Optional[list[float]] = None,
    std: Optional[list[float]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert an image to a normalized tensor ready for model input.

    Handles PIL images, numpy arrays, and existing tensors.

    Args:
        image: Input image in any supported format.
        size: Target (height, width) for resizing.
        mean: Normalization mean per channel. Defaults to ImageNet.
        std: Normalization std per channel. Defaults to ImageNet.
        device: Target device. Defaults to CPU.

    Returns:
        Tensor of shape (1, 3, H, W), normalized and batched.
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    # Convert to tensor
    if isinstance(image, Image.Image):
        image = image.resize((size[1], size[0]), Image.BILINEAR)
        tensor = torch.from_numpy(np.array(image)).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
    elif isinstance(image, np.ndarray):
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).float()
        if tensor.ndim == 3 and tensor.shape[2] == 3:
            tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
    elif isinstance(image, torch.Tensor):
        tensor = image.float()
        if tensor.ndim == 3 and tensor.shape[0] != 3:
            tensor = tensor.permute(2, 0, 1)
    else:
        raise TypeError(
            f"Unsupported image type: {type(image)}. "
            f"Expected a torch.Tensor, PIL.Image, numpy array, "
            f"or a file path (str/Path) to an image. "
            f"Example: explain(model, 'photo.jpg') or explain(model, tensor)"
        )

    # Normalize
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    tensor = (tensor - mean_t) / std_t

    # Add batch dimension
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    if device is not None:
        tensor = tensor.to(device)

    return tensor


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to numpy array in HWC format, values in [0, 1].

    Args:
        tensor: Input tensor of shape (C, H, W) or (1, C, H, W).

    Returns:
        Numpy array of shape (H, W, C) with values in [0, 1].
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
        tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    return arr


def denormalize(
    tensor: torch.Tensor,
    mean: Optional[list[float]] = None,
    std: Optional[list[float]] = None,
) -> torch.Tensor:
    """Reverse ImageNet normalization on a tensor.

    Args:
        tensor: Normalized tensor (C, H, W) or (1, C, H, W).
        mean: Normalization mean. Defaults to ImageNet.
        std: Normalization std. Defaults to ImageNet.

    Returns:
        Denormalized tensor.
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    squeeze = False
    if tensor.ndim == 4:
        squeeze = True
        tensor = tensor.squeeze(0)

    mean_t = torch.tensor(mean, device=tensor.device).view(3, 1, 1)
    std_t = torch.tensor(std, device=tensor.device).view(3, 1, 1)
    tensor = tensor * std_t + mean_t
    tensor = torch.clamp(tensor, 0, 1)

    if squeeze:
        tensor = tensor.unsqueeze(0)
    return tensor


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Normalize a heatmap to [0, 1] range.

    Args:
        heatmap: Raw heatmap of any shape.

    Returns:
        Normalized heatmap in [0, 1].
    """
    heatmap = heatmap.astype(np.float32)
    vmin, vmax = heatmap.min(), heatmap.max()
    if vmax - vmin > 1e-8:
        heatmap = (heatmap - vmin) / (vmax - vmin)
    else:
        heatmap = np.zeros_like(heatmap)
    return heatmap
