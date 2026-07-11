"""
Fidelity metrics: Insertion and Deletion scores.

These metrics measure whether the saliency map correctly identifies
the most important image regions for the model's prediction.

- **Deletion**: Progressively remove the most important pixels.
  A good saliency map causes the confidence to drop quickly.
  Lower AUC = better explanation.

- **Insertion**: Progressively reveal the most important pixels
  on a blank canvas. A good saliency map causes confidence to
  rise quickly. Higher AUC = better explanation.

Reference:
    Petsiuk et al., "RISE: Randomized Input Sampling for Explanation
    of Black-box Models", BMVC 2018.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def deletion_score(
    model: nn.Module,
    input_tensor: torch.Tensor,
    heatmap: np.ndarray,
    target_class: Optional[int] = None,
    steps: int = 50,
    baseline: str = "black",
) -> float:
    """Compute the deletion score (AUC) for a saliency map.

    Progressively replaces the most important pixels (as identified
    by the heatmap) with a baseline value, measuring how quickly
    the model's confidence drops.

    A LOWER score indicates a BETTER saliency map.

    Args:
        model: The model.
        input_tensor: Preprocessed input (1, 3, H, W).
        heatmap: Saliency map (H, W) in [0, 1].
        target_class: Class to measure. If None, uses predicted class.
        steps: Number of deletion steps.
        baseline: "black" (zeros) or "blur" (Gaussian blur).

    Returns:
        AUC score in [0, 1]. Lower is better.
    """
    device = input_tensor.device
    model.eval()

    B, C, H, W = input_tensor.shape

    # Resize heatmap to match input
    if heatmap.shape[0] != H or heatmap.shape[1] != W:
        from PIL import Image

        hm = Image.fromarray((heatmap * 255).astype(np.uint8))
        hm = hm.resize((W, H), Image.BILINEAR)
        heatmap = np.array(hm).astype(np.float32) / 255.0

    # Create baseline
    if baseline == "blur":
        blurred = F.avg_pool2d(input_tensor, kernel_size=11, stride=1, padding=5)
    else:
        blurred = torch.zeros_like(input_tensor)

    # Get initial prediction
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim > 2:
            output = output.mean(dim=tuple(range(2, output.ndim)))
        probs = F.softmax(output, dim=-1)
        if target_class is None:
            target_class = probs.argmax(dim=-1).item()

    # Sort pixels by importance (highest first for deletion)
    flat_heatmap = heatmap.flatten()
    sorted_indices = np.argsort(-flat_heatmap)

    # Compute scores at each step
    scores = []
    pixels_per_step = max(1, len(sorted_indices) // steps)

    modified = input_tensor.clone()

    for step in range(steps + 1):
        with torch.no_grad():
            out = model(modified)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.ndim > 2:
                out = out.mean(dim=tuple(range(2, out.ndim)))
            prob = F.softmax(out, dim=-1)[0, target_class].item()
        scores.append(prob)

        if step < steps:
            start_idx = step * pixels_per_step
            end_idx = min(start_idx + pixels_per_step, len(sorted_indices))
            pixel_indices = sorted_indices[start_idx:end_idx]

            for idx in pixel_indices:
                row, col = divmod(idx, W)
                modified[0, :, row, col] = blurred[0, :, row, col]

    # Compute AUC using trapezoidal rule
    # np.trapz was removed in NumPy 2.0, use np.trapezoid if available
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = _trapz(scores, dx=1.0 / steps) / 1.0
    return float(auc)


def insertion_score(
    model: nn.Module,
    input_tensor: torch.Tensor,
    heatmap: np.ndarray,
    target_class: Optional[int] = None,
    steps: int = 50,
    baseline: str = "blur",
) -> float:
    """Compute the insertion score (AUC) for a saliency map.

    Starts from a baseline image and progressively reveals the most
    important pixels. A good saliency map causes the confidence to
    rise quickly.

    A HIGHER score indicates a BETTER saliency map.

    Args:
        model: The model.
        input_tensor: Preprocessed input (1, 3, H, W).
        heatmap: Saliency map (H, W) in [0, 1].
        target_class: Class to measure. If None, uses predicted class.
        steps: Number of insertion steps.
        baseline: Starting baseline. "blur" or "black".

    Returns:
        AUC score in [0, 1]. Higher is better.
    """
    device = input_tensor.device
    model.eval()

    B, C, H, W = input_tensor.shape

    # Resize heatmap
    if heatmap.shape[0] != H or heatmap.shape[1] != W:
        from PIL import Image

        hm = Image.fromarray((heatmap * 255).astype(np.uint8))
        hm = hm.resize((W, H), Image.BILINEAR)
        heatmap = np.array(hm).astype(np.float32) / 255.0

    # Create baseline
    if baseline == "blur":
        canvas = F.avg_pool2d(input_tensor, kernel_size=11, stride=1, padding=5)
    else:
        canvas = torch.zeros_like(input_tensor)

    # Get target class from original prediction
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim > 2:
            output = output.mean(dim=tuple(range(2, output.ndim)))
        probs = F.softmax(output, dim=-1)
        if target_class is None:
            target_class = probs.argmax(dim=-1).item()

    # Sort by importance (highest first for insertion)
    flat_heatmap = heatmap.flatten()
    sorted_indices = np.argsort(-flat_heatmap)

    scores = []
    pixels_per_step = max(1, len(sorted_indices) // steps)

    modified = canvas.clone()

    for step in range(steps + 1):
        with torch.no_grad():
            out = model(modified)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.ndim > 2:
                out = out.mean(dim=tuple(range(2, out.ndim)))
            prob = F.softmax(out, dim=-1)[0, target_class].item()
        scores.append(prob)

        if step < steps:
            start_idx = step * pixels_per_step
            end_idx = min(start_idx + pixels_per_step, len(sorted_indices))
            pixel_indices = sorted_indices[start_idx:end_idx]

            for idx in pixel_indices:
                row, col = divmod(idx, W)
                modified[0, :, row, col] = input_tensor[0, :, row, col]

    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = _trapz(scores, dx=1.0 / steps) / 1.0
    return float(auc)
