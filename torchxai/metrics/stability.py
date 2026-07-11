"""
Stability metric for saliency maps.

Measures whether small perturbations to the input produce similar
saliency maps. A stable explanation method should produce consistent
results under minor noise.

Higher stability score = more consistent explanations.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch


def stability_score(
    explain_fn: Callable[[torch.Tensor], np.ndarray],
    input_tensor: torch.Tensor,
    num_perturbations: int = 10,
    noise_scale: float = 0.02,
    seed: Optional[int] = None,
) -> float:
    """Compute the stability score of an explanation method.

    Generates slightly perturbed versions of the input and measures
    the consistency of the resulting saliency maps using cosine similarity.

    Args:
        explain_fn: A function that takes an input tensor and returns
            a saliency map (numpy array). Usually `method.__call__`.
        input_tensor: The original input tensor (1, 3, H, W).
        num_perturbations: Number of perturbations to test.
        noise_scale: Standard deviation of Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        Mean cosine similarity between the original and perturbed
        saliency maps. Range [0, 1]. Higher = more stable.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Get baseline saliency map
    # Detach input to avoid "requires grad" errors during numpy conversion
    input_detached = input_tensor.detach()
    base_heatmap = explain_fn(input_detached)
    base_flat = base_heatmap.flatten()

    # Normalize baseline
    base_norm = np.linalg.norm(base_flat)
    if base_norm < 1e-8:
        return 0.0  # Trivial map

    similarities = []

    for _ in range(num_perturbations):
        # Add small Gaussian noise
        noise = torch.randn_like(input_detached) * noise_scale
        perturbed = input_detached + noise

        # Get perturbed saliency map
        perturbed_heatmap = explain_fn(perturbed)
        perturbed_flat = perturbed_heatmap.flatten()

        perturbed_norm = np.linalg.norm(perturbed_flat)
        if perturbed_norm < 1e-8:
            similarities.append(0.0)
            continue

        # Cosine similarity
        cos_sim = np.dot(base_flat, perturbed_flat) / (base_norm * perturbed_norm)
        similarities.append(float(cos_sim))

    return float(np.mean(similarities))
