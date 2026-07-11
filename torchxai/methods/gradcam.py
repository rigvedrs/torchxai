"""
GradCAM — Gradient-weighted Class Activation Mapping.

The most popular CAM method. Works by weighting each feature map channel
by the global-average-pooled gradient of the target class w.r.t. that channel.

Supports both CNNs and Vision Transformers (with automatic reshape via
ActivationsAndGradients hook).

Uses forward hooks for BOTH activations and gradients (via tensor.register_hook)
instead of register_full_backward_hook, which fixes VGG/DenseNet inplace ReLU
crashes and produces correct heatmaps on EfficientNetV2, MobileNetV3/V4, etc.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", IJCV 2020.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from torchxai.methods.base import BaseExplainer
from torchxai.utils.hooks import ActivationsAndGradients


class GradCAM(BaseExplainer):
    """Gradient-weighted Class Activation Mapping.

    The most widely-used explainability method. Produces class-discriminative
    saliency maps by weighting feature map channels by their gradient
    importance.

    Usage:
        cam = GradCAM(model)
        heatmap = cam(image)  # numpy array (H, W) in [0, 1]

        # Explain a specific class:
        heatmap = cam(image, target_class=243)
    """

    requires_grad = True

    def _compute_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int],
    ) -> np.ndarray:
        activations_and_grads = ActivationsAndGradients(
            self.model,
            [self.target_layer],
            reshape_transform=self.reshape_transform,
        )

        try:
            # Forward pass — captures activations and registers gradient hooks
            output = activations_and_grads(input_tensor)

            # Handle different output formats
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim > 2:
                output = output.mean(dim=tuple(range(2, output.ndim)))

            # Get target class score
            score = self._get_class_score(output, target_class)

            # Backward pass — gradient hooks fire and store gradients
            self.model.zero_grad()
            score.backward(retain_graph=False)

            # Get activations and gradients (already reshaped by hook if needed)
            if not activations_and_grads.activations:
                raise RuntimeError(
                    f"Failed to capture activations from "
                    f"{type(self.target_layer).__name__}. "
                    f"Ensure the target layer is on the forward path."
                )
            if not activations_and_grads.gradients:
                raise RuntimeError(
                    f"Failed to capture gradients from "
                    f"{type(self.target_layer).__name__}. "
                    f"This can happen if:\n"
                    f"  1. The target layer isn't on the forward path\n"
                    f"  2. The model uses in-place operations that break the graph\n"
                    f"  3. torch.no_grad() is active (it shouldn't be for GradCAM)\n"
                    f"Try specifying a different target_layer."
                )

            activations = activations_and_grads.activations[0]
            gradients = activations_and_grads.gradients[0]

            # activations/gradients are already (B, C, H, W) — reshaped by hook
            # Global average pool gradients over spatial dims
            weights = np.mean(gradients.numpy(), axis=(2, 3))  # (B, C)

            # Weighted combination of activation maps
            cam = (weights[:, :, None, None] * activations.numpy()).sum(axis=1)  # (B, H, W)

            return cam[0]

        finally:
            activations_and_grads.release()
