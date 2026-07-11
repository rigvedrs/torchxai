"""
GradCAM++ — Improved GradCAM with pixel-wise gradient weighting.

Uses second-order gradients to compute per-pixel importance weights,
producing better localization than standard GradCAM, especially for
multiple instances of the same class in one image.

Uses ActivationsAndGradients with tensor.register_hook for gradient capture,
avoiding register_full_backward_hook issues with inplace operations.

Reference:
    Chattopadhyay et al., "Grad-CAM++: Generalized Gradient-based Visual
    Explanations for Deep Convolutional Networks", WACV 2018.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from torchxai.methods.base import BaseExplainer
from torchxai.utils.hooks import ActivationsAndGradients


class GradCAMPlusPlus(BaseExplainer):
    """GradCAM++ with higher-order gradient weighting.

    Better than GradCAM at localizing multiple objects of the same class,
    and produces more complete coverage of each object.

    Usage:
        cam = GradCAMPlusPlus(model)
        heatmap = cam(image)
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
            output = activations_and_grads(input_tensor)

            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim > 2:
                output = output.mean(dim=tuple(range(2, output.ndim)))

            score = self._get_class_score(output, target_class)

            self.model.zero_grad()
            score.backward(retain_graph=False)

            if not activations_and_grads.activations or not activations_and_grads.gradients:
                raise RuntimeError(
                    "Failed to capture activations/gradients for GradCAM++. "
                    "Try specifying a different target_layer."
                )

            activations = activations_and_grads.activations[0].numpy()
            gradients = activations_and_grads.gradients[0].numpy()

            # GradCAM++ weighting (Eq. 19 in the paper, matching
            # pytorch-grad-cam's reference implementation):
            #   alpha = grad^2 / (2 * grad^2 + sum_spatial(A) * grad^3 + eps)
            #   weights = sum_spatial(alpha * relu(grad))
            grad_2 = gradients**2
            grad_3 = grad_2 * gradients

            sum_activations = activations.sum(axis=(2, 3), keepdims=True)
            alpha = grad_2 / (2 * grad_2 + sum_activations * grad_3 + 1e-6)
            # alpha is only defined where the gradient is non-zero
            alpha = np.where(gradients != 0, alpha, 0)

            weights = (alpha * np.maximum(gradients, 0)).sum(axis=(2, 3), keepdims=True)

            cam = (weights * activations).sum(axis=1)  # (B, h, w)

            return cam[0]

        finally:
            activations_and_grads.release()
