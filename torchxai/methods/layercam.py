"""
LayerCAM — Layer-wise Class Activation Mapping.

Uses positive gradient weighting (element-wise, not channel-wise like GradCAM)
to produce finer-grained saliency maps. Especially effective at earlier layers
where spatial resolution is higher.

Uses ActivationsAndGradients with tensor.register_hook for gradient capture,
avoiding register_full_backward_hook issues with inplace operations.

Reference:
    Jiang et al., "LayerCAM: Exploring Hierarchical Class Activation Maps
    for Localization", IEEE TIP 2021.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from torchxai.methods.base import BaseExplainer
from torchxai.utils.hooks import ActivationsAndGradients


class LayerCAM(BaseExplainer):
    """Layer-wise relevance CAM with per-element gradient weighting.

    Produces finer-grained saliency maps than GradCAM by weighting
    each spatial position independently rather than averaging across
    the spatial dimensions.

    Usage:
        cam = LayerCAM(model)
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
                    "Failed to capture activations/gradients for LayerCAM. "
                    "Try specifying a different target_layer."
                )

            activations = activations_and_grads.activations[0].numpy()
            gradients = activations_and_grads.gradients[0].numpy()

            # LayerCAM: element-wise positive gradient weighting
            positive_grads = np.maximum(gradients, 0)
            cam = (positive_grads * activations).sum(axis=1)  # (B, h, w)

            return cam[0]

        finally:
            activations_and_grads.release()
