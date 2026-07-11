"""
EigenCAM — Gradient-free CAM using PCA on activations.

Computes the first principal component of the activation maps,
which captures the most significant pattern in the feature maps.
Does NOT require gradients, making it faster and compatible with
models where backward passes are problematic (e.g., quantized models).

Uses ActivationsAndGradients for activation capture (forward hook only),
with reshape_transform applied inside the hook for ViT/Swin compatibility.

Implementation follows pytorch-grad-cam's get_2d_projection:
1. Reshape activations to (C, H*W)
2. Transpose to (H*W, C)
3. Center by subtracting the mean
4. SVD to get principal components
5. Project onto first component: projection = centered @ V[0]

The SVD projection's sign is inherently ambiguous ((-U)S(-V^T) is an equally
valid decomposition). The signed map is returned as-is; BaseExplainer
resolves the faithful polarity with a masked-input confidence test, which is
far more reliable than magnitude heuristics (which invert on ViTs whose
high-norm tokens sit in the background).

Reference:
    Muhammad & Yeasin, "Eigen-CAM: Class Activation Map using Principal
    Components", IJCNN 2020.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from torchxai.methods.base import BaseExplainer
from torchxai.utils.hooks import ActivationsAndGradients


class EigenCAM(BaseExplainer):
    """Gradient-free explainability using PCA on feature activations.

    This is the fastest CAM method since it doesn't need a backward pass.
    Ideal for:
    - Quick sanity checks on what a model "sees"
    - Models without clean gradient paths (quantized, pruned, compiled)
    - Batch processing large datasets for analysis
    - YOLO and other detection models where gradients break

    Usage:
        cam = EigenCAM(model)
        heatmap = cam(image)  # No gradient computation needed
    """

    requires_grad = False  # No backward pass needed

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
            # Forward pass only — no gradients needed
            with torch.no_grad():
                activations_and_grads(input_tensor)

            if not activations_and_grads.activations:
                raise RuntimeError(
                    f"Failed to capture activations from "
                    f"{type(self.target_layer).__name__}. "
                    f"Ensure the target layer is on the model's forward path."
                )

            # Activations are already (B, C, H, W) — reshaped by hook if needed
            activations = activations_and_grads.activations[0].numpy()
            B, C, h, w = activations.shape

            # Follow pytorch-grad-cam's get_2d_projection exactly:
            # 1. Reshape to (C, h*w) then transpose to (h*w, C)
            reshaped = activations[0].reshape(C, h * w)
            reshaped = reshaped.transpose()  # (h*w, C)

            # 2. Center — critical for correct SVD results
            reshaped = reshaped - reshaped.mean(axis=0)

            # 3. Handle NaN values
            reshaped[np.isnan(reshaped)] = 0

            # 4. SVD
            try:
                U, S, VT = np.linalg.svd(reshaped, full_matrices=True)
            except np.linalg.LinAlgError:
                # Fallback: use mean activation if SVD fails
                return activations[0].mean(axis=0)

            # 5. Project onto first principal component. The projection is
            # signed; orient it by correlation with the activation-energy
            # map (mean |activation| per position), which highlights the
            # regions the network actually responds to. For classification
            # models BaseExplainer._resolve_polarity then double-checks the
            # sign with a masked class-score test; for detection models
            # (no class logits) this correlation IS the polarity decision.
            projection = reshaped @ VT[0, :]  # (h*w,)

            energy = np.abs(activations[0]).mean(axis=0).reshape(-1)  # (h*w,)
            energy = energy - energy.mean()
            if float(projection @ energy) < 0:
                projection = -projection

            # 6. Reshape back to spatial dimensions
            cam = projection.reshape(h, w)

            return cam

        finally:
            activations_and_grads.release()
