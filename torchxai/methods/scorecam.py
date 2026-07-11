"""
ScoreCAM — Score-Weighted Visual Explanations for CNNs.

A perturbation-based CAM method that uses forward passes rather than
gradients to determine channel importance. Each activation channel is
upsampled and used as a mask; the masked input is forwarded through
the network, and the resulting class confidence increase becomes the
channel weight. This produces more faithful explanations than
gradient-based methods at the cost of extra forward passes.

To keep inference time practical, only the top-K channels (ranked by
activation magnitude) are used by default.

Reference:
    Wang et al., "Score-CAM: Score-Weighted Visual Explanations for
    Convolutional Neural Networks", CVPR 2020.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from torchxai.methods.base import BaseExplainer
from torchxai.utils.hooks import ActivationHook


class ScoreCAM(BaseExplainer):
    """Score-Weighted Class Activation Mapping.

    Perturbation-based explainability: each activation channel is used
    as a soft mask over the input, and the resulting class score increase
    determines its importance weight. Requires no gradient computation.

    Args:
        model: The PyTorch model to explain.
        target_layer: Layer whose activations define the channels.
            Auto-detected if None.
        device: Computation device. Auto-detected if None.
        top_k: Number of highest-magnitude channels to evaluate. Using
            fewer channels dramatically reduces runtime while retaining
            quality. Set to None to use all channels.
        batch_size: Number of masked inputs to process per forward pass.

    Usage:
        cam = ScoreCAM(model)
        heatmap = cam(image)  # numpy array (H, W) in [0, 1]

        # Limit to top 16 channels for speed:
        cam = ScoreCAM(model, top_k=16)
        heatmap = cam(image, target_class=243)
    """

    requires_grad = False

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        top_k: int = 32,
        batch_size: int = 16,
    ) -> None:
        super().__init__(model, target_layer, device)
        self.top_k = top_k
        self.batch_size = batch_size

    def _compute_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int],
    ) -> np.ndarray:
        """Compute ScoreCAM saliency map.

        Args:
            input_tensor: Preprocessed input tensor of shape (1, 3, H, W).
            target_class: Target class index, or None for predicted class.

        Returns:
            Raw heatmap as numpy array (H, W).
        """
        H, W = input_tensor.shape[2], input_tensor.shape[3]

        # ── Step 1: Capture activations from the target layer ─────────────
        with ActivationHook(self.target_layer, detach=True) as hook:
            with torch.no_grad():
                output = self.model(input_tensor)

            activations = hook.activation  # (1, C, h, w) or (1, N, C)

        if activations is None:
            raise RuntimeError(
                f"Failed to capture activations from "
                f"{type(self.target_layer).__name__}. "
                f"Ensure the target layer lies on the forward path."
            )

        # Handle transformer token layout: (B, N, C) -> (B, C, h, w)
        if activations.ndim == 3:
            from torchxai.methods.base import reshape_transformer_tokens

            activations = reshape_transformer_tokens(activations)

        activations = activations.detach()  # (1, C, h, w)
        # Pre-activation target layers (e.g. DenseNet's denseblock4) yield
        # signed maps, but the network only propagates the positive side
        # through its nonlinearity — clamp so masks match the paper's
        # post-ReLU assumption. No-op for post-activation layers.
        if (activations < 0).any():
            activations = activations.clamp(min=0)
        num_channels = activations.shape[1]

        # ── Step 2: Determine target class from baseline forward pass ──────
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim > 2:
            output = output.mean(dim=tuple(range(2, output.ndim)))

        with torch.no_grad():
            baseline_score: torch.Tensor = self._get_class_score(output, target_class)
            if target_class is None:
                target_class = output.argmax(dim=-1).item()

        baseline_score_val = baseline_score.item()

        # ── Step 3: Select top-K channels by L1 norm ──────────────────────
        if self.top_k is not None and self.top_k < num_channels:
            # Rank channels by their spatial L1 norm
            norms = activations[0].abs().sum(dim=(1, 2))  # (C,)
            top_indices = norms.argsort(descending=True)[: self.top_k].tolist()
        else:
            top_indices = list(range(num_channels))

        # ── Step 4: Build masked inputs and score each channel ─────────────
        scores = torch.zeros(len(top_indices), device=self.device)

        # Upsample each selected activation channel to input resolution,
        # normalize to [0, 1], and apply as a soft mask over the input.
        # The normalized channels are reused for the final weighted sum:
        # target layers on some architectures (VGG's last Conv2d, DenseNet's
        # denseblock4) sit BEFORE the ReLU, so raw activations are mostly
        # negative and a raw-activation sum would be wiped out by the final
        # ReLU. The normalized channel keeps the same spatial ordering.
        # Process in mini-batches to avoid excessive GPU memory use.
        masked_inputs: list[torch.Tensor] = []
        norm_channels: list[torch.Tensor] = []
        for idx in top_indices:
            act_map = activations[:, idx : idx + 1, :, :]  # (1, 1, h, w)
            # Upsample to input size
            act_up = F.interpolate(
                act_map, size=(H, W), mode="bilinear", align_corners=False
            )  # (1, 1, H, W)
            # Normalize channel to [0, 1]
            act_min = act_up.min()
            act_max = act_up.max()
            if act_max - act_min > 1e-8:
                act_norm = (act_up - act_min) / (act_max - act_min)
            else:
                act_norm = torch.zeros_like(act_up)
            norm_channels.append(act_norm)
            # Apply mask: element-wise multiply across all input channels
            masked = input_tensor * act_norm  # (1, 3, H, W)
            masked_inputs.append(masked)

        # Batch forward passes
        for batch_start in range(0, len(masked_inputs), self.batch_size):
            batch = torch.cat(
                masked_inputs[batch_start : batch_start + self.batch_size],
                dim=0,
            )  # (B, 3, H, W)
            with torch.no_grad():
                batch_out = self.model(batch)
            if isinstance(batch_out, (tuple, list)):
                batch_out = batch_out[0]
            if batch_out.ndim > 2:
                batch_out = batch_out.mean(dim=tuple(range(2, batch_out.ndim)))

            for i, out in enumerate(batch_out):
                global_i = batch_start + i
                scores[global_i] = out[target_class]

        # ── Step 5: Compute score increases (weight = score - baseline) ────
        weights = scores - baseline_score_val  # (K,)
        # Apply softmax to stabilise the weighting
        weights = torch.softmax(weights, dim=0)

        # ── Step 6: Weighted sum of normalized activation channels ─────────
        cam = torch.zeros(H, W, device=self.device)
        for i, act_norm in enumerate(norm_channels):
            cam += weights[i] * act_norm.squeeze()

        return cam.cpu().numpy()
