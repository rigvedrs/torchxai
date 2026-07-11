"""
SmoothGrad — Removing noise by adding noise.

SmoothGrad improves vanilla gradient saliency maps by averaging gradients
computed on N noisy copies of the input. Gaussian noise is added to each
copy; averaging over the samples suppresses noise in individual gradient
maps and sharpens the attribution signal.

This method operates on input-level gradients (pixel saliency), not on an
intermediate activation layer. The ``__call__`` method is overridden to
bypass the layer-centric pipeline in ``BaseExplainer``.

Reference:
    Smilkov et al., "SmoothGrad: removing noise by adding noise", 2017.
    https://arxiv.org/abs/1706.03825
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from PIL import Image
import torch

from torchxai.methods.base import BaseExplainer
from torchxai.utils.image import normalize_heatmap, preprocess_image


class SmoothGrad(BaseExplainer):
    """SmoothGrad pixel-level saliency via averaged noisy gradients.

    Rather than using a CAM from an intermediate layer, SmoothGrad produces
    a pixel-wise attribution map by averaging ``∂score/∂x`` over N noisy
    copies of the input.

    Args:
        model: The PyTorch model to explain.
        target_layer: Not used for saliency; accepted for API consistency.
        device: Computation device. Auto-detected if None.
        n_samples: Number of noisy copies to average. More samples = smoother
            maps but proportionally longer runtime (default: 50).
        noise_level: Standard deviation of the Gaussian noise relative to the
            input value range (default: 0.1). The actual std is computed as
            ``noise_level × (x_max − x_min)``.

    Usage:
        sg = SmoothGrad(model)
        heatmap = sg(image)  # numpy array (H, W) in [0, 1]

        sg = SmoothGrad(model, n_samples=25, noise_level=0.15)
        heatmap = sg(image, target_class=243)
    """

    requires_grad = True

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        n_samples: int = 50,
        noise_level: float = 0.1,
    ) -> None:
        super().__init__(model, target_layer, device)
        self.n_samples = n_samples
        self.noise_level = noise_level

    # ── Override __call__ — SmoothGrad does not use _compute_cam ──────────

    def __call__(
        self,
        image: Union[torch.Tensor, Image.Image, np.ndarray],
        target_class: Optional[int] = None,
        image_size: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """Generate a SmoothGrad saliency map.

        Args:
            image: Input image as a tensor (1,3,H,W) or (3,H,W),
                PIL Image, or numpy array (H,W,3).
            target_class: Class index to explain. None = predicted class.
            image_size: Resize target if image is not already a tensor.

        Returns:
            Numpy array (H, W) with saliency values in [0, 1].
        """
        if image_size is None:
            image_size = self._default_input_size()

        # Preprocess to (1, 3, H, W)
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image.unsqueeze(0)
            input_tensor = image.to(self.device).float()
        else:
            input_tensor = preprocess_image(image, size=image_size, device=self.device)

        H, W = input_tensor.shape[2], input_tensor.shape[3]

        # Determine noise std from input range
        x_min = input_tensor.min().item()
        x_max = input_tensor.max().item()
        noise_std = self.noise_level * (x_max - x_min)

        # ── Determine target class from a clean forward pass ──────────────
        with torch.no_grad():
            out = self.model(input_tensor.detach())
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.ndim > 2:
                out = out.mean(dim=tuple(range(2, out.ndim)))
            if target_class is None:
                target_class = out.argmax(dim=-1).item()

        # ── Accumulate gradients over N noisy samples ─────────────────────
        grad_sum = torch.zeros_like(input_tensor)

        for _ in range(self.n_samples):
            noise = torch.randn_like(input_tensor) * noise_std
            noisy_input = (input_tensor + noise).detach().requires_grad_(True)

            output = self.model(noisy_input)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim > 2:
                output = output.mean(dim=tuple(range(2, output.ndim)))

            score = self._get_class_score(output, target_class)
            self.model.zero_grad()
            score.backward()

            if noisy_input.grad is not None:
                grad_sum += noisy_input.grad.detach().abs()

        # Average gradient across samples and channels
        avg_grad = grad_sum / self.n_samples  # (1, 3, H, W)
        saliency = avg_grad.mean(dim=1).squeeze()  # (H, W)

        heatmap = saliency.cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        return normalize_heatmap(heatmap)

    # ── _compute_cam is required by the abstract base ──────────────────────

    def _compute_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int],
    ) -> np.ndarray:
        """Not used — SmoothGrad overrides ``__call__`` directly.

        This stub satisfies the abstract base class contract.
        """
        raise NotImplementedError("SmoothGrad overrides __call__ and does not use _compute_cam.")
