"""
Integrated Gradients — Axiomatic Attribution for Deep Networks.

Integrated Gradients satisfies two fundamental axioms (sensitivity and
implementation invariance) by computing attributions as the path integral
of gradients between a baseline input and the actual input. In practice,
this is approximated via a Riemann sum: N linearly-interpolated images
are created between the baseline and input, gradients are computed for
each, and their average is multiplied by (input − baseline).

This method operates at the pixel level and overrides ``__call__``
rather than using the layer-based ``_compute_cam`` pipeline.

Reference:
    Sundararajan et al., "Axiomatic Attribution for Deep Networks",
    ICML 2017. https://arxiv.org/abs/1703.01365
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
from PIL import Image
import torch

from torchxai.methods.base import BaseExplainer
from torchxai.utils.image import normalize_heatmap, preprocess_image


class IntegratedGradients(BaseExplainer):
    """Integrated Gradients pixel-level attribution.

    Computes attributions as the Riemann approximation of the path
    integral of gradients from a reference (baseline) to the input:

        IG(x) = (x − x') × (1/N) Σ ∂f(x' + k/N × (x − x')) / ∂x

    Args:
        model: The PyTorch model to explain.
        target_layer: Not used; accepted for API consistency.
        device: Computation device. Auto-detected if None.
        n_steps: Number of interpolation steps for the Riemann sum
            approximation. More steps = higher fidelity but slower
            (default: 50).
        baseline: Type of baseline reference image. Options:
            - ``"black"`` — all-zeros tensor (default).
            - ``"white"`` — all-ones tensor (before normalisation).
            - ``"blur"`` — Gaussian-blurred version of the input.

    Usage:
        ig = IntegratedGradients(model)
        heatmap = ig(image)  # numpy array (H, W) in [0, 1]

        ig = IntegratedGradients(model, n_steps=100, baseline="blur")
        heatmap = ig(image, target_class=243)
    """

    requires_grad = True

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        n_steps: int = 50,
        baseline: Literal["black", "white", "blur"] = "black",
    ) -> None:
        super().__init__(model, target_layer, device)
        self.n_steps = n_steps
        self.baseline = baseline

    # ── Override __call__ — IG operates on input pixels, not layer CAMs ───

    def __call__(
        self,
        image: Union[torch.Tensor, Image.Image, np.ndarray],
        target_class: Optional[int] = None,
        image_size: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """Generate an Integrated Gradients attribution map.

        Args:
            image: Input image as a tensor (1,3,H,W) or (3,H,W),
                PIL Image, or numpy array (H,W,3).
            target_class: Class index to explain. None = predicted class.
            image_size: Resize target if image is not already a tensor.

        Returns:
            Numpy array (H, W) with attribution values in [0, 1].
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

        # ── Build baseline ─────────────────────────────────────────────────
        baseline_tensor = self._make_baseline(input_tensor)

        # ── Determine target class from a clean forward pass ──────────────
        with torch.no_grad():
            out = self.model(input_tensor.detach())
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.ndim > 2:
                out = out.mean(dim=tuple(range(2, out.ndim)))
            if target_class is None:
                target_class = out.argmax(dim=-1).item()

        # ── Riemann sum: accumulate gradients at N interpolated inputs ─────
        grad_sum = torch.zeros_like(input_tensor)
        delta = input_tensor - baseline_tensor  # (1, 3, H, W)

        for step in range(self.n_steps):
            alpha = step / self.n_steps
            interp = (baseline_tensor + alpha * delta).detach().requires_grad_(True)

            output = self.model(interp)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim > 2:
                output = output.mean(dim=tuple(range(2, output.ndim)))

            score = self._get_class_score(output, target_class)
            self.model.zero_grad()
            score.backward()

            if interp.grad is not None:
                grad_sum += interp.grad.detach()

        # Average gradients → multiply by (input − baseline)
        avg_grads = grad_sum / self.n_steps  # (1, 3, H, W)
        integrated = avg_grads * delta  # (1, 3, H, W)

        # Reduce to (H, W): absolute value, sum over channels
        saliency = integrated.abs().sum(dim=1).squeeze()  # (H, W)

        heatmap = saliency.cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        return normalize_heatmap(heatmap)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _make_baseline(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Construct the baseline reference image.

        Args:
            input_tensor: The real input tensor (1, 3, H, W).

        Returns:
            Baseline tensor of the same shape and device.
        """
        if self.baseline == "black":
            # For ImageNet-normalized inputs, a true black baseline
            # is the normalized version of [0,0,0], not raw zeros.
            # mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
            # normalized_black = (0 - mean) / std ≈ [-2.12, -2.04, -1.80]
            mean = torch.tensor([0.485, 0.456, 0.406], device=input_tensor.device)
            std = torch.tensor([0.229, 0.224, 0.225], device=input_tensor.device)
            black_normalized = (-mean / std).view(1, 3, 1, 1)
            return black_normalized.expand_as(input_tensor)
        elif self.baseline == "white":
            mean = torch.tensor([0.485, 0.456, 0.406], device=input_tensor.device)
            std = torch.tensor([0.229, 0.224, 0.225], device=input_tensor.device)
            white_normalized = ((1.0 - mean) / std).view(1, 3, 1, 1)
            return white_normalized.expand_as(input_tensor)
        elif self.baseline == "blur":
            return self._gaussian_blur(input_tensor)
        else:
            raise ValueError(
                f"Unknown baseline '{self.baseline}'. Expected one of: 'black', 'white', 'blur'."
            )

    @staticmethod
    def _gaussian_blur(
        tensor: torch.Tensor, kernel_size: int = 51, sigma: float = 10.0
    ) -> torch.Tensor:
        """Apply Gaussian blur to create a smooth baseline.

        Args:
            tensor: Input tensor (1, 3, H, W).
            kernel_size: Size of the Gaussian kernel (must be odd).
            sigma: Standard deviation of the Gaussian.

        Returns:
            Blurred tensor with the same shape.
        """
        import torch.nn.functional as F

        # Build 2-D Gaussian kernel
        ax = torch.arange(kernel_size, device=tensor.device, dtype=torch.float32)
        ax = ax - (kernel_size - 1) / 2.0
        gauss_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        kernel_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
        kernel_4d = kernel_2d.expand(tensor.shape[1], 1, kernel_size, kernel_size)

        padding = kernel_size // 2
        blurred = F.conv2d(
            tensor,
            kernel_4d,
            padding=padding,
            groups=tensor.shape[1],
        )
        return blurred.detach()

    # ── _compute_cam is required by the abstract base ──────────────────────

    def _compute_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int],
    ) -> np.ndarray:
        """Not used — IntegratedGradients overrides ``__call__`` directly.

        This stub satisfies the abstract base class contract.
        """
        raise NotImplementedError(
            "IntegratedGradients overrides __call__ and does not use _compute_cam."
        )
