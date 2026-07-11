"""
RISE — Randomized Input Sampling for Explanation.

RISE explains black-box models by generating a large number of random
binary masks, applying each mask to the input image, forwarding the
masked image through the model, and computing a weighted sum of the
masks where each mask's weight equals the model's confidence for the
target class. This produces a saliency map purely from input/output
pairs with no gradient information required.

For speed, masks are generated at a low resolution (default 7×7) and
bilinearly upsampled to the input size, which introduces smooth
spatial edges that reduce aliasing artifacts. Inference is performed
in mini-batches to amortise GPU overhead.

Reference:
    Petsiuk et al., "RISE: Randomized Input Sampling for Explanation
    of Black-box Models", BMVC 2018.
    https://arxiv.org/abs/1806.07421
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from torchxai.methods.base import BaseExplainer
from torchxai.utils.image import normalize_heatmap, preprocess_image


class RISE(BaseExplainer):
    """Randomized Input Sampling for Explanation.

    Generates saliency maps by averaging randomly masked inputs weighted
    by the model's confidence for the target class. No gradients needed —
    fully black-box compatible.

    Args:
        model: The PyTorch model to explain.
        target_layer: Not used; accepted for API consistency.
        device: Computation device. Auto-detected if None.
        n_masks: Total number of random masks to sample. More masks give
            smoother, more reliable saliency maps (default: 4000).
        mask_resolution: Spatial resolution of the binary mask before
            upsampling. Lower values produce smoother masks; higher values
            retain finer structure (default: 7).
        batch_size: Number of masked inputs to forward in a single batch.
            Adjust to fit GPU memory (default: 64).
        p1: Probability that each cell in the low-resolution mask is 1
            (i.e., unmasked). Defaults to 0.5.

    Usage:
        rise = RISE(model)
        heatmap = rise(image)  # numpy array (H, W) in [0, 1]

        rise = RISE(model, n_masks=1000, mask_resolution=8)
        heatmap = rise(image, target_class=243)
    """

    requires_grad = False

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        n_masks: int = 4000,
        mask_resolution: int = 7,
        batch_size: int = 64,
        p1: float = 0.5,
    ) -> None:
        super().__init__(model, target_layer, device)
        self.n_masks = n_masks
        self.mask_resolution = mask_resolution
        self.batch_size = batch_size
        self.p1 = p1

    # ── Override __call__ — RISE is fully input-level, no layer needed ────

    def __call__(
        self,
        image: Union[torch.Tensor, Image.Image, np.ndarray],
        target_class: Optional[int] = None,
        image_size: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """Generate a RISE saliency map.

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

        # ── Determine target class ─────────────────────────────────────────
        with torch.no_grad():
            out = self.model(input_tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.ndim > 2:
                out = out.mean(dim=tuple(range(2, out.ndim)))
            if target_class is None:
                target_class = out.argmax(dim=-1).item()

        # ── Accumulate weighted mask sum ───────────────────────────────────
        # sal_sum[i, j] = Σ_k  score_k × mask_k[i, j]
        sal_sum = torch.zeros(H, W, device=self.device)
        # Normalisation: Σ_k mask_k[i, j]  (≈ n_masks × p1 for large N)
        mask_sum = torch.zeros(H, W, device=self.device)

        masks_done = 0
        while masks_done < self.n_masks:
            current_batch = min(self.batch_size, self.n_masks - masks_done)

            # ── Generate low-resolution binary masks ──────────────────────
            m_r = self.mask_resolution
            low_res = (
                torch.rand(current_batch, 1, m_r, m_r, device=self.device) < self.p1
            ).float()  # (B, 1, m_r, m_r)

            # Upsample to slightly larger size, then random-crop to (H, W)
            # This eliminates the edge bias from bilinear interpolation
            # (following the original RISE paper, Section 3.1)
            cell_h = int(np.ceil(H / m_r))
            cell_w = int(np.ceil(W / m_r))
            up_h = (m_r + 1) * cell_h
            up_w = (m_r + 1) * cell_w

            masks = F.interpolate(
                low_res, size=(up_h, up_w), mode="bilinear", align_corners=False
            )  # (B, 1, up_h, up_w)

            # Random crop to (H, W) — different offset for each mask
            max_y = up_h - H
            max_x = up_w - W
            offset_y = torch.randint(0, max_y + 1, (current_batch,), device=self.device)
            offset_x = torch.randint(0, max_x + 1, (current_batch,), device=self.device)

            cropped_masks = torch.zeros(current_batch, 1, H, W, device=self.device)
            for b in range(current_batch):
                oy = offset_y[b].item()
                ox = offset_x[b].item()
                cropped_masks[b] = masks[b, :, oy : oy + H, ox : ox + W]
            masks = cropped_masks  # (B, 1, H, W)

            # Apply each mask to the input
            masked_inputs = input_tensor * masks  # (B, 3, H, W)

            # Forward pass
            with torch.no_grad():
                batch_out = self.model(masked_inputs)
            if isinstance(batch_out, (tuple, list)):
                batch_out = batch_out[0]
            if batch_out.ndim > 2:
                batch_out = batch_out.mean(dim=tuple(range(2, batch_out.ndim)))

            # Confidence scores for target class: (B,)
            confidences = batch_out[:, target_class]  # (B,)

            # Weighted accumulation
            # masks: (B, 1, H, W) → (B, H, W)
            masks_2d = masks.squeeze(1)  # (B, H, W)
            # confidences: (B,) → (B, 1, 1)
            weights = confidences.view(-1, 1, 1)  # (B, 1, 1)

            sal_sum += (weights * masks_2d).sum(dim=0)  # (H, W)
            mask_sum += masks_2d.sum(dim=0)  # (H, W)

            masks_done += current_batch

        # ── Normalise by total mask coverage ──────────────────────────────
        # Avoid division by zero in degenerate regions
        mask_sum = torch.clamp(mask_sum, min=1e-8)
        saliency = sal_sum / mask_sum  # (H, W)

        heatmap = saliency.cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        return normalize_heatmap(heatmap)

    # ── _compute_cam is required by the abstract base ──────────────────────

    def _compute_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int],
    ) -> np.ndarray:
        """Not used — RISE overrides ``__call__`` directly.

        This stub satisfies the abstract base class contract.
        """
        raise NotImplementedError("RISE overrides __call__ and does not use _compute_cam.")
