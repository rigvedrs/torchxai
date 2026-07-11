"""
Attention Rollout — Aggregate attention across transformer layers.

For Vision Transformers, each layer's attention matrix shows how tokens
attend to each other. Attention Rollout multiplies these matrices across
layers to approximate the total attention flow from input patches to
the final representation.

This is a gradient-free method specific to transformer architectures.

Reference:
    Abnar & Zuidema, "Quantifying Attention Flow in Transformers", ACL 2020.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from torchxai.methods.base import BaseExplainer, patches_to_spatial
from torchxai.models.registry import find_attention_layers


class AttentionRollout(BaseExplainer):
    """Attention Rollout for Vision Transformers.

    Computes the total attention flow from each input patch to the CLS token
    by recursively multiplying attention matrices across all layers.

    Works with any model that has multi-head attention layers (ViT, DeiT,
    Swin, DINO, CLIP vision encoder).

    Usage:
        rollout = AttentionRollout(model)
        heatmap = rollout(image)

    Args:
        model: Vision Transformer model.
        head_fusion: How to fuse attention across heads.
            Options: "max" (default), "mean", "min". "max" is the most
            robust across ViT sizes in our verification runs.
        discard_ratio: Fraction of lowest-attention values to discard
            at each layer. Rollout without discarding is dominated by
            diffuse residual attention and looks washed out; 0.9 is the
            widely-used default (vit-explain, Gildenblat's examples).
    """

    requires_grad = False  # No backward pass needed

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        head_fusion: str = "max",
        discard_ratio: float = 0.9,
    ) -> None:
        super().__init__(model, target_layer, device)
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self._attention_maps: list[torch.Tensor] = []

    def _compute_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int],
    ) -> np.ndarray:
        self._attention_maps = []
        handles = []
        unfused = []  # (module, previous fused_attn value)

        # Find and hook all attention layers
        attn_layers = find_attention_layers(self.model)

        if not attn_layers:
            # Graceful fallback for models without accessible attention
            import warnings

            from torchxai.methods.eigencam import EigenCAM

            warnings.warn(
                f"No attention layers found in {type(self.model).__name__}. "
                f"Falling back to EigenCAM (gradient-free). "
                f"For best results with transformers, ensure the model "
                f"exposes attention weights in forward hooks.",
                UserWarning,
                stacklevel=3,
            )
            fallback = EigenCAM(self.model, self.target_layer, self.device)
            return fallback._compute_cam(input_tensor, None)

        def make_tuple_hook(storage: list):
            # Attention modules that return (output, attn_weights)
            def hook_fn(module, input, output):
                if isinstance(output, tuple) and len(output) >= 2:
                    attn = output[1]
                    if attn is not None and isinstance(attn, torch.Tensor):
                        storage.append(attn.detach())

            return hook_fn

        def make_drop_hook(storage: list):
            # timm-style modules: the softmaxed (B, heads, N, N) attention
            # matrix passes through attn_drop right after softmax.
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor) and output.ndim == 4:
                    if output.shape[-1] == output.shape[-2]:
                        storage.append(output.detach())

            return hook_fn

        for layer in attn_layers:
            # timm >= 0.9 uses F.scaled_dot_product_attention by default,
            # which never materializes the attention matrix. Temporarily
            # switch to the unfused path so attn_drop sees the weights.
            attn_drop = getattr(layer, "attn_drop", None)
            if attn_drop is not None and isinstance(attn_drop, nn.Module):
                if hasattr(layer, "fused_attn"):
                    unfused.append((layer, layer.fused_attn))
                    layer.fused_attn = False
                handles.append(
                    attn_drop.register_forward_hook(make_drop_hook(self._attention_maps))
                )
            else:
                handles.append(layer.register_forward_hook(make_tuple_hook(self._attention_maps)))

        try:
            # Forward pass — collect attention maps
            with torch.no_grad():
                self.model(input_tensor)

            batch = input_tensor.shape[0]
            # Keep only global attention maps: batch dim must match the
            # input batch (window attention runs at B*num_windows and
            # cannot be rolled out globally) and all maps must share the
            # same token count.
            maps = [m for m in self._attention_maps if m.shape[0] == batch]
            if maps:
                from collections import Counter

                n_common = Counter(m.shape[-1] for m in maps).most_common(1)[0][0]
                maps = [m for m in maps if m.shape[-1] == n_common]
            self._attention_maps = maps

            if not self._attention_maps:
                # Hooks didn't capture usable attention — fall back
                import warnings

                from torchxai.methods.eigencam import EigenCAM

                warnings.warn(
                    f"Could not capture global attention weights from "
                    f"{type(self.model).__name__} (windowed or fused "
                    f"attention). Falling back to EigenCAM.",
                    UserWarning,
                    stacklevel=3,
                )
                fallback = EigenCAM(self.model, self.target_layer, self.device)
                return fallback._compute_cam(input_tensor, None)

            # Rollout computation
            return self._compute_rollout()

        finally:
            for h in handles:
                h.remove()
            for layer, prev in unfused:
                layer.fused_attn = prev
            self._attention_maps = []  # Free memory

    def _compute_rollout(self) -> np.ndarray:
        """Compute attention rollout from collected attention maps."""
        result = None

        for attn in self._attention_maps:
            # attn shape: (B, num_heads, N, N)
            if attn.ndim == 4:
                if self.head_fusion == "mean":
                    attn = attn.mean(dim=1)
                elif self.head_fusion == "max":
                    attn = attn.max(dim=1).values
                elif self.head_fusion == "min":
                    attn = attn.min(dim=1).values
                else:
                    attn = attn.mean(dim=1)

            # Add identity matrix (residual connection)
            I = torch.eye(attn.shape[-1], device=attn.device).unsqueeze(0)
            attn = 0.5 * attn + 0.5 * I

            # Normalize rows to sum to 1
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

            # Optional: discard lowest attention
            if self.discard_ratio > 0:
                flat = attn.reshape(attn.shape[0], -1)
                threshold = torch.quantile(flat, self.discard_ratio, dim=-1, keepdim=True)
                flat_mask = (flat >= threshold).float()
                attn = attn * flat_mask.view(attn.shape)
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

            # Chain multiply
            if result is None:
                result = attn
            else:
                result = torch.bmm(attn, result)

        if result is None:
            raise RuntimeError("No attention maps collected during forward pass.")

        # Extract CLS token row (how each patch contributes to CLS),
        # skipping all prefix tokens (CLS + any register tokens).
        # result shape: (B, N, N)
        from torchxai.methods.base import _find_model_attr, _infer_prefix_and_grid

        n_tokens = result.shape[-1]
        num_prefix = _find_model_attr(self.model, "num_prefix_tokens")
        prefix, _, _ = _infer_prefix_and_grid(n_tokens, num_prefix)
        prefix = max(prefix, 1)  # CLS row must exist for rollout
        cls_attention = result[0, 0, prefix:]

        return patches_to_spatial(cls_attention)
