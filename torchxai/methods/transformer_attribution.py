"""
Transformer Attribution — Gradient-weighted attention attribution.

Combines attention weights with gradient information for class-specific
attribution in Vision Transformers. Unlike Attention Rollout (which is
class-agnostic), this method produces class-discriminative explanations.

Inspired by:
    Chefer et al., "Transformer Interpretability Beyond Attention
    Visualization", CVPR 2021.
"""

from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
import torch
import torch.nn as nn

from torchxai.methods.base import BaseExplainer, patches_to_spatial
from torchxai.models.registry import find_attention_layers


class TransformerAttribution(BaseExplainer):
    """Class-specific transformer attribution using gradient-attention fusion.

    Produces class-discriminative saliency maps by weighting attention
    matrices with the gradient of the target class. This answers "which
    patches matter for THIS specific class?" rather than just "where does
    the model attend generally?"

    Usage:
        attr = TransformerAttribution(model)
        heatmap = attr(image, target_class=243)  # "bull mastiff"
    """

    requires_grad = True  # Needs gradients for class-specific attribution

    def _compute_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int],
    ) -> np.ndarray:
        attention_maps: list[torch.Tensor] = []
        gradient_maps: list[torch.Tensor] = []
        handles: list[torch.utils.hooks.RemovableHook] = []

        attn_layers = find_attention_layers(self.model)

        if not attn_layers:
            # Graceful fallback: use Attention Rollout (which itself falls
            # back to EigenCAM if no attention layers exist)
            warnings.warn(
                f"No attention layers found in {type(self.model).__name__}. "
                f"TransformerAttribution requires a transformer-based model. "
                f"Falling back to AttentionRollout → EigenCAM.",
                UserWarning,
                stacklevel=3,
            )
            from torchxai.methods.attention_rollout import AttentionRollout

            fallback = AttentionRollout(self.model, device=self.device)
            return fallback._compute_cam(input_tensor, target_class)

        def make_fwd_hook(attn_storage: list):
            def hook_fn(module, input, output):
                # Transformer attention layers often return (output, attn_weights)
                if isinstance(output, tuple) and len(output) >= 2:
                    attn = output[1]
                    if attn is not None and isinstance(attn, torch.Tensor):
                        attn.retain_grad()
                        attn_storage.append(attn)

            return hook_fn

        def make_drop_hook(attn_storage: list):
            # timm-style modules: the softmaxed (B, heads, N, N) attention
            # matrix passes through attn_drop right after softmax.
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor) and output.ndim == 4:
                    if output.shape[-1] == output.shape[-2]:
                        if output.requires_grad:
                            output.retain_grad()
                        attn_storage.append(output)

            return hook_fn

        unfused = []  # (module, previous fused_attn value)
        for layer in attn_layers:
            # timm >= 0.9 uses fused SDPA by default, which never
            # materializes attention weights — switch to the unfused path.
            attn_drop = getattr(layer, "attn_drop", None)
            if attn_drop is not None and isinstance(attn_drop, nn.Module):
                if hasattr(layer, "fused_attn"):
                    unfused.append((layer, layer.fused_attn))
                    layer.fused_attn = False
                handles.append(attn_drop.register_forward_hook(make_drop_hook(attention_maps)))
            else:
                handles.append(layer.register_forward_hook(make_fwd_hook(attention_maps)))

        try:
            # Forward pass
            output = self.model(input_tensor)

            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim > 2:
                output = output.mean(dim=tuple(range(2, output.ndim)))

            score = self._get_class_score(output, target_class)

            # Backward pass to get attention gradients
            self.model.zero_grad()
            score.backward(retain_graph=False)

            # Keep only global attention maps (window attention runs at
            # B*num_windows and cannot be attributed globally), all with
            # the same token count.
            batch = input_tensor.shape[0]
            attention_maps = [a for a in attention_maps if a.shape[0] == batch]
            if attention_maps:
                from collections import Counter

                n_common = Counter(a.shape[-1] for a in attention_maps).most_common(1)[0][0]
                attention_maps = [a for a in attention_maps if a.shape[-1] == n_common]

            # Collect gradients
            for attn in attention_maps:
                if attn.grad is not None:
                    gradient_maps.append(attn.grad.detach())
                else:
                    gradient_maps.append(torch.ones_like(attn))

            if not attention_maps:
                # Hooks didn't capture attention — fall back gracefully
                warnings.warn(
                    "Attention hooks didn't capture any weights. Falling back to AttentionRollout.",
                    UserWarning,
                    stacklevel=3,
                )
                from torchxai.methods.attention_rollout import AttentionRollout

                fallback = AttentionRollout(self.model, device=self.device)
                return fallback._compute_cam(input_tensor, target_class)

            # Compute gradient-weighted attention rollout
            result = None

            for attn, grad in zip(attention_maps, gradient_maps):
                attn = attn.detach()

                # Fuse heads: mean of gradient-weighted attention
                if attn.ndim == 4:
                    weighted_attn = (attn * grad.clamp(min=0)).mean(dim=1)
                else:
                    weighted_attn = attn

                # Add residual identity connection
                I = torch.eye(weighted_attn.shape[-1], device=weighted_attn.device).unsqueeze(0)
                weighted_attn = 0.5 * weighted_attn + 0.5 * I

                # Normalize rows
                weighted_attn = weighted_attn / (weighted_attn.sum(dim=-1, keepdim=True) + 1e-8)

                if result is None:
                    result = weighted_attn
                else:
                    result = torch.bmm(weighted_attn, result)

            if result is None:
                raise RuntimeError(
                    "Failed to compute transformer attribution — "
                    "no valid attention-gradient pairs collected."
                )

            # Extract CLS token attention over patches → spatial heatmap,
            # skipping all prefix tokens (CLS + any register tokens).
            from torchxai.methods.base import _find_model_attr, _infer_prefix_and_grid

            n_tokens = result.shape[-1]
            num_prefix = _find_model_attr(self.model, "num_prefix_tokens")
            prefix, _, _ = _infer_prefix_and_grid(n_tokens, num_prefix)
            prefix = max(prefix, 1)  # CLS row must exist
            cls_attention = result[0, 0, prefix:]
            return patches_to_spatial(cls_attention)

        finally:
            for h in handles:
                h.remove()
            for layer, prev in unfused:
                layer.fused_attn = prev
