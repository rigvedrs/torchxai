"""
Hook utilities for extracting activations and gradients from PyTorch models.

These hooks are the foundation of all CAM-based explainability methods.
They register forward/backward hooks on target layers to capture intermediate
activations and their gradients without modifying the model.

Key design decisions:
- Forward hooks store activations WITH the computation graph by default
  (needed for gradient-based methods like GradCAM).
- The `detach` parameter controls whether to break the graph.
- All hooks support context manager protocol for safe cleanup.
- Hooks handle tuple outputs gracefully (common in transformers).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ActivationHook:
    """Captures forward activations from a target layer.

    Usage:
        hook = ActivationHook(model.layer4)
        output = model(input)
        activations = hook.activation  # Tensor
        hook.remove()

    Or as a context manager:
        with ActivationHook(model.layer4) as hook:
            output = model(input)
            print(hook.activation.shape)
    """

    def __init__(self, layer: nn.Module, detach: bool = True) -> None:
        self.activation: Optional[torch.Tensor] = None
        self._detach = detach
        self._handle = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output,
    ) -> None:
        # Handle tuple/list outputs (transformers, multi-scale backbones)
        if isinstance(output, (tuple, list)):
            out = output[0]
        elif isinstance(output, dict):
            out = next(iter(output.values()))
        else:
            out = output

        if isinstance(out, torch.Tensor):
            self.activation = out.detach() if self._detach else out

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self) -> ActivationHook:
        return self

    def __exit__(self, *args) -> None:
        self.remove()


class GradientHook:
    """Captures gradients flowing back through a target layer.

    Usage:
        hook = GradientHook(model.layer4)
        output = model(input)
        output.backward()
        gradients = hook.gradient  # Tensor
        hook.remove()
    """

    def __init__(self, layer: nn.Module) -> None:
        self.gradient: Optional[torch.Tensor] = None
        self._handle = layer.register_full_backward_hook(self._hook_fn)

    def _hook_fn(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        if grad_output[0] is not None:
            self.gradient = grad_output[0].detach()

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self) -> GradientHook:
        return self

    def __exit__(self, *args) -> None:
        self.remove()


class MultiHook:
    """Captures both activations and gradients from a target layer.

    This is the primary hook used by gradient-based CAM methods. The forward
    hook stores activations WITHOUT detaching (preserving the computation
    graph so backward hooks can capture gradients).

    Usage:
        hook = MultiHook(model.layer4)
        output = model(input)
        loss.backward()
        print(hook.activation.shape, hook.gradient.shape)
        hook.remove()
    """

    def __init__(self, layer: nn.Module) -> None:
        self.activation: Optional[torch.Tensor] = None
        self.gradient: Optional[torch.Tensor] = None
        self._fwd_handle = layer.register_forward_hook(self._fwd_hook)
        self._bwd_handle = layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output,
    ) -> None:
        # Handle tuple/list outputs (transformers return (hidden_states,
        # attn_weights); multi-scale backbones return lists of feature maps)
        if isinstance(output, (tuple, list)):
            out = output[0]
        elif isinstance(output, dict):
            out = next(iter(output.values()))
        else:
            out = output

        if isinstance(out, torch.Tensor):
            # Keep graph alive for backward — detach only a reference for storage
            self.activation = out

    def _bwd_hook(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        if grad_output[0] is not None:
            self.gradient = grad_output[0].detach()

    def remove(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()
        # Release references to break potential cycles
        self.activation = None
        self.gradient = None

    def __enter__(self) -> MultiHook:
        return self

    def __exit__(self, *args) -> None:
        self.remove()


class ActivationsAndGradients:
    """Captures activations and gradients from target layers.

    Uses forward hooks for BOTH activations and gradients (via tensor.register_hook).
    This avoids register_full_backward_hook which has issues with VGG/DenseNet
    inplace operations.

    Optionally applies reshape_transform inside the hook for ViT/Swin compatibility.

    Args:
        model: The PyTorch model.
        target_layers: List of layers to hook.
        reshape_transform: Optional callable applied to activations/gradients
            before storing (used for ViT/Swin token-to-spatial reshaping).
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: list[nn.Module],
        reshape_transform=None,
    ) -> None:
        self.model = model
        self.gradients: list[torch.Tensor] = []
        self.activations: list[torch.Tensor] = []
        self.reshape_transform = reshape_transform
        self.handles: list = []

        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self._save_activation))
            self.handles.append(target_layer.register_forward_hook(self._save_gradient))

    @staticmethod
    def _first_tensor(output):
        """Recursively unwrap nested tuples/lists/NestedTensors to the first
        torch.Tensor (multi-scale backbones return lists of lists)."""
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)):
            for item in output:
                found = ActivationsAndGradients._first_tensor(item)
                if found is not None:
                    return found
            return None
        # e.g. DETR-style NestedTensor with .tensors
        tensors = getattr(output, "tensors", None)
        if isinstance(tensors, torch.Tensor):
            return tensors
        return None

    def _save_activation(
        self,
        module: nn.Module,
        input: tuple,
        output,
    ) -> None:
        activation = self._first_tensor(output)
        if activation is None:
            return
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def _save_gradient(
        self,
        module: nn.Module,
        input: tuple,
        output,
    ) -> None:
        output = self._first_tensor(output)
        if output is None or not output.requires_grad:
            return

        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self) -> None:
        """Remove all hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self) -> ActivationsAndGradients:
        return self

    def __exit__(self, *args) -> None:
        self.release()


class AttentionHook:
    """Captures attention weights from transformer attention layers.

    Designed for Vision Transformers where we need the attention
    probability matrix (after softmax) from each attention head.

    Handles multiple output formats:
    - (output, attention_weights) — standard PyTorch MHA
    - output only — stores None (graceful fallback)

    Usage:
        hook = AttentionHook(model.blocks[0].attn)
        output = model(input)
        attn_weights = hook.attention  # (B, num_heads, N, N) or None
        hook.remove()
    """

    def __init__(self, layer: nn.Module) -> None:
        self.attention: Optional[torch.Tensor] = None
        self._handle = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output,
    ) -> None:
        # Many attention implementations return (output, attention_weights)
        if isinstance(output, tuple) and len(output) >= 2:
            attn = output[1]
            if attn is not None and isinstance(attn, torch.Tensor):
                self.attention = attn.detach()

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self) -> AttentionHook:
        return self

    def __exit__(self, *args) -> None:
        self.remove()
