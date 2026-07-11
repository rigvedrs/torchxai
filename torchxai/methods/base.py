"""
Base class for all explainability methods.

Every method in torchxai inherits from BaseExplainer, which provides:
- Automatic model architecture detection
- Target layer resolution
- Common preprocessing/postprocessing pipeline
- Device management
- Transformer token reshaping (shared utility)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchxai.models.registry import ArchType, detect_architecture, resolve_target_layer
from torchxai.utils.image import normalize_heatmap, preprocess_image

# ── Reshape Transforms for Transformer Architectures ─────────────────


def vit_reshape_transform(tensor, height=14, width=14):
    """Reshape ViT/DeiT/EVA outputs: (B, 1+H*W, C) -> (B, C, H, W).

    Strips the CLS token (first token) and reshapes remaining patch tokens
    into a spatial grid. This matches pytorch-grad-cam's approach.

    Args:
        tensor: Activation tensor of shape (B, 1+H*W, C).
        height: Spatial grid height (default: 14 for patch16 @ 224px).
        width: Spatial grid width (default: 14 for patch16 @ 224px).

    Returns:
        Reshaped tensor (B, C, H, W).
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def swin_reshape_transform(tensor, height=7, width=7):
    """Reshape Swin outputs: (B, H*W, C) -> (B, C, H, W).

    Swin does NOT have a CLS token — all tokens are spatial.

    Args:
        tensor: Activation tensor of shape (B, H*W, C).
        height: Spatial grid height (default: 7 for Swin-T last stage @ 224px).
        width: Spatial grid width (default: 7 for Swin-T last stage @ 224px).

    Returns:
        Reshaped tensor (B, C, H, W).
    """
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def _infer_vit_grid_size(model):
    """Infer the patch grid size for a ViT model.

    Looks for patch_embed to determine patch_size, then computes
    grid_size = img_size / patch_size.

    Returns:
        (height, width) tuple, defaults to (14, 14) if detection fails.
    """
    # timm ViT: model.patch_embed.patch_size, model.patch_embed.img_size
    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is not None:
        patch_size = getattr(patch_embed, "patch_size", None)
        img_size = getattr(patch_embed, "img_size", None)
        if patch_size is not None and img_size is not None:
            if isinstance(patch_size, (tuple, list)):
                ph, pw = patch_size[0], patch_size[1]
            else:
                ph = pw = int(patch_size)
            if isinstance(img_size, (tuple, list)):
                ih, iw = img_size[0], img_size[1]
            else:
                ih = iw = int(img_size)
            return ih // ph, iw // pw

    # Fallback: try num_patches
    if patch_embed is not None:
        num_patches = getattr(patch_embed, "num_patches", None)
        if num_patches is not None:
            h = int(np.sqrt(num_patches))
            if h * h == num_patches:
                return h, h

    return 14, 14  # default for ViT-B/16 @ 224


def _infer_swin_grid_size(model):
    """Infer the spatial size of the last Swin stage output.

    For Swin-T at 224px: 224 / 32 = 7, so last stage is 7x7.

    Returns:
        (height, width) tuple, defaults to (7, 7) if detection fails.
    """
    # timm Swin: model.patch_embed.img_size and model has 4 stages with 2x downsample each
    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is not None:
        img_size = getattr(patch_embed, "img_size", None)
        patch_size = getattr(patch_embed, "patch_size", None)
        if img_size is not None and patch_size is not None:
            if isinstance(img_size, (tuple, list)):
                ih, iw = img_size[0], img_size[1]
            else:
                ih = iw = int(img_size)
            if isinstance(patch_size, (tuple, list)):
                ph, pw = patch_size[0], patch_size[1]
            else:
                ph = pw = int(patch_size)
            # Swin: patch_embed does patch_size downsample, then 3 more 2x downsamples
            # Total: patch_size * 2^3 = patch_size * 8
            # So grid_size = img_size / (patch_size * 8)
            # For Swin-T: 224 / (4 * 8) = 7
            layers = getattr(model, "layers", None)
            if layers is not None:
                num_downsample_stages = 0
                for layer in layers:
                    ds = getattr(layer, "downsample", None)
                    # Only count actual PatchMerging, not Identity placeholders
                    if ds is not None and type(ds).__name__ != "Identity":
                        num_downsample_stages += 1
                total_downsample = ph * (2**num_downsample_stages)
                return ih // total_downsample, iw // total_downsample

    return 7, 7  # default for Swin-T @ 224


def _find_model_attr(model, attr):
    """Search a model (and common wrapper attributes) for an attribute.

    Handles wrapped backbones like ``DINOv2Classifier(backbone=..., head=...)``
    where the ViT metadata lives on ``model.backbone`` rather than ``model``.
    """
    candidates = [model]
    for name in ("backbone", "model", "visual", "vision_model", "encoder"):
        sub = getattr(model, name, None)
        if isinstance(sub, nn.Module):
            candidates.append(sub)
    for cand in candidates:
        value = getattr(cand, attr, None)
        if value is not None:
            return value
    return None


def _infer_prefix_and_grid(num_tokens, num_prefix_tokens=None):
    """Infer (num_prefix_tokens, height, width) from a token count.

    If ``num_prefix_tokens`` is known (timm exposes it), use it directly.
    Otherwise try common prefix counts (CLS only, none, CLS + register
    tokens) until the remaining token count forms a square grid.
    """
    if num_prefix_tokens is not None:
        spatial = num_tokens - num_prefix_tokens
        g = int(np.sqrt(spatial))
        if g * g == spatial:
            return num_prefix_tokens, g, g

    for prefix in (1, 0, 5, 2, 4, 8):  # CLS / none / CLS+registers / distilled
        spatial = num_tokens - prefix
        if spatial <= 0:
            continue
        g = int(np.sqrt(spatial))
        if g * g == spatial:
            return prefix, g, g

    # Fallback: largest square that fits, treat the remainder as prefix
    g = int(np.sqrt(num_tokens))
    return num_tokens - g * g, g, g


def _build_reshape_transform(model, arch_type):
    """Build the appropriate reshape_transform for a model architecture.

    The returned transform is DYNAMIC: it derives the spatial grid from the
    actual token count at call time instead of static model metadata. This
    keeps it correct for wrapped backbones (no patch_embed on the wrapper),
    non-default input sizes (EVA-02 @ 448), and register tokens (DINOv2-reg).

    Args:
        model: The PyTorch model.
        arch_type: Detected ArchType.

    Returns:
        A reshape_transform callable, or None for CNN architectures.
    """
    if arch_type in (ArchType.VIT, ArchType.DINO, ArchType.CLIP):
        num_prefix = _find_model_attr(model, "num_prefix_tokens")

        def transform(tensor):
            if tensor.ndim == 4:
                return tensor  # already spatial (B, C, H, W)
            B, N, C = tensor.shape
            prefix, h, w = _infer_prefix_and_grid(N, num_prefix)
            result = tensor[:, prefix:, :].reshape(B, h, w, C)
            return result.transpose(2, 3).transpose(1, 2)

        return transform

    elif arch_type == ArchType.SWIN:
        static_hw = _infer_swin_grid_size(model)

        def transform(tensor):
            if tensor.ndim == 4:
                # timm >= 0.9 Swin blocks use NHWC layout: (B, H, W, C)
                if tensor.shape[-1] > tensor.shape[1]:
                    return tensor.permute(0, 3, 1, 2)
                return tensor
            B, N, C = tensor.shape
            g = int(np.sqrt(N))
            h, w = (g, g) if g * g == N else static_hw
            result = tensor.reshape(B, h, w, C)
            return result.transpose(2, 3).transpose(1, 2)

        return transform

    # CNN / YOLO / DETR / GENERIC: no reshape needed
    return None


class BaseExplainer(ABC):
    """Abstract base class for all torchxai explainability methods.

    Subclasses must implement `_compute_cam()` which produces the raw
    saliency map from activations/gradients/attention weights.

    Args:
        model: The PyTorch model to explain.
        target_layer: The layer to compute explanations from. If None,
            torchxai auto-detects the best layer based on model architecture.
        device: Device to run computations on. Auto-detected if None.

    Example:
        >>> cam = GradCAM(model)
        >>> heatmap = cam(image)
        >>> heatmap.shape
        (224, 224)
    """

    # Whether this method needs gradients (subclasses override)
    requires_grad: bool = True

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[Union[nn.Module, str]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.model.eval()

        # Auto-detect device
        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Detect architecture type
        self.arch_type = detect_architecture(model)

        # Resolve target layer
        if target_layer is None:
            self.target_layer = resolve_target_layer(model, self.arch_type)
        elif isinstance(target_layer, str):
            self.target_layer = self._get_layer_by_name(target_layer)
        else:
            self.target_layer = target_layer

        # Build reshape transform for ViT/Swin architectures
        self.reshape_transform = _build_reshape_transform(model, self.arch_type)

    def _get_layer_by_name(self, name: str) -> nn.Module:
        """Resolve a layer by its dot-separated name.

        Args:
            name: Dot-separated layer name, e.g. "layer4.1.conv2"

        Returns:
            The nn.Module at the specified path.

        Raises:
            AttributeError: If any part of the path doesn't exist,
                with a helpful error message listing available layers.
        """
        module = self.model
        for part in name.split("."):
            try:
                # Try as attribute first, then as integer index
                if hasattr(module, part):
                    module = getattr(module, part)
                elif part.isdigit():
                    module = list(module.children())[int(part)]
                else:
                    available = [n for n, _ in module.named_children()]
                    raise AttributeError(
                        f"Layer '{part}' not found in {type(module).__name__}. "
                        f"Available: {available}"
                    )
            except (AttributeError, IndexError):
                available = [n for n, _ in module.named_children()]
                raise AttributeError(
                    f"Could not resolve layer path '{name}' — failed at '{part}'. "
                    f"Available children: {available}"
                )
        return module

    def _default_input_size(self) -> tuple[int, int]:
        """Resolve the model's expected input size.

        Checks patch_embed.img_size (reflects runtime overrides), then the
        timm pretrained_cfg. Falls back to (224, 224). This keeps models
        like EVA-02 @ 448 working without manual configuration.
        """
        patch_embed = _find_model_attr(self.model, "patch_embed")
        if patch_embed is not None:
            img_size = getattr(patch_embed, "img_size", None)
            if img_size is not None:
                if isinstance(img_size, (tuple, list)):
                    return int(img_size[0]), int(img_size[1])
                return int(img_size), int(img_size)

        cfg = _find_model_attr(self.model, "pretrained_cfg")
        if isinstance(cfg, dict):
            input_size = cfg.get("input_size")
            if input_size is not None and len(input_size) == 3:
                return int(input_size[1]), int(input_size[2])

        return 224, 224

    def __call__(
        self,
        image: Union[torch.Tensor, Image.Image, np.ndarray],
        target_class: Optional[int] = None,
        image_size: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """Generate a saliency map for the given image.

        Args:
            image: Input image as tensor (1,3,H,W) or (3,H,W),
                   PIL Image, or numpy array (H,W,3).
            target_class: Class index to explain. If None, uses the predicted class.
            image_size: Size to resize input to if not a tensor.
                If None, resolved from the model (falls back to 224x224).

        Returns:
            Numpy array of shape (H, W) with values in [0, 1] representing
            the saliency map.
        """
        if image_size is None:
            image_size = self._default_input_size()

        # Preprocess if needed
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image.unsqueeze(0)
            input_tensor = image.to(self.device).float()
        else:
            input_tensor = preprocess_image(image, size=image_size, device=self.device)

        # Only require gradients for methods that need them
        if self.requires_grad:
            input_tensor = input_tensor.detach().requires_grad_(True)

        # Compute the raw saliency map
        with torch.set_grad_enabled(self.requires_grad):
            raw_cam = self._compute_cam(input_tensor, target_class)

        # Resize to input spatial dimensions
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        if raw_cam.shape[0] != h or raw_cam.shape[1] != w:
            cam_tensor = torch.from_numpy(raw_cam).float().unsqueeze(0).unsqueeze(0)
            cam_tensor = F.interpolate(
                cam_tensor, size=(h, w), mode="bilinear", align_corners=False
            )
            raw_cam = cam_tensor.squeeze().numpy()

        # Signed CAMs (EigenCAM projections, GradCAM on architectures with
        # normalization between the target layer and the classifier) have an
        # ambiguous polarity. Resolve it faithfully before the ReLU discards
        # one side.
        raw_cam = self._resolve_polarity(input_tensor, raw_cam, target_class)

        # All-negative maps (e.g. pre-ReLU target layers) carry their signal
        # in the ordering, not the sign — shift instead of zeroing them out.
        if raw_cam.max() <= 0 and raw_cam.min() < 0:
            raw_cam = raw_cam - raw_cam.min()

        # Apply ReLU — we only care about positive influence
        raw_cam = np.maximum(raw_cam, 0)

        # Normalize to [0, 1]
        return normalize_heatmap(raw_cam)

    def _resolve_polarity(
        self,
        input_tensor: torch.Tensor,
        raw_cam: np.ndarray,
        target_class: Optional[int],
    ) -> np.ndarray:
        """Pick the faithful sign for a signed saliency map.

        SVD projections (EigenCAM) and gradient-weighted sums on models with
        LayerNorm/GRN between the target layer and classifier are only
        defined up to sign — the wrong sign highlights the background. We
        disambiguate empirically with an insertion/deletion test: for each
        polarity, score(input masked TO that region) minus score(input with
        that region DELETED). The faithful side keeps the class score high
        when kept and collapses it when removed; a keep-only test is biased
        toward whichever side covers more area. Costs one batched forward
        pass of 4 masked inputs, and only runs when the map actually has a
        significant negative side.
        """
        pos = np.maximum(raw_cam, 0)
        neg = np.maximum(-raw_cam, 0)

        # Already (almost) single-signed — nothing to resolve.
        if neg.max() <= 1e-8 or pos.max() <= 1e-8:
            return raw_cam
        if neg.max() < 0.1 * pos.max():
            return raw_cam
        if pos.max() < 0.1 * neg.max():
            return -raw_cam

        # Detection models don't produce class logits, so a masked class
        # score is meaningless — keep the method's own orientation.
        if self.arch_type in (ArchType.YOLO, ArchType.DETR):
            return raw_cam

        try:
            x = input_tensor.detach()
            device = x.device

            def class_score(logits, cls):
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                if logits.ndim > 2:
                    logits = logits.mean(dim=tuple(range(2, logits.ndim)))
                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                return logits[:, cls]

            with torch.no_grad():
                base_logits = self.model(x)
                if isinstance(base_logits, (tuple, list)):
                    base_logits = base_logits[0]
                if base_logits.ndim > 2:
                    base_logits = base_logits.mean(dim=tuple(range(2, base_logits.ndim)))
                cls = (
                    target_class
                    if target_class is not None
                    else int(base_logits.argmax(dim=-1).item())
                )

                masks = []
                for side in (pos, neg):
                    m = side / (side.max() + 1e-8)
                    masks.append(torch.from_numpy(m).float().to(device).unsqueeze(0).unsqueeze(0))
                masked = torch.cat(
                    [
                        x * masks[0],  # keep positive side
                        x * masks[1],  # keep negative side
                        x * (1 - masks[0]),  # delete positive side
                        x * (1 - masks[1]),  # delete negative side
                    ],
                    dim=0,
                )
                scores = class_score(self.model(masked), cls)

            keep_pos, keep_neg, del_pos, del_neg = (s.item() for s in scores)
            pos_evidence = keep_pos - del_pos
            neg_evidence = keep_neg - del_neg
            return raw_cam if pos_evidence >= neg_evidence else -raw_cam
        except Exception:
            # Fall back to the magnitude heuristic (Seg-Eigen-CAM Eq. 13).
            if abs(float(raw_cam.min())) > abs(float(raw_cam.max())):
                return -raw_cam
            return raw_cam

    @abstractmethod
    def _compute_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int],
    ) -> np.ndarray:
        """Compute the raw class activation map.

        Must be implemented by each method.

        Args:
            input_tensor: Preprocessed input tensor (1, 3, H, W).
            target_class: Target class index or None for predicted class.

        Returns:
            Raw heatmap as numpy array (h, w) — not necessarily normalized.
        """
        ...

    def _get_class_score(
        self,
        output: torch.Tensor,
        target_class: Optional[int],
    ) -> torch.Tensor:
        """Extract the score for a target class from model output.

        Handles both classification (logit vector) and detection outputs.

        Args:
            output: Model output tensor.
            target_class: Class index. If None, uses argmax.

        Returns:
            Scalar tensor — the score for the target class.
        """
        if output.ndim == 1:
            output = output.unsqueeze(0)

        if target_class is None:
            target_class = output.argmax(dim=-1).item()

        return output[0, target_class]

    def __repr__(self) -> str:
        layer_name = type(self.target_layer).__name__
        return (
            f"{type(self).__name__}("
            f"arch={self.arch_type.name}, "
            f"target_layer={layer_name}, "
            f"device={self.device})"
        )


# ── Shared Transformer Utilities ──────────────────────────────────────────


def reshape_transformer_tokens(
    activations: torch.Tensor,
    gradients: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Reshape transformer (B, N, C) outputs to (B, C, h, w) spatial format.

    Automatically detects and removes the class token, then reshapes
    the remaining patch tokens into a spatial grid. Works for:
    - ViT (14x14 = 196 patches + CLS)
    - DeiT (same as ViT)
    - Swin (7x7, 14x14 etc.)
    - CLIP ViT (16x16 = 256 patches + CLS)

    Args:
        activations: Tensor of shape (B, N, C).
        gradients: Optional tensor of same shape.

    Returns:
        Reshaped activations (B, C, h, w) if no gradients,
        or tuple of (activations, gradients) both reshaped.
    """
    B, N, C = activations.shape

    # Detect and remove class token
    spatial_tokens = N
    has_cls_token = False

    # Common patch counts: check N-1 against known grid sizes
    known_grids = [7, 8, 12, 14, 16, 24, 32, 28, 48, 56]
    for grid_size in known_grids:
        if N == grid_size * grid_size + 1:
            has_cls_token = True
            spatial_tokens = N - 1
            break
        elif N == grid_size * grid_size:
            break

    if has_cls_token:
        activations = activations[:, 1:, :]
        if gradients is not None:
            gradients = gradients[:, 1:, :]
        spatial_tokens = N - 1

    # Find best spatial grid dimensions
    h = int(np.sqrt(spatial_tokens))
    w = spatial_tokens // h
    while h * w != spatial_tokens and h > 1:
        h -= 1
        w = spatial_tokens // h

    if h * w != spatial_tokens:
        # Non-square fallback — pad to nearest square
        h = int(np.ceil(np.sqrt(spatial_tokens)))
        w = h
        pad = h * w - spatial_tokens
        if pad > 0:
            activations = torch.nn.functional.pad(activations, (0, 0, 0, pad))
            if gradients is not None:
                gradients = torch.nn.functional.pad(gradients, (0, 0, 0, pad))

    # Reshape: (B, N, C) -> (B, C, h, w)
    activations = activations.reshape(B, h, w, C).permute(0, 3, 1, 2)

    if gradients is not None:
        gradients = gradients.reshape(B, h, w, C).permute(0, 3, 1, 2)
        return activations, gradients

    return activations


def patches_to_spatial(
    cls_attention: torch.Tensor,
) -> np.ndarray:
    """Convert a 1D patch attention vector to 2D spatial map.

    Used by AttentionRollout and TransformerAttribution to convert
    the CLS token's attention over patches into a spatial heatmap.

    Args:
        cls_attention: 1D tensor of shape (num_patches,).

    Returns:
        2D numpy array of shape (h, w).
    """
    num_patches = cls_attention.shape[0]
    h = int(np.sqrt(num_patches))
    w = num_patches // h

    while h * w != num_patches and h > 1:
        h -= 1
        w = num_patches // h

    if h * w != num_patches:
        h = w = int(np.ceil(np.sqrt(num_patches)))
        pad = h * w - num_patches
        cls_attention = torch.nn.functional.pad(cls_attention, (0, pad))

    return cls_attention.reshape(h, w).cpu().numpy()
