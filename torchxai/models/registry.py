"""
Model architecture detection and target layer resolution.

torchxai auto-detects the model architecture and selects the optimal
target layer for explainability. This enables the one-line API:
    explain(model, image)  # works for ResNet, ViT, CLIP, YOLO, etc.

Target layer selection follows the same conventions as pytorch-grad-cam
(the most widely-used CAM library), ensuring consistent and correct
saliency maps across all architectures.

Supported architecture families:
- CNN classifiers (ResNet, VGG, EfficientNet, DenseNet, ConvNeXt, etc.)
- Vision Transformers (ViT, DeiT, Swin, BEiT, EVA, MaxViT)
- Foundation models (DINO, DINOv2, SAM, MAE)
- Object detectors (YOLO, DETR, Faster-RCNN)
- Vision-Language models (CLIP, SigLIP)
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional

import torch.nn as nn


class ArchType(Enum):
    """Detected model architecture type."""

    CNN = auto()
    VIT = auto()
    SWIN = auto()
    CLIP = auto()
    DETR = auto()
    YOLO = auto()
    DINO = auto()
    GENERIC = auto()


def detect_architecture(model: nn.Module) -> ArchType:
    """Auto-detect the architecture family of a PyTorch model.

    Uses a combination of module name inspection and structural analysis
    to identify the model type without requiring explicit user input.

    Args:
        model: Any PyTorch model.

    Returns:
        ArchType indicating the detected architecture family.
    """
    model_name = type(model).__name__.lower()
    module_names = {name.lower() for name, _ in model.named_modules()}
    module_str = " ".join(module_names)

    # CLIP detection
    if "clip" in model_name or "visual" in model_name:
        if any("visual" in n for n in module_names):
            return ArchType.CLIP
    if hasattr(model, "visual") or hasattr(model, "encode_image"):
        return ArchType.CLIP

    # YOLO detection. Ultralytics module NAMES are opaque ("model.22"), so
    # look at module TYPE names for the Detect/DFL head markers.
    if "yolo" in model_name or hasattr(model, "detect"):
        return ArchType.YOLO
    module_type_names = {type(m).__name__.lower() for m in model.modules()}
    if "detect" in module_type_names or "dfl" in module_type_names:
        return ArchType.YOLO
    if any("detect" in n and "c2f" in module_str for n in module_names):
        return ArchType.YOLO

    # DETR detection
    if "detr" in model_name or "deformabledetr" in model_name:
        return ArchType.DETR
    if hasattr(model, "transformer") and hasattr(model, "class_embed"):
        return ArchType.DETR

    # DINO / DINOv2
    if "dino" in model_name:
        return ArchType.DINO

    # Swin Transformer
    if "swin" in model_name:
        return ArchType.SWIN
    if any("swinblock" in n or "swin" in n for n in module_names):
        return ArchType.SWIN

    # MaxViT / CoAtNet — hybrid models with both conv and attention
    # These should be treated as CNN (they have stages with conv blocks)
    if "maxvit" in model_name or "coatnet" in model_name or "maxxvit" in model_name:
        return ArchType.CNN

    # Vision Transformer (ViT, DeiT, BEiT, MAE, EVA)
    if _is_vit(model, model_name, module_names):
        return ArchType.VIT

    # Default CNN
    if _has_conv_layers(model):
        return ArchType.CNN

    return ArchType.GENERIC


def _is_vit(model: nn.Module, model_name: str, module_names: set) -> bool:
    """Check if model is a Vision Transformer variant."""
    vit_indicators = ["vit", "deit", "beit", "mae", "visiontransformer", "eva"]
    if any(ind in model_name for ind in vit_indicators):
        return True
    # Check for transformer block structure
    if hasattr(model, "blocks") or hasattr(model, "encoder"):
        # Check for attention layers inside
        for name, module in model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                if hasattr(module, "qkv") or hasattr(module, "in_proj_weight"):
                    return True
    if any("blocks" in n and "attn" in " ".join(module_names) for n in module_names):
        return True
    return False


def _has_conv_layers(model: nn.Module) -> bool:
    """Check if model contains convolutional layers."""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            return True
    return False


def resolve_target_layer(model: nn.Module, arch_type: ArchType) -> nn.Module:
    """Automatically select the best target layer for explainability.

    Follows the same conventions as pytorch-grad-cam:
    - For CNNs: last convolutional block before the classifier
    - For ViTs: last transformer block's norm layer (norm1)
    - For Swin: last block's norm2
    - For CLIP: visual encoder's last layer
    - For YOLO: second-to-last layer (before Detect head)

    Args:
        model: The model.
        arch_type: Detected architecture type.

    Returns:
        The nn.Module to hook for activations/gradients.
    """
    if arch_type == ArchType.VIT:
        return _resolve_vit_layer(model)
    elif arch_type == ArchType.SWIN:
        return _resolve_swin_layer(model)
    elif arch_type == ArchType.CLIP:
        return _resolve_clip_layer(model)
    elif arch_type == ArchType.DINO:
        return _resolve_vit_layer(model)
    elif arch_type == ArchType.DETR:
        return _resolve_detr_layer(model)
    elif arch_type == ArchType.YOLO:
        return _resolve_yolo_layer(model)
    elif arch_type == ArchType.CNN:
        return _resolve_cnn_layer(model)
    else:
        return _resolve_generic_layer(model)


def _resolve_cnn_layer(model: nn.Module) -> nn.Module:
    """Find the best target layer for a CNN.

    Strategy (in priority order):
    1. Architecture-specific known paths
    2. Last Conv2d layer in the model

    Following pytorch-grad-cam conventions:
    - ResNet: model.layer4[-1]
    - VGG: model.features (last Conv2d, typically features[-3] or features[28])
    - DenseNet: model.features[-1]
    - EfficientNet: model.features[-1]
    - MobileNet: model.features[-1]
    - ConvNeXt: model.features[-1] (torchvision) or model.stages[-1].blocks[-1] (timm)
    """
    model_name = type(model).__name__.lower()

    # ResNet: model.layer4[-1] — the last BasicBlock or Bottleneck
    if hasattr(model, "layer4"):
        children = list(model.layer4.children())
        if children:
            return children[-1]

    # ConvNeXt V2 / timm ConvNeXt / MaxViT: model.stages[-1].blocks[-1]
    # For MaxViT, hook the FULL block output (B, C, H, W) — matching the
    # pytorch-grad-cam reference. Hooking only the MbConv sub-block places
    # attention layers between the hook and the classifier, which breaks
    # the CAM weighting.
    if hasattr(model, "stages"):
        stages = list(model.stages)
        if stages:
            last_stage = stages[-1]
            if hasattr(last_stage, "blocks"):
                blocks = list(last_stage.blocks)
                if blocks:
                    return blocks[-1]
            return last_stage

    # RegNet: model.trunk_output[-1] or last AnyStage
    if "regnet" in model_name:
        if hasattr(model, "trunk_output"):
            children = list(model.trunk_output.children())
            if children:
                return children[-1]

    # VGG: hook the ReLU right after the last Conv2d. Post-activation maps
    # are non-negative (matching the assumptions of activation-based
    # methods like ScoreCAM/GradCAM++) and keep the 14x14 resolution that
    # the 7x7 MaxPool output would lose.
    if "vgg" in model_name and hasattr(model, "features"):
        # Disable inplace ReLU to fix backward hook compatibility
        for module in model.features.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        last_conv_relu = None
        prev_was_conv = False
        for module in model.features.children():
            if isinstance(module, nn.ReLU) and prev_was_conv:
                last_conv_relu = module
            prev_was_conv = isinstance(module, nn.Conv2d)
        if last_conv_relu is not None:
            return last_conv_relu

    # DenseNet: model.features.denseblock4 or model.features[-1]
    if "densenet" in model_name and hasattr(model, "features"):
        # DenseNet features end with norm5 after denseblock4
        if hasattr(model.features, "denseblock4"):
            return model.features.denseblock4
        children = list(model.features.children())
        if children:
            return children[-1]

    # EfficientNet / MobileNet / generic with model.features
    if hasattr(model, "features"):
        children = list(model.features.children())
        if children:
            return children[-1]

    # timm EfficientNet family (EfficientNet, MobileNetV3/V4, GhostNet, ...):
    # these expose conv_stem + blocks. The conv_head placement differs:
    #   - EfficientNet: conv_head runs BEFORE global pooling -> spatial (B,C,7,7)
    #   - MobileNetV3/V4, GhostNet: conv_head runs AFTER global pooling -> (B,C,1,1)
    # A 1x1 post-pool map upsamples to a uniform (useless) heatmap, so for the
    # post-pool family we hook the last spatial block instead.
    if hasattr(model, "conv_stem") and hasattr(model, "blocks"):
        if type(model).__name__ == "EfficientNet" and hasattr(model, "conv_head"):
            return model.conv_head
        blocks = list(model.blocks)
        if blocks:
            return blocks[-1]

    # RepVGG / other timm models: find last Conv2d with a spatial output.
    # Never return a conv that runs after global pooling (1x1 output) —
    # probe candidate layers with a dummy forward to check.
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is not None:
        spatial = _find_last_spatial_conv(model)
        return spatial if spatial is not None else last_conv

    raise RuntimeError(
        "Could not auto-detect target layer for CNN. Please specify target_layer manually."
    )


def _find_last_spatial_conv(model: nn.Module) -> Optional[nn.Module]:
    """Find the last Conv2d whose output keeps a spatial extent (H, W > 1).

    Runs one dummy forward pass with hooks on every Conv2d and returns the
    last-executed conv with a spatial output. Returns None if the probe
    fails (e.g. the model needs a non-224 input size).
    """
    import torch

    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        return None

    executed: list[nn.Module] = []

    def make_hook(mod):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor) and out.ndim == 4:
                if out.shape[2] > 1 and out.shape[3] > 1:
                    executed.append(mod)

        return hook

    handles = [m.register_forward_hook(make_hook(m)) for m in convs]
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, 3, 224, 224))
    except Exception:
        return None
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    return executed[-1] if executed else None


def _resolve_vit_layer(model: nn.Module) -> nn.Module:
    """Find the last transformer block's norm layer in ViT.

    Following pytorch-grad-cam convention: model.blocks[-1].norm1
    This provides spatial token activations before attention mixing.
    """
    # timm ViT / EVA / DeiT: model.blocks[-1].norm1
    if hasattr(model, "blocks"):
        blocks = (
            list(model.blocks.children())
            if hasattr(model.blocks, "children")
            else list(model.blocks)
        )
        if blocks:
            last_block = blocks[-1]
            if hasattr(last_block, "norm1"):
                return last_block.norm1
            return last_block

    # DINOv2 wrapper: check for backbone.blocks
    if hasattr(model, "backbone"):
        backbone = model.backbone
        if hasattr(backbone, "blocks"):
            blocks = (
                list(backbone.blocks.children())
                if hasattr(backbone.blocks, "children")
                else list(backbone.blocks)
            )
            if blocks:
                last_block = blocks[-1]
                if hasattr(last_block, "norm1"):
                    return last_block.norm1
                return last_block

    # HuggingFace ViT: model.encoder.layer[-1].layernorm_before
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        layers = list(model.encoder.layer)
        if layers:
            last = layers[-1]
            if hasattr(last, "layernorm_before"):
                return last.layernorm_before
            return last

    # torchvision ViT
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        layers = list(model.encoder.layers)
        if layers:
            return layers[-1]

    return _resolve_generic_layer(model)


def _resolve_swin_layer(model: nn.Module) -> nn.Module:
    """Find the last Swin transformer block's norm2.

    Following pytorch-grad-cam convention: model.layers[-1].blocks[-1].norm2
    The Swin output at norm2 is (B, H*W, C) — a full spatial sequence
    (NOT windowed), which can be reshaped to (B, C, H, W).
    """
    if hasattr(model, "layers"):
        layers = list(model.layers)
        if layers:
            last_stage = layers[-1]
            blocks = list(last_stage.blocks) if hasattr(last_stage, "blocks") else [last_stage]
            if blocks:
                last_block = blocks[-1]
                # Prefer norm2 (post-attention) for Swin
                if hasattr(last_block, "norm2"):
                    return last_block.norm2
                if hasattr(last_block, "norm1"):
                    return last_block.norm1
                return last_block

    return _resolve_generic_layer(model)


def _resolve_clip_layer(model: nn.Module) -> nn.Module:
    """Find the visual encoder's last layer in CLIP."""
    visual = None
    if hasattr(model, "visual"):
        visual = model.visual
    elif hasattr(model, "vision_model"):
        visual = model.vision_model

    if visual is not None:
        # OpenAI CLIP: visual.transformer.resblocks[-1]
        if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
            blocks = list(visual.transformer.resblocks)
            if blocks:
                return blocks[-1]

        # HuggingFace CLIP: visual.encoder.layers[-1]
        if hasattr(visual, "encoder") and hasattr(visual.encoder, "layers"):
            layers = list(visual.encoder.layers)
            if layers:
                return layers[-1]

        # timm CLIP: visual.blocks[-1]
        if hasattr(visual, "blocks"):
            blocks = list(visual.blocks)
            if blocks:
                return blocks[-1]

    return _resolve_generic_layer(model)


def _resolve_detr_layer(model: nn.Module) -> nn.Module:
    """Find the encoder's last layer in DETR."""
    if hasattr(model, "transformer"):
        transformer = model.transformer
        if hasattr(transformer, "encoder") and hasattr(transformer.encoder, "layers"):
            layers = list(transformer.encoder.layers)
            if layers:
                return layers[-1]
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        layers = list(model.encoder.layers)
        if layers:
            return layers[-1]

    return _resolve_generic_layer(model)


def _resolve_yolo_layer(model: nn.Module) -> nn.Module:
    """Find the backbone's last feature layer in YOLO.

    Hook model.model[-2] (second-to-last), NOT the Detect head.
    The Detect head has tiny feature maps and breaks gradients.
    Use EigenCAM (gradient-free) for YOLO.
    """
    # Descend nested wrappers: ultralytics YOLO -> DetectionModel -> Sequential
    inner = model
    for _ in range(3):
        if not hasattr(inner, "model"):
            break
        inner = inner.model
        layers = list(inner.children()) if isinstance(inner, nn.Module) else []
        if len(layers) >= 2:
            # Second-to-last layer (before Detect head)
            for layer in reversed(layers[:-1]):
                if "detect" not in type(layer).__name__.lower():
                    return layer

    # Fallback: find last Conv2d that has spatial feature maps
    target = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Skip 1x1 convs in the detection head (small spatial dims)
            if module.kernel_size[0] >= 3 or (
                hasattr(module, "out_channels") and module.out_channels <= 256
            ):
                target = module
    if target is not None:
        return target

    return _resolve_generic_layer(model)


def _resolve_generic_layer(model: nn.Module) -> nn.Module:
    """Fallback: find the last meaningful layer in any model."""
    last_layer = None

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.LayerNorm, nn.BatchNorm2d)):
            last_layer = module

    if last_layer is not None:
        return last_layer

    # Absolute fallback: last named child
    children = list(model.children())
    if children:
        return children[-1]

    raise RuntimeError(
        "Could not auto-detect any suitable target layer. Please specify target_layer manually."
    )


def find_attention_layers(model: nn.Module) -> list[nn.Module]:
    """Find all attention modules in a transformer model.

    Searches for modules with names containing 'attn' or 'attention'
    that appear to be self-attention layers (have qkv or similar params).

    Args:
        model: The model to search.

    Returns:
        List of attention nn.Module instances.
    """
    attn_layers = []

    for name, module in model.named_modules():
        name_lower = name.lower()
        type_name = type(module).__name__.lower()

        # Match attention modules
        is_attn = (
            "attn" in name_lower
            or "attention" in name_lower
            or "attn" in type_name
            or "attention" in type_name
        )

        if not is_attn:
            continue

        # Verify it's an actual attention layer (not a wrapper)
        has_proj = (
            hasattr(module, "qkv")
            or hasattr(module, "in_proj_weight")
            or hasattr(module, "q_proj")
            or hasattr(module, "query")
            or hasattr(module, "to_qkv")
        )

        # Also accept if it has num_heads attribute
        has_heads = hasattr(module, "num_heads") or hasattr(module, "n_head")

        if has_proj or has_heads:
            attn_layers.append(module)

    return attn_layers
