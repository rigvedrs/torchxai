"""Model architecture detection and layer resolution for torchxai."""

from torchxai.models.registry import (
    ArchType,
    detect_architecture,
    find_attention_layers,
    resolve_target_layer,
)

__all__ = [
    "detect_architecture",
    "resolve_target_layer",
    "find_attention_layers",
    "ArchType",
]
