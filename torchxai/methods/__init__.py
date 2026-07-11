"""
Explainability methods for vision models.

Each method implements a different approach to generating saliency maps:

- **GradCAM**: Gradient-weighted Class Activation Mapping (CNN + ViT)
- **GradCAM++**: Improved GradCAM with pixel-wise gradient weighting
- **EigenCAM**: Gradient-free method using PCA on activations
- **LayerCAM**: Layer-wise relevance with positive gradient weighting
- **AttentionRollout**: Attention aggregation across transformer layers
- **TransformerAttribution**: LRP-based attribution for transformers
"""

from torchxai.methods.attention_rollout import AttentionRollout
from torchxai.methods.eigencam import EigenCAM
from torchxai.methods.gradcam import GradCAM
from torchxai.methods.gradcam_pp import GradCAMPlusPlus
from torchxai.methods.layercam import LayerCAM
from torchxai.methods.transformer_attribution import TransformerAttribution

__all__ = [
    "GradCAM",
    "GradCAMPlusPlus",
    "EigenCAM",
    "LayerCAM",
    "AttentionRollout",
    "TransformerAttribution",
]
