"""
torchxai — Universal explainability for modern vision models.

Supports CNNs, Vision Transformers, CLIP, SAM, DETR, YOLOv12, and VLMs
with a single unified API.

Usage:
    from torchxai import explain
    saliency = explain(model, image)

    # Or use specific methods:
    from torchxai.methods import GradCAM, EigenCAM, AttentionRollout
    cam = GradCAM(model)
    result = cam(image)
"""

__version__ = "0.2.0"

from torchxai.api import explain
from torchxai.batch import explain_batch, explain_directory, export_results
from torchxai.detection import DetectionExplanation, explain_detection, visualize_detections
from torchxai.methods.attention_rollout import AttentionRollout
from torchxai.methods.eigencam import EigenCAM
from torchxai.methods.gradcam import GradCAM
from torchxai.methods.gradcam_pp import GradCAMPlusPlus
from torchxai.methods.integrated_gradients import IntegratedGradients
from torchxai.methods.layercam import LayerCAM
from torchxai.methods.rise import RISE
from torchxai.methods.scorecam import ScoreCAM
from torchxai.methods.smoothgrad import SmoothGrad
from torchxai.methods.transformer_attribution import TransformerAttribution
from torchxai.metrics.fidelity import deletion_score, insertion_score
from torchxai.metrics.stability import stability_score
from torchxai.viz.visualize import (
    create_comparison,
    overlay_heatmap,
    save_heatmap,
    show_explanation,
)

__all__ = [
    "explain",
    # Methods
    "GradCAM",
    "GradCAMPlusPlus",
    "EigenCAM",
    "LayerCAM",
    "AttentionRollout",
    "TransformerAttribution",
    "ScoreCAM",
    "SmoothGrad",
    "IntegratedGradients",
    "RISE",
    # Visualization
    "overlay_heatmap",
    "show_explanation",
    "create_comparison",
    "save_heatmap",
    # Metrics
    "insertion_score",
    "deletion_score",
    "stability_score",
    # Detection
    "explain_detection",
    "DetectionExplanation",
    "visualize_detections",
    # Batch Processing
    "explain_batch",
    "explain_directory",
    "export_results",
]
