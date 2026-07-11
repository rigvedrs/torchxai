"""
Evaluation metrics for explainability quality.

Provides quantitative measures to assess how faithful and stable
saliency maps are — critical for comparing methods and validating
explanations.
"""

from torchxai.metrics.fidelity import deletion_score, insertion_score
from torchxai.metrics.stability import stability_score

__all__ = ["insertion_score", "deletion_score", "stability_score"]
