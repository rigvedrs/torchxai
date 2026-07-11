"""
Object detection explainability for torchxai.

Provides bounding-box-level explanations for object detection models
like YOLO, DETR, Faster R-CNN, and SSD. Instead of explaining the
entire image, this module explains WHY the model detected a specific
object in a specific bounding box.

Usage:
    from torchxai.detection import explain_detection

    # With any detection model that returns boxes + scores
    explanations = explain_detection(model, image, detections)

    # Each explanation has: box, class_id, confidence, heatmap
    for exp in explanations:
        print(f"Class {exp.class_id}: confidence={exp.confidence:.2f}")
        overlay = overlay_heatmap(image, exp.heatmap)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
import warnings

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from torchxai.methods.eigencam import EigenCAM
from torchxai.methods.gradcam import GradCAM
from torchxai.utils.image import normalize_heatmap, preprocess_image


@dataclass
class DetectionExplanation:
    """Explanation for a single detected object.

    Attributes:
        box: Bounding box [x1, y1, x2, y2] in pixel coordinates.
        class_id: Predicted class index.
        class_name: Human-readable class name (if provided).
        confidence: Detection confidence score.
        heatmap: Saliency heatmap (H, W) in [0, 1] for this detection.
        heatmap_cropped: Saliency heatmap cropped to bounding box region.
    """

    box: list[float]
    class_id: int
    confidence: float
    heatmap: np.ndarray
    class_name: str = ""
    heatmap_cropped: Optional[np.ndarray] = None


def explain_detection(
    model: nn.Module,
    image: Union[torch.Tensor, Image.Image, np.ndarray, str],
    detections: Optional[list[dict]] = None,
    method: str = "eigencam",
    target_layer: Optional[Union[str, nn.Module]] = None,
    max_detections: int = 10,
    class_names: Optional[list[str]] = None,
    image_size: tuple[int, int] = (640, 640),
) -> list[DetectionExplanation]:
    """Generate per-detection explanations for an object detection model.

    Works with any model that produces bounding box detections. If
    detections are not provided, the function runs the model to get them.

    Args:
        model: Detection model (YOLO, DETR, Faster R-CNN, etc.).
        image: Input image in any format.
        detections: Pre-computed detections as list of dicts, each with:
            - "box": [x1, y1, x2, y2]
            - "class_id": int
            - "confidence": float
            - "class_name": str (optional)
            If None, attempts to run model and parse output.
        method: Explainability method ("eigencam" or "gradcam").
            EigenCAM is recommended for detection (faster, no gradients).
        target_layer: Target layer for explanation. If None, auto-detected.
        max_detections: Maximum number of detections to explain.
        class_names: List of class names indexed by class_id.
        image_size: Size to preprocess image to.

    Returns:
        List of DetectionExplanation objects, one per detection.

    Example:
        >>> detections = [
        ...     {"box": [100, 50, 300, 400], "class_id": 0, "confidence": 0.92},
        ...     {"box": [350, 100, 500, 350], "class_id": 1, "confidence": 0.87},
        ... ]
        >>> explanations = explain_detection(model, image, detections)
        >>> for exp in explanations:
        ...     print(f"{exp.class_name}: {exp.confidence:.2f}")
    """
    model.eval()

    # Preprocess image
    if isinstance(image, str):
        pil_img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        pil_img = None
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Get image tensor
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        input_tensor = image.float()
        h, w = input_tensor.shape[2], input_tensor.shape[3]
    else:
        input_tensor = preprocess_image(pil_img, size=image_size)
        h, w = image_size

    device = next(model.parameters(), torch.zeros(1)).device
    input_tensor = input_tensor.to(device)

    # If no detections provided, try to run the model
    if detections is None:
        detections = _run_detection(model, input_tensor)

    if not detections:
        warnings.warn(
            "No detections found. Provide detections manually or check "
            "that the model produces bounding box outputs.",
            UserWarning,
        )
        return []

    # Limit detections
    detections = detections[:max_detections]

    # Generate global saliency map
    if method == "gradcam":
        explainer = GradCAM(model, target_layer=target_layer, device=device)
    else:
        explainer = EigenCAM(model, target_layer=target_layer, device=device)

    try:
        global_heatmap = explainer(input_tensor)
    except Exception as e:
        warnings.warn(
            f"Could not generate heatmap: {e}. Using uniform heatmap.",
            UserWarning,
        )
        global_heatmap = np.ones((h, w), dtype=np.float32) * 0.5

    # Generate per-detection explanations
    explanations = []
    for det in detections:
        box = det["box"]
        class_id = det.get("class_id", 0)
        confidence = det.get("confidence", 0.0)
        class_name = det.get("class_name", "")

        if not class_name and class_names and class_id < len(class_names):
            class_name = class_names[class_id]

        # Crop heatmap to bounding box region
        x1, y1, x2, y2 = [int(c) for c in box]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        # Scale box coordinates to heatmap size
        hm_h, hm_w = global_heatmap.shape
        scale_x = hm_w / w
        scale_y = hm_h / h
        hx1 = int(x1 * scale_x)
        hy1 = int(y1 * scale_y)
        hx2 = int(x2 * scale_x)
        hy2 = int(y2 * scale_y)
        hx2 = max(hx1 + 1, hx2)
        hy2 = max(hy1 + 1, hy2)

        cropped = global_heatmap[hy1:hy2, hx1:hx2]
        if cropped.size > 0:
            cropped = normalize_heatmap(cropped)
        else:
            cropped = np.zeros((1, 1), dtype=np.float32)

        # Create detection-focused heatmap (mask outside box)
        det_heatmap = np.zeros_like(global_heatmap)
        det_heatmap[hy1:hy2, hx1:hx2] = global_heatmap[hy1:hy2, hx1:hx2]
        if det_heatmap.max() > 0:
            det_heatmap = normalize_heatmap(det_heatmap)

        explanations.append(
            DetectionExplanation(
                box=[x1, y1, x2, y2],
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                heatmap=det_heatmap,
                heatmap_cropped=cropped,
            )
        )

    return explanations


def _run_detection(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> list[dict]:
    """Attempt to run a detection model and parse its output.

    Handles common output formats from YOLO, DETR, and torchvision
    detection models.
    """
    try:
        with torch.no_grad():
            output = model(input_tensor)
    except Exception:
        return []

    detections = []

    # Torchvision Faster R-CNN / SSD format: list of dicts
    if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
        d = output[0]
        boxes = d.get("boxes", torch.zeros(0, 4))
        scores = d.get("scores", torch.zeros(0))
        labels = d.get("labels", torch.zeros(0, dtype=torch.int64))

        for i in range(min(len(boxes), 50)):
            if scores[i].item() > 0.3:
                detections.append(
                    {
                        "box": boxes[i].tolist(),
                        "class_id": int(labels[i].item()),
                        "confidence": float(scores[i].item()),
                    }
                )

    # Ultralytics YOLO format: Results object with .boxes
    elif hasattr(output, "boxes") or (isinstance(output, list) and hasattr(output[0], "boxes")):
        result = output[0] if isinstance(output, list) else output
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                det = {
                    "box": boxes.xyxy[i].tolist() if hasattr(boxes, "xyxy") else [0, 0, 1, 1],
                    "class_id": int(boxes.cls[i].item()) if hasattr(boxes, "cls") else 0,
                    "confidence": float(boxes.conf[i].item()) if hasattr(boxes, "conf") else 0.5,
                }
                detections.append(det)

    return detections


def visualize_detections(
    image: Union[np.ndarray, Image.Image],
    explanations: list[DetectionExplanation],
    colormap: str = "jet",
    alpha: float = 0.5,
    show_boxes: bool = True,
    show_labels: bool = True,
    save_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Visualize detection explanations with bounding boxes and heatmaps.

    Creates an overlay showing bounding boxes with per-detection
    saliency heatmaps.

    Args:
        image: Original image.
        explanations: List of DetectionExplanation objects.
        colormap: Matplotlib colormap.
        alpha: Heatmap transparency.
        show_boxes: Draw bounding boxes.
        show_labels: Show class labels and confidence.
        save_path: If provided, saves the figure.

    Returns:
        Overlay image as numpy array if headless, else None.
    """
    import matplotlib
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    if isinstance(image, Image.Image):
        image = np.array(image).astype(np.float32) / 255.0
    elif image.max() > 1.0:
        image = image.astype(np.float32) / 255.0

    # Combine all detection heatmaps
    combined_heatmap = np.zeros(image.shape[:2], dtype=np.float32)
    for exp in explanations:
        # Resize heatmap to image size if needed
        hm = exp.heatmap
        if hm.shape[0] != image.shape[0] or hm.shape[1] != image.shape[1]:
            hm_img = Image.fromarray((hm * 255).astype(np.uint8))
            hm_img = hm_img.resize((image.shape[1], image.shape[0]), Image.BILINEAR)
            hm = np.array(hm_img).astype(np.float32) / 255.0
        combined_heatmap = np.maximum(combined_heatmap, hm)

    # Create overlay
    from torchxai.viz.visualize import overlay_heatmap

    overlay = overlay_heatmap(image, combined_heatmap, colormap=colormap, alpha=alpha)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(overlay)

    if show_boxes:
        for exp in explanations:
            x1, y1, x2, y2 = exp.box
            w = x2 - x1
            h = y2 - y1
            rect = patches.Rectangle(
                (x1, y1), w, h, linewidth=2, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)

            if show_labels:
                label = exp.class_name or f"class_{exp.class_id}"
                label = f"{label} {exp.confidence:.0%}"
                ax.text(
                    x1,
                    y1 - 5,
                    label,
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.8),
                )

    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    result = None
    backend = matplotlib.get_backend().lower()
    if backend != "agg":
        plt.show()
    elif not save_path:
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        result = np.asarray(buf)[:, :, :3].copy()

    plt.close(fig)
    return result
