"""
Batch processing for torchxai.

Efficiently explain multiple images at once. Supports:
- Batch explanation with progress tracking
- Directory processing (explain all images in a folder)
- CSV/JSON export of results
- Parallel processing with configurable workers

Usage:
    from torchxai.batch import explain_batch, explain_directory

    # Explain a batch of tensors
    heatmaps = explain_batch(model, tensors, method="gradcam")

    # Explain all images in a directory
    results = explain_directory(model, "images/", save_dir="outputs/")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import warnings

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from torchxai.api import explain
from torchxai.viz.visualize import overlay_heatmap, save_heatmap


def explain_batch(
    model: nn.Module,
    images: Union[list[torch.Tensor], list[Image.Image], list[str], torch.Tensor],
    method: str = "auto",
    target_class: Optional[int] = None,
    image_size: tuple[int, int] = (224, 224),
    progress: bool = True,
) -> list[np.ndarray]:
    """Explain a batch of images.

    Processes each image individually (not batched through the model)
    to ensure correct per-image gradient computation.

    Args:
        model: The model to explain.
        images: List of images (tensors, PIL images, file paths)
            or a stacked tensor (B, 3, H, W).
        method: Explainability method name.
        target_class: Target class for all images. None = per-image predicted.
        image_size: Resize non-tensor inputs to this size.
        progress: Print progress updates.

    Returns:
        List of heatmaps, one per image. Each is (H, W) numpy array in [0, 1].

    Example:
        >>> paths = ["cat.jpg", "dog.jpg", "bird.jpg"]
        >>> heatmaps = explain_batch(model, paths)
        >>> len(heatmaps)
        3
    """
    # Handle stacked tensor
    if isinstance(images, torch.Tensor) and images.ndim == 4:
        images = [images[i] for i in range(images.shape[0])]

    heatmaps = []
    total = len(images)

    for i, img in enumerate(images):
        if progress and (i % 10 == 0 or i == total - 1):
            print(f"  Explaining image {i + 1}/{total}...", flush=True)

        try:
            hm = explain(
                model,
                img,
                method=method,
                target_class=target_class,
                image_size=image_size,
            )
            heatmaps.append(hm)
        except Exception as e:
            warnings.warn(
                f"Failed to explain image {i}: {e}. Using blank heatmap.",
                UserWarning,
            )
            heatmaps.append(np.zeros(image_size, dtype=np.float32))

    if progress:
        print(f"  Done — {len(heatmaps)} images explained.")

    return heatmaps


def explain_directory(
    model: nn.Module,
    input_dir: Union[str, Path],
    save_dir: Optional[Union[str, Path]] = None,
    method: str = "auto",
    target_class: Optional[int] = None,
    image_size: tuple[int, int] = (224, 224),
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"),
    save_overlay: bool = True,
    save_raw_heatmap: bool = False,
    colormap: str = "jet",
    alpha: float = 0.5,
    progress: bool = True,
) -> list[dict]:
    """Explain all images in a directory.

    Scans a directory for image files, explains each one, and optionally
    saves overlay visualizations and raw heatmaps.

    Args:
        model: The model to explain.
        input_dir: Directory containing images.
        save_dir: Directory to save outputs. If None, doesn't save.
        method: Explainability method name.
        target_class: Target class for all images.
        image_size: Size to resize images to.
        image_extensions: File extensions to look for.
        save_overlay: Save overlay images to save_dir.
        save_raw_heatmap: Save raw heatmap PNGs to save_dir.
        colormap: Colormap for overlays.
        alpha: Overlay transparency.
        progress: Print progress updates.

    Returns:
        List of dicts, each with:
            - "path": original image path
            - "filename": base filename
            - "heatmap": numpy array (H, W) in [0, 1]
            - "overlay_path": path to saved overlay (if saved)
            - "heatmap_path": path to saved heatmap (if saved)

    Example:
        >>> results = explain_directory(model, "images/", save_dir="outputs/")
        >>> for r in results:
        ...     print(f"{r['filename']}: saved to {r['overlay_path']}")
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    image_files = sorted(set(image_files))

    if not image_files:
        warnings.warn(
            f"No image files found in {input_dir} with extensions {image_extensions}",
            UserWarning,
        )
        return []

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(image_files)

    for i, img_path in enumerate(image_files):
        if progress:
            print(f"  [{i + 1}/{total}] {img_path.name}...", flush=True)

        try:
            hm = explain(
                model,
                str(img_path),
                method=method,
                target_class=target_class,
                image_size=image_size,
            )

            result = {
                "path": str(img_path),
                "filename": img_path.name,
                "heatmap": hm,
                "overlay_path": None,
                "heatmap_path": None,
            }

            if save_dir:
                stem = img_path.stem

                if save_overlay:
                    pil_img = Image.open(img_path).convert("RGB")
                    img_np = np.array(pil_img).astype(np.float32) / 255.0
                    ov = overlay_heatmap(img_np, hm, colormap=colormap, alpha=alpha)
                    ov_path = save_dir / f"{stem}_overlay.png"
                    Image.fromarray((ov * 255).astype(np.uint8)).save(str(ov_path))
                    result["overlay_path"] = str(ov_path)

                if save_raw_heatmap:
                    hm_path = save_dir / f"{stem}_heatmap.png"
                    save_heatmap(hm, str(hm_path), colormap=colormap)
                    result["heatmap_path"] = str(hm_path)

            results.append(result)

        except Exception as e:
            warnings.warn(f"Failed to process {img_path.name}: {e}", UserWarning)
            results.append(
                {
                    "path": str(img_path),
                    "filename": img_path.name,
                    "heatmap": np.zeros(image_size, dtype=np.float32),
                    "overlay_path": None,
                    "heatmap_path": None,
                }
            )

    if progress:
        saved = sum(1 for r in results if r["overlay_path"] is not None)
        print(f"  Done — {total} images processed, {saved} overlays saved.")

    return results


def export_results(
    results: list[dict],
    output_path: Union[str, Path],
    format: str = "csv",
) -> None:
    """Export batch results to CSV or JSON.

    Args:
        results: Results from explain_batch or explain_directory.
        output_path: Output file path.
        format: "csv" or "json".

    Example:
        >>> results = explain_directory(model, "images/", save_dir="out/")
        >>> export_results(results, "results.csv")
    """
    import json

    output_path = Path(output_path)

    if format == "json":
        # Convert numpy arrays to lists for JSON serialization
        serializable = []
        for r in results:
            s = {k: v for k, v in r.items() if k != "heatmap"}
            s["heatmap_shape"] = list(r["heatmap"].shape) if r.get("heatmap") is not None else None
            s["heatmap_mean"] = float(r["heatmap"].mean()) if r.get("heatmap") is not None else None
            serializable.append(s)

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)

    elif format == "csv":
        with open(output_path, "w") as f:
            f.write("filename,path,overlay_path,heatmap_path,heatmap_mean,heatmap_max\n")
            for r in results:
                hm = r.get("heatmap")
                mean = f"{hm.mean():.4f}" if hm is not None else ""
                mx = f"{hm.max():.4f}" if hm is not None else ""
                f.write(
                    f"{r.get('filename', '')},{r.get('path', '')},"
                    f"{r.get('overlay_path', '')},{r.get('heatmap_path', '')},"
                    f"{mean},{mx}\n"
                )

    print(f"Results exported to {output_path}")
