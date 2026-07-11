"""
Visualization module for torchxai.

Provides publication-quality visualization of saliency maps:
- Heatmap overlays on original images
- Side-by-side comparisons of multiple methods
- Detection-specific visualizations with bounding boxes
- Customizable colormaps and transparency

Works in all environments:
- Jupyter notebooks (inline display)
- Desktop scripts (GUI windows)
- Headless servers (auto-detects and uses Agg backend)
- CI/CD pipelines (no display required for save-only)
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Union

# ── Headless-safe matplotlib import ──────────────────────────────────
# Must happen before any matplotlib.pyplot import. If there's no display
# available (SSH, Docker, CI, servers), switch to Agg to avoid crashes.
import matplotlib
import numpy as np
from PIL import Image


def _ensure_safe_backend() -> None:
    """Switch to Agg backend if no display is available.

    This prevents the common crash:
        _tkinter.TclError: no display name and no $DISPLAY environment variable

    Which hits anyone running torchxai on a server, in Docker, or in CI.
    """
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return  # Display available — use whatever backend is configured

    # Check if we're in a Jupyter/IPython environment (which handles display)
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        if shell in ("ZMQInteractiveShell", "TerminalInteractiveShell"):
            return  # Jupyter or IPython — they handle backends themselves
    except NameError:
        pass

    # macOS can use macosx backend without DISPLAY
    if sys.platform == "darwin":
        return

    # No display detected — switch to non-interactive Agg backend
    current = matplotlib.get_backend().lower()
    interactive_backends = {"tkagg", "qt5agg", "qt4agg", "gtkagg", "gtk3agg", "wxagg"}
    if current in interactive_backends or current == "":
        try:
            matplotlib.use("Agg")
        except Exception:
            pass  # Already imported with a different backend — can't switch


_ensure_safe_backend()

import matplotlib.pyplot as plt  # noqa: E402 — must come after backend setup


def overlay_heatmap(
    image: Union[np.ndarray, Image.Image],
    heatmap: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.5,
    output_size: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Overlay a saliency heatmap on the original image.

    Args:
        image: Original image (H, W, 3) in [0, 1] or uint8, or PIL Image.
        heatmap: Saliency map (H, W) in [0, 1].
        colormap: Matplotlib colormap name. Default: "jet".
        alpha: Heatmap transparency. 0=image only, 1=heatmap only.
        output_size: Optional (H, W) to resize the output.

    Returns:
        Blended image as numpy array (H, W, 3) in [0, 1].
    """
    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        image = np.array(image).astype(np.float32) / 255.0

    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0

    # Ensure image is (H, W, 3)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]  # Drop alpha channel from RGBA

    h, w = image.shape[:2]

    # Resize heatmap to match image
    if heatmap.shape[0] != h or heatmap.shape[1] != w:
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize((w, h), Image.BILINEAR)
        heatmap = np.array(heatmap_img).astype(np.float32) / 255.0

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored_heatmap = cmap(heatmap)[:, :, :3]  # Drop alpha channel

    # Blend
    blended = (1 - alpha) * image + alpha * colored_heatmap
    blended = np.clip(blended, 0, 1)

    if output_size is not None:
        blended_img = Image.fromarray((blended * 255).astype(np.uint8))
        blended_img = blended_img.resize((output_size[1], output_size[0]), Image.BILINEAR)
        blended = np.array(blended_img).astype(np.float32) / 255.0

    return blended


def show_explanation(
    image: Union[np.ndarray, Image.Image],
    heatmap: np.ndarray,
    title: str = "",
    colormap: str = "jet",
    alpha: float = 0.5,
    figsize: tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Display original image, heatmap, and overlay side by side.

    Creates a three-panel visualization showing:
    1. Original image
    2. Raw saliency heatmap
    3. Overlay of heatmap on image

    Works in all environments:
    - GUI: shows interactive window
    - Jupyter: shows inline
    - Headless: saves to file (if save_path provided), returns array

    Args:
        image: Original image.
        heatmap: Saliency map (H, W) in [0, 1].
        title: Title for the figure.
        colormap: Matplotlib colormap.
        alpha: Overlay transparency.
        figsize: Figure size.
        save_path: If provided, saves the figure to this path.

    Returns:
        If running headless with no save_path, returns the figure as a
        numpy array (H, W, 3) so you can still use the visualization.
        Otherwise returns None.
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image).astype(np.float32) / 255.0
    else:
        image_np = image.copy()
        if image_np.max() > 1.0:
            image_np = image_np.astype(np.float32) / 255.0

    overlay = overlay_heatmap(image_np, heatmap, colormap=colormap, alpha=alpha)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap=colormap, vmin=0, vmax=1)
    axes[1].set_title("Saliency Map")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    # Show if we have a display, otherwise return as array
    result = None
    backend = matplotlib.get_backend().lower()
    if backend != "agg":
        plt.show()
    elif not save_path:
        # Headless with no save path — render to array so the user
        # gets something usable instead of nothing
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        result = np.asarray(buf)[:, :, :3].copy()  # RGBA → RGB

    plt.close(fig)
    return result


def create_comparison(
    image: Union[np.ndarray, Image.Image],
    heatmaps: dict[str, np.ndarray],
    colormap: str = "jet",
    alpha: float = 0.5,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Compare multiple explainability methods side by side.

    Creates a row showing the original image followed by overlay
    visualizations for each method.

    Args:
        image: Original image.
        heatmaps: Dict mapping method name -> saliency map.
            Example: {"GradCAM": cam1, "EigenCAM": cam2, "Rollout": cam3}
        colormap: Matplotlib colormap.
        alpha: Overlay transparency.
        figsize: Figure size. Auto-computed if None.
        save_path: If provided, saves the figure.

    Returns:
        If running headless with no save_path, returns figure as array.
        Otherwise returns None.
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image).astype(np.float32) / 255.0
    else:
        image_np = image.copy()
        if image_np.max() > 1.0:
            image_np = image_np.astype(np.float32) / 255.0

    n = len(heatmaps) + 1  # +1 for original
    if figsize is None:
        figsize = (4 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    axes[0].imshow(image_np)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    for i, (name, heatmap) in enumerate(heatmaps.items(), 1):
        overlay = overlay_heatmap(image_np, heatmap, colormap=colormap, alpha=alpha)
        axes[i].imshow(overlay)
        axes[i].set_title(name, fontsize=12)
        axes[i].axis("off")

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
        result = np.asarray(buf)[:, :, :3].copy()  # RGBA → RGB

    plt.close(fig)
    return result


def save_heatmap(
    heatmap: np.ndarray,
    path: str,
    colormap: str = "jet",
) -> None:
    """Save a saliency heatmap as an image file.

    Works in all environments (no display required).

    Args:
        heatmap: Saliency map (H, W) in [0, 1].
        path: Output file path (supports PNG, JPEG, BMP, TIFF).
        colormap: Matplotlib colormap.
    """
    cmap = plt.get_cmap(colormap)
    colored = cmap(heatmap)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    Image.fromarray(colored).save(path)
