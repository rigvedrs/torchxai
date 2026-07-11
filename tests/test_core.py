"""
Tests for torchxai core functionality.

Uses torchvision's built-in models (ResNet, ViT) to test all methods
end-to-end without requiring external model downloads.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def dummy_image():
    """Random image tensor (1, 3, 224, 224)."""
    torch.manual_seed(42)
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def pil_image():
    """Random PIL image."""
    from PIL import Image

    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def resnet_model():
    """Small ResNet model for testing."""
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.eval()
    return model


@pytest.fixture
def simple_cnn():
    """Minimal CNN for fast testing."""
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10),
    )
    model.eval()
    return model


# ── Architecture Detection ────────────────────────────────────────────


class TestArchDetection:
    def test_detect_cnn(self, resnet_model):
        from torchxai.models.registry import ArchType, detect_architecture

        assert detect_architecture(resnet_model) == ArchType.CNN

    def test_detect_simple_cnn(self, simple_cnn):
        from torchxai.models.registry import ArchType, detect_architecture

        assert detect_architecture(simple_cnn) == ArchType.CNN


# ── Target Layer Resolution ───────────────────────────────────────────


class TestLayerResolution:
    def test_resolve_resnet_layer(self, resnet_model):
        from torchxai.models.registry import ArchType, resolve_target_layer

        layer = resolve_target_layer(resnet_model, ArchType.CNN)
        assert layer is not None
        # ResNet should resolve to layer4[-1]
        assert layer == list(resnet_model.layer4.children())[-1]


# ── GradCAM ───────────────────────────────────────────────────────────


class TestGradCAM:
    def test_gradcam_output_shape(self, resnet_model, dummy_image):
        from torchxai.methods.gradcam import GradCAM

        cam = GradCAM(resnet_model)
        heatmap = cam(dummy_image)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (224, 224)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_gradcam_with_target_class(self, resnet_model, dummy_image):
        from torchxai.methods.gradcam import GradCAM

        cam = GradCAM(resnet_model)
        heatmap = cam(dummy_image, target_class=5)
        assert heatmap.shape == (224, 224)

    def test_gradcam_with_pil_image(self, resnet_model, pil_image):
        from torchxai.methods.gradcam import GradCAM

        cam = GradCAM(resnet_model)
        heatmap = cam(pil_image)
        assert heatmap.shape == (224, 224)

    def test_gradcam_custom_layer(self, resnet_model, dummy_image):
        from torchxai.methods.gradcam import GradCAM

        cam = GradCAM(resnet_model, target_layer=resnet_model.layer3[-1])
        heatmap = cam(dummy_image)
        assert heatmap.shape == (224, 224)

    def test_gradcam_simple_cnn(self, simple_cnn, dummy_image):
        from torchxai.methods.gradcam import GradCAM

        cam = GradCAM(simple_cnn)
        heatmap = cam(dummy_image)
        assert heatmap.shape == (224, 224)


# ── EigenCAM ──────────────────────────────────────────────────────────


class TestEigenCAM:
    def test_eigencam_output_shape(self, resnet_model, dummy_image):
        from torchxai.methods.eigencam import EigenCAM

        cam = EigenCAM(resnet_model)
        heatmap = cam(dummy_image)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (224, 224)

    def test_eigencam_no_gradient(self, resnet_model, dummy_image):
        """EigenCAM should work without gradient computation."""
        from torchxai.methods.eigencam import EigenCAM

        cam = EigenCAM(resnet_model)
        with torch.no_grad():
            heatmap = cam(dummy_image)
        assert heatmap.shape == (224, 224)


# ── LayerCAM ──────────────────────────────────────────────────────────


class TestLayerCAM:
    def test_layercam_output_shape(self, resnet_model, dummy_image):
        from torchxai.methods.layercam import LayerCAM

        cam = LayerCAM(resnet_model)
        heatmap = cam(dummy_image)
        assert heatmap.shape == (224, 224)


# ── GradCAM++ ─────────────────────────────────────────────────────────


class TestGradCAMPP:
    def test_gradcampp_output_shape(self, resnet_model, dummy_image):
        from torchxai.methods.gradcam_pp import GradCAMPlusPlus

        cam = GradCAMPlusPlus(resnet_model)
        heatmap = cam(dummy_image)
        assert heatmap.shape == (224, 224)


# ── High-level API ────────────────────────────────────────────────────


class TestExplainAPI:
    def test_explain_auto(self, resnet_model, dummy_image):
        from torchxai import explain

        heatmap = explain(resnet_model, dummy_image)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (224, 224)

    def test_explain_specific_method(self, resnet_model, dummy_image):
        from torchxai import explain

        heatmap = explain(resnet_model, dummy_image, method="eigencam")
        assert heatmap.shape == (224, 224)

    def test_explain_all_methods(self, resnet_model, dummy_image):
        from torchxai import explain

        results = explain(resnet_model, dummy_image, method="all")
        assert isinstance(results, dict)
        assert len(results) > 0
        for name, heatmap in results.items():
            assert heatmap.shape == (224, 224)

    def test_explain_with_pil(self, resnet_model, pil_image):
        from torchxai import explain

        heatmap = explain(resnet_model, pil_image)
        assert heatmap.shape == (224, 224)

    def test_explain_invalid_method(self, resnet_model, dummy_image):
        from torchxai import explain

        with pytest.raises(ValueError, match="Unknown method"):
            explain(resnet_model, dummy_image, method="nonexistent")


# ── Visualization ─────────────────────────────────────────────────────


class TestVisualization:
    def test_overlay_heatmap(self, pil_image):
        from torchxai.viz.visualize import overlay_heatmap

        heatmap = np.random.rand(224, 224).astype(np.float32)
        result = overlay_heatmap(pil_image, heatmap)
        assert result.shape == (224, 224, 3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_overlay_with_resize(self):
        from torchxai.viz.visualize import overlay_heatmap

        img = np.random.rand(100, 100, 3).astype(np.float32)
        heatmap = np.random.rand(50, 50).astype(np.float32)
        result = overlay_heatmap(img, heatmap)
        assert result.shape == (100, 100, 3)


# ── Metrics ───────────────────────────────────────────────────────────


class TestMetrics:
    def test_stability_score(self, resnet_model, dummy_image):
        from torchxai.methods.eigencam import EigenCAM
        from torchxai.metrics.stability import stability_score

        cam = EigenCAM(resnet_model)
        score = stability_score(
            explain_fn=cam,
            input_tensor=dummy_image,
            num_perturbations=3,
            noise_scale=0.01,
            seed=42,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ── Hooks ─────────────────────────────────────────────────────────────


class TestHooks:
    def test_activation_hook(self, resnet_model, dummy_image):
        from torchxai.utils.hooks import ActivationHook

        hook = ActivationHook(resnet_model.layer4)
        resnet_model(dummy_image)
        assert hook.activation is not None
        assert hook.activation.ndim == 4
        hook.remove()

    def test_multi_hook(self, resnet_model, dummy_image):
        from torchxai.utils.hooks import MultiHook

        hook = MultiHook(resnet_model.layer4)
        output = resnet_model(dummy_image)
        output.sum().backward()
        assert hook.activation is not None
        assert hook.gradient is not None
        hook.remove()

    def test_context_manager(self, resnet_model, dummy_image):
        from torchxai.utils.hooks import ActivationHook

        with ActivationHook(resnet_model.layer4) as hook:
            resnet_model(dummy_image)
            assert hook.activation is not None


# ── Image Utils ───────────────────────────────────────────────────────


class TestImageUtils:
    def test_preprocess_pil(self, pil_image):
        from torchxai.utils.image import preprocess_image

        tensor = preprocess_image(pil_image)
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_numpy(self):
        from torchxai.utils.image import preprocess_image

        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        tensor = preprocess_image(arr)
        assert tensor.shape == (1, 3, 224, 224)

    def test_normalize_heatmap(self):
        from torchxai.utils.image import normalize_heatmap

        heatmap = np.array([[1.0, 5.0], [3.0, 9.0]])
        result = normalize_heatmap(heatmap)
        assert result.min() == 0.0
        assert result.max() == 1.0

    def test_normalize_constant_heatmap(self):
        from torchxai.utils.image import normalize_heatmap

        heatmap = np.ones((5, 5))
        result = normalize_heatmap(heatmap)
        assert np.allclose(result, 0.0)  # Constant -> zero


# ── New Features (added during code audit) ────────────────────────────


class TestStringPathInput:
    """Test that explain() accepts file path strings."""

    def test_explain_with_string_path(self, resnet_model, tmp_path):
        from PIL import Image

        from torchxai import explain

        # Create a temporary image file
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        path = tmp_path / "test_image.jpg"
        img.save(str(path))

        heatmap = explain(resnet_model, str(path))
        assert heatmap.shape == (224, 224)

    def test_explain_with_pathlib_path(self, resnet_model, tmp_path):
        from pathlib import Path

        from PIL import Image

        from torchxai import explain

        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        path = tmp_path / "test_image.png"
        img.save(str(path))

        heatmap = explain(resnet_model, Path(path))
        assert heatmap.shape == (224, 224)

    def test_explain_nonexistent_path(self, resnet_model):
        from torchxai import explain

        with pytest.raises(FileNotFoundError, match="Image not found"):
            explain(resnet_model, "/nonexistent/fake_image.jpg")


class TestRepr:
    """Test __repr__ on explainers."""

    def test_gradcam_repr(self, resnet_model):
        from torchxai.methods.gradcam import GradCAM

        cam = GradCAM(resnet_model)
        r = repr(cam)
        assert "GradCAM" in r
        assert "arch=CNN" in r
        assert "device=" in r

    def test_eigencam_repr(self, resnet_model):
        from torchxai.methods.eigencam import EigenCAM

        cam = EigenCAM(resnet_model)
        r = repr(cam)
        assert "EigenCAM" in r
        assert "requires_grad" not in r  # Should not be in repr


class TestHeadlessVisualization:
    """Test that visualization works without a display."""

    def test_show_explanation_headless_save(self, pil_image, tmp_path):
        from torchxai.viz.visualize import show_explanation

        heatmap = np.random.rand(224, 224).astype(np.float32)
        save_path = str(tmp_path / "explanation.png")
        show_explanation(pil_image, heatmap, save_path=save_path)
        assert (tmp_path / "explanation.png").exists()

    def test_create_comparison_headless_save(self, pil_image, tmp_path):
        from torchxai.viz.visualize import create_comparison

        heatmaps = {
            "Method A": np.random.rand(224, 224).astype(np.float32),
            "Method B": np.random.rand(224, 224).astype(np.float32),
        }
        save_path = str(tmp_path / "comparison.png")
        create_comparison(pil_image, heatmaps, save_path=save_path)
        assert (tmp_path / "comparison.png").exists()

    def test_save_heatmap(self, tmp_path):
        from torchxai.viz.visualize import save_heatmap

        heatmap = np.random.rand(14, 14).astype(np.float32)
        save_path = str(tmp_path / "heatmap.png")
        save_heatmap(heatmap, save_path)
        assert (tmp_path / "heatmap.png").exists()
        # Verify it's a valid image
        from PIL import Image

        img = Image.open(save_path)
        assert img.size == (14, 14)


class TestGradientControl:
    """Test the requires_grad class attribute behavior."""

    def test_eigencam_no_grad_flag(self):
        from torchxai.methods.eigencam import EigenCAM

        assert EigenCAM.requires_grad is False

    def test_gradcam_grad_flag(self):
        from torchxai.methods.gradcam import GradCAM

        assert GradCAM.requires_grad is True

    def test_attention_rollout_no_grad_flag(self):
        from torchxai.methods.attention_rollout import AttentionRollout

        assert AttentionRollout.requires_grad is False


class TestEdgeCases:
    """Test edge cases that real users hit."""

    def test_explain_3d_tensor(self, resnet_model):
        """Users often pass (3,H,W) instead of (1,3,H,W)."""
        from torchxai import explain

        tensor = torch.randn(3, 224, 224)
        heatmap = explain(resnet_model, tensor)
        assert heatmap.shape == (224, 224)

    def test_overlay_rgba_image(self):
        """RGBA images should not crash overlay."""
        from torchxai.viz.visualize import overlay_heatmap

        rgba = np.random.rand(100, 100, 4).astype(np.float32)
        heatmap = np.random.rand(100, 100).astype(np.float32)
        result = overlay_heatmap(rgba, heatmap)
        assert result.shape == (100, 100, 3)

    def test_overlay_grayscale_image(self):
        """Grayscale images should be auto-converted to 3-channel."""
        from torchxai.viz.visualize import overlay_heatmap

        gray = np.random.rand(100, 100).astype(np.float32)
        heatmap = np.random.rand(100, 100).astype(np.float32)
        result = overlay_heatmap(gray, heatmap)
        assert result.shape == (100, 100, 3)

    def test_layer_by_name_string(self, resnet_model):
        """Users should be able to pass layer names as strings."""
        from torchxai.methods.gradcam import GradCAM

        cam = GradCAM(resnet_model, target_layer="layer4")
        assert cam.target_layer is resnet_model.layer4

    def test_layer_by_name_invalid(self, resnet_model):
        """Invalid layer names should give helpful error messages."""
        from torchxai.methods.gradcam import GradCAM

        with pytest.raises(AttributeError, match="Available"):
            GradCAM(resnet_model, target_layer="nonexistent_layer")
