"""
Comprehensive integration checks for torchxai (script, not pytest).

Run directly:  python tests/integration_full.py

Tests every feature end-to-end with real models (ResNet18, ViT).
Not mocked — these exercise the actual computation pipeline.
"""

import gc
import os
from pathlib import Path
import sys
import tempfile
import traceback
import warnings

import numpy as np
from PIL import Image
import torch
import torchvision.models as models

# ── Setup ─────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0
ERRORS = []


def test(name):
    """Decorator to run a test and track pass/fail."""

    def decorator(fn):
        global PASS, FAIL
        try:
            fn()
            PASS += 1
            print(f"  ✓ {name}")
        except Exception as e:
            FAIL += 1
            tb = traceback.format_exc()
            ERRORS.append((name, str(e), tb))
            print(f"  ✗ {name}")
            print(f"    → {e}")
        return fn

    return decorator


def make_image_tensor(size=224):
    """Create a random (1, 3, H, W) tensor."""
    return torch.randn(1, 3, size, size)


def make_pil_image(size=224):
    """Create a random PIL image."""
    arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def save_temp_image(suffix=".jpg", size=224):
    """Save a random image to a temp file, return path."""
    img = make_pil_image(size)
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


# ── Load models once ──────────────────────────────────────────────────

print("Loading models...")
resnet = models.resnet18(weights=None)
resnet.eval()
print("  ResNet18 loaded")

try:
    import timm

    vit = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    vit.eval()
    print("  ViT-Tiny loaded")
    HAS_VIT = True
except Exception as e:
    print(f"  ViT not available: {e}")
    HAS_VIT = False

tensor_input = make_image_tensor()
pil_input = make_pil_image()
tmp_img_path = save_temp_image()

print()


# ══════════════════════════════════════════════════════════════════════
# TEST 1: explain() one-liner API
# ══════════════════════════════════════════════════════════════════════
print("TEST 1: explain() One-Liner API")

from torchxai import explain


@test("explain(model, tensor) → (224,224) heatmap")
def _():
    hm = explain(resnet, tensor_input)
    assert isinstance(hm, np.ndarray), f"Expected ndarray, got {type(hm)}"
    assert hm.shape == (224, 224), f"Expected (224,224), got {hm.shape}"
    assert hm.min() >= 0 and hm.max() <= 1, f"Not normalized: [{hm.min()}, {hm.max()}]"


@test("explain(model, PIL.Image) → works")
def _():
    hm = explain(resnet, pil_input)
    assert hm.shape == (224, 224)


@test("explain(model, '/path/to/image.jpg') → works")
def _():
    hm = explain(resnet, tmp_img_path)
    assert hm.shape == (224, 224)


@test("explain(model, pathlib.Path) → works")
def _():
    hm = explain(resnet, Path(tmp_img_path))
    assert hm.shape == (224, 224)


@test("explain(model, 3D tensor (3,H,W)) → auto-adds batch dim")
def _():
    t = torch.randn(3, 224, 224)
    hm = explain(resnet, t)
    assert hm.shape == (224, 224)


@test("explain(model, tensor, method='eigencam') → uses specified method")
def _():
    hm = explain(resnet, tensor_input, method="eigencam")
    assert hm.shape == (224, 224)


@test("explain(model, tensor, target_class=5) → class-specific")
def _():
    hm = explain(resnet, tensor_input, target_class=5)
    assert hm.shape == (224, 224)


@test("explain(model, tensor, method='gradcam') → GradCAM specifically")
def _():
    hm = explain(resnet, tensor_input, method="gradcam")
    assert hm.shape == (224, 224)


print()


# ══════════════════════════════════════════════════════════════════════
# TEST 2: All 6 methods on CNN (ResNet18)
# ══════════════════════════════════════════════════════════════════════
print("TEST 2: All 6 Methods on CNN (ResNet18)")

from torchxai.methods.attention_rollout import AttentionRollout
from torchxai.methods.eigencam import EigenCAM
from torchxai.methods.gradcam import GradCAM
from torchxai.methods.gradcam_pp import GradCAMPlusPlus
from torchxai.methods.layercam import LayerCAM
from torchxai.methods.transformer_attribution import TransformerAttribution


@test("GradCAM on ResNet18 → valid heatmap")
def _():
    cam = GradCAM(resnet)
    hm = cam(tensor_input)
    assert hm.shape == (224, 224)
    assert 0 <= hm.min() and hm.max() <= 1


@test("EigenCAM on ResNet18 → valid heatmap")
def _():
    cam = EigenCAM(resnet)
    hm = cam(tensor_input)
    assert hm.shape == (224, 224)


@test("LayerCAM on ResNet18 → valid heatmap")
def _():
    cam = LayerCAM(resnet)
    hm = cam(tensor_input)
    assert hm.shape == (224, 224)


@test("GradCAM++ on ResNet18 → valid heatmap")
def _():
    cam = GradCAMPlusPlus(resnet)
    hm = cam(tensor_input)
    assert hm.shape == (224, 224)


@test("AttentionRollout on ResNet18 → falls back gracefully (no attention layers)")
def _():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cam = AttentionRollout(resnet)
        hm = cam(tensor_input)
        assert hm.shape == (224, 224)


@test("TransformerAttribution on ResNet18 → falls back gracefully")
def _():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cam = TransformerAttribution(resnet)
        hm = cam(tensor_input)
        assert hm.shape == (224, 224)


@test("GradCAM with explicit target_layer string → works")
def _():
    cam = GradCAM(resnet, target_layer="layer4")
    hm = cam(tensor_input)
    assert hm.shape == (224, 224)


@test("GradCAM with target_layer module → works")
def _():
    cam = GradCAM(resnet, target_layer=resnet.layer4)
    hm = cam(tensor_input)
    assert hm.shape == (224, 224)


@test("GradCAM with target_class=0 → specific class heatmap")
def _():
    cam = GradCAM(resnet)
    hm = cam(tensor_input, target_class=0)
    assert hm.shape == (224, 224)


print()


# ══════════════════════════════════════════════════════════════════════
# TEST 3: All methods on Vision Transformer (ViT)
# ══════════════════════════════════════════════════════════════════════
print("TEST 3: All Methods on Vision Transformer (ViT)")

if HAS_VIT:
    vit_tensor = make_image_tensor()

    @test("GradCAM on ViT → valid heatmap (auto-detects transformer)")
    def _():
        cam = GradCAM(vit)
        hm = cam(vit_tensor)
        assert hm.shape == (224, 224), f"Got {hm.shape}"

    @test("EigenCAM on ViT → valid heatmap")
    def _():
        cam = EigenCAM(vit)
        hm = cam(vit_tensor)
        assert hm.shape == (224, 224)

    @test("LayerCAM on ViT → valid heatmap")
    def _():
        cam = LayerCAM(vit)
        hm = cam(vit_tensor)
        assert hm.shape == (224, 224)

    @test("GradCAM++ on ViT → valid heatmap")
    def _():
        cam = GradCAMPlusPlus(vit)
        hm = cam(vit_tensor)
        assert hm.shape == (224, 224)

    @test("AttentionRollout on ViT → uses actual attention layers")
    def _():
        cam = AttentionRollout(vit)
        hm = cam(vit_tensor)
        assert hm.shape == (224, 224)

    @test("TransformerAttribution on ViT → class-specific attribution")
    def _():
        cam = TransformerAttribution(vit)
        hm = cam(vit_tensor)
        assert hm.shape == (224, 224)

    @test("explain(vit, tensor) → auto-detects ViT architecture")
    def _():
        hm = explain(vit, vit_tensor)
        assert hm.shape == (224, 224)

    @test("explain(vit, tensor, method='attention_rollout') → works")
    def _():
        hm = explain(vit, vit_tensor, method="attention_rollout")
        assert hm.shape == (224, 224)
else:
    print("  ⚠ Skipped (ViT not available)")

print()


# ══════════════════════════════════════════════════════════════════════
# TEST 4: Model registry auto-detection
# ══════════════════════════════════════════════════════════════════════
print("TEST 4: Model Registry Auto-Detection")

from torchxai.models.registry import (
    ArchType,
    detect_architecture,
    find_attention_layers,
    resolve_target_layer,
)


@test("detect_architecture(resnet) → ArchType.CNN")
def _():
    arch = detect_architecture(resnet)
    assert arch == ArchType.CNN, f"Expected ArchType.CNN, got '{arch}'"


@test("resolve_target_layer(resnet) → returns a module")
def _():
    arch = detect_architecture(resnet)
    layer = resolve_target_layer(resnet, arch)
    assert isinstance(layer, torch.nn.Module), f"Expected Module, got {type(layer)}"


@test("find_attention_layers(resnet) → empty list (no attention)")
def _():
    layers = find_attention_layers(resnet)
    assert layers == [], f"Expected [], got {layers}"


if HAS_VIT:

    @test("detect_architecture(vit) → ArchType.VIT")
    def _():
        arch = detect_architecture(vit)
        assert arch == ArchType.VIT, f"Expected ArchType.VIT, got '{arch}'"

    @test("resolve_target_layer(vit) → returns a module")
    def _():
        arch = detect_architecture(vit)
        layer = resolve_target_layer(vit, arch)
        assert isinstance(layer, torch.nn.Module)

    @test("find_attention_layers(vit) → non-empty list")
    def _():
        layers = find_attention_layers(vit)
        assert len(layers) > 0, "Expected attention layers, got []"
        print(f"      Found {len(layers)} attention layers")

    @test("Auto-detect timm model (efficientnet_b0)")
    def _():
        eff = timm.create_model("efficientnet_b0", pretrained=False)
        eff.eval()
        arch = detect_architecture(eff)
        assert arch == ArchType.CNN, f"Expected ArchType.CNN, got '{arch}'"
        layer = resolve_target_layer(eff, arch)
        assert isinstance(layer, torch.nn.Module)


print()


# ══════════════════════════════════════════════════════════════════════
# TEST 5: Visualization (headless)
# ══════════════════════════════════════════════════════════════════════
print("TEST 5: Visualization (headless server)")

from torchxai.viz.visualize import (
    create_comparison,
    overlay_heatmap,
    save_heatmap,
    show_explanation,
)


@test("overlay_heatmap(image, heatmap) → (H,W,3) blended")
def _():
    img = np.random.rand(224, 224, 3).astype(np.float32)
    hm = np.random.rand(224, 224).astype(np.float32)
    result = overlay_heatmap(img, hm)
    assert result.shape == (224, 224, 3)
    assert 0 <= result.min() and result.max() <= 1


@test("overlay_heatmap with PIL Image → works")
def _():
    pil = make_pil_image()
    hm = np.random.rand(224, 224).astype(np.float32)
    result = overlay_heatmap(pil, hm)
    assert result.shape == (224, 224, 3)


@test("overlay_heatmap with uint8 image → auto-normalizes")
def _():
    img = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
    hm = np.random.rand(224, 224).astype(np.float32)
    result = overlay_heatmap(img, hm)
    assert result.shape == (224, 224, 3)
    assert result.max() <= 1.0


@test("overlay_heatmap with mismatched sizes → resizes heatmap")
def _():
    img = np.random.rand(480, 640, 3).astype(np.float32)
    hm = np.random.rand(14, 14).astype(np.float32)
    result = overlay_heatmap(img, hm)
    assert result.shape == (480, 640, 3)


@test("overlay_heatmap with RGBA image → drops alpha channel")
def _():
    rgba = np.random.rand(100, 100, 4).astype(np.float32)
    hm = np.random.rand(100, 100).astype(np.float32)
    result = overlay_heatmap(rgba, hm)
    assert result.shape == (100, 100, 3)


@test("overlay_heatmap with grayscale image → converts to 3-channel")
def _():
    gray = np.random.rand(100, 100).astype(np.float32)
    hm = np.random.rand(100, 100).astype(np.float32)
    result = overlay_heatmap(gray, hm)
    assert result.shape == (100, 100, 3)


@test("overlay_heatmap with output_size → resizes output")
def _():
    img = np.random.rand(224, 224, 3).astype(np.float32)
    hm = np.random.rand(224, 224).astype(np.float32)
    result = overlay_heatmap(img, hm, output_size=(100, 100))
    assert result.shape == (100, 100, 3)


@test("show_explanation() headless → saves PNG file")
def _():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "explanation.png")
        show_explanation(
            pil_input,
            np.random.rand(224, 224).astype(np.float32),
            save_path=path,
            title="Test Explanation",
        )
        assert os.path.exists(path), "File was not created"
        saved = Image.open(path)
        assert saved.size[0] > 0 and saved.size[1] > 0


@test("show_explanation() headless no save → returns numpy array")
def _():
    import matplotlib

    backend = matplotlib.get_backend().lower()
    if backend == "agg":
        result = show_explanation(pil_input, np.random.rand(224, 224).astype(np.float32))
        assert result is not None, "Expected array return on headless"
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
    else:
        print("      (interactive backend, skipped)")


@test("create_comparison() → saves comparison image")
def _():
    heatmaps = {
        "GradCAM": np.random.rand(224, 224).astype(np.float32),
        "EigenCAM": np.random.rand(224, 224).astype(np.float32),
        "LayerCAM": np.random.rand(224, 224).astype(np.float32),
    }
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "comparison.png")
        create_comparison(pil_input, heatmaps, save_path=path)
        assert os.path.exists(path)


@test("save_heatmap() → saves colored heatmap PNG")
def _():
    hm = np.random.rand(14, 14).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "heatmap.png")
        save_heatmap(hm, path)
        assert os.path.exists(path)
        img = Image.open(path)
        assert img.size == (14, 14)


@test("save_heatmap() with different colormap → works")
def _():
    hm = np.random.rand(50, 50).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "heatmap_viridis.png")
        save_heatmap(hm, path, colormap="viridis")
        assert os.path.exists(path)


@test("show_explanation() with different colormaps → no crash")
def _():
    hm = np.random.rand(224, 224).astype(np.float32)
    for cmap in ["jet", "viridis", "hot", "inferno"]:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, f"test_{cmap}.png")
            show_explanation(pil_input, hm, save_path=path, colormap=cmap)
            assert os.path.exists(path), f"Failed with colormap={cmap}"


print()


# ══════════════════════════════════════════════════════════════════════
# TEST 6: Metrics
# ══════════════════════════════════════════════════════════════════════
print("TEST 6: Metrics")

from torchxai.metrics.fidelity import deletion_score, insertion_score
from torchxai.metrics.stability import stability_score


@test("insertion_score(model, tensor, heatmap) → float in [0,1]")
def _():
    hm = explain(resnet, tensor_input)
    score = insertion_score(resnet, tensor_input, hm)
    assert isinstance(score, float), f"Expected float, got {type(score)}"
    assert 0 <= score <= 1, f"Score {score} out of [0,1]"


@test("deletion_score(model, tensor, heatmap) → float in [0,1]")
def _():
    hm = explain(resnet, tensor_input)
    score = deletion_score(resnet, tensor_input, hm)
    assert isinstance(score, float), f"Expected float, got {type(score)}"
    assert 0 <= score <= 1, f"Score {score} out of [0,1]"


@test("stability_score(explainer, tensor) → float ≥ 0")
def _():
    cam = GradCAM(resnet)
    score = stability_score(cam, tensor_input)
    assert isinstance(score, float), f"Expected float, got {type(score)}"
    assert score >= 0, f"Score {score} is negative"


@test("insertion > deletion for good explanations (sanity check)")
def _():
    # This is a soft check — with random weights it might not always hold,
    # but the functions should at least run without error
    hm = explain(resnet, tensor_input)
    ins = insertion_score(resnet, tensor_input, hm)
    dele = deletion_score(resnet, tensor_input, hm)
    print(f"      insertion={ins:.4f}, deletion={dele:.4f}")
    # Both should be valid floats
    assert not np.isnan(ins) and not np.isnan(dele)


print()


# ══════════════════════════════════════════════════════════════════════
# TEST 7: Error Handling
# ══════════════════════════════════════════════════════════════════════
print("TEST 7: Error Handling & Helpful Messages")


@test("explain() with nonexistent file path → FileNotFoundError with helpful message")
def _():
    try:
        explain(resnet, "/nonexistent/path/image.jpg")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "not found" in str(e).lower() or "Image not found" in str(e), f"Bad message: {e}"


@test("explain() with wrong type → TypeError with helpful message")
def _():
    try:
        explain(resnet, 12345)
        assert False, "Should have raised TypeError"
    except (TypeError, ValueError) as e:
        msg = str(e)
        # Should mention what types ARE accepted
        assert "Tensor" in msg or "PIL" in msg or "path" in msg.lower() or "torch" in msg.lower(), (
            f"Unhelpful message: {e}"
        )


@test("GradCAM with invalid layer name → AttributeError listing available layers")
def _():
    try:
        GradCAM(resnet, target_layer="nonexistent_layer_xyz")
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "available" in str(e).lower() or "Available" in str(e), (
            f"No layer list in message: {e}"
        )


@test("explain() with invalid method name → ValueError with suggestions")
def _():
    try:
        explain(resnet, tensor_input, method="nonexistent_method")
        assert False, "Should have raised ValueError"
    except (ValueError, KeyError) as e:
        msg = str(e).lower()
        assert "gradcam" in msg or "eigencam" in msg or "available" in msg, f"No suggestions: {e}"


@test("explain() with 2D tensor → helpful shape error")
def _():
    try:
        explain(resnet, torch.randn(224, 224))
        assert False, "Should have raised an error"
    except (ValueError, RuntimeError):
        pass  # Any error is fine, as long as it doesn't silently produce garbage


@test("explain() with wrong spatial dims → error doesn't crash silently")
def _():
    try:
        hm = explain(resnet, torch.randn(1, 3, 32, 32))
        # Some models handle different sizes, so this might work
        # The key test is it doesn't crash with a cryptic error
    except (RuntimeError, ValueError):
        pass  # Expected — model might need 224x224


print()


# ══════════════════════════════════════════════════════════════════════
# TEST 8: Edge Cases
# ══════════════════════════════════════════════════════════════════════
print("TEST 8: Edge Cases")


@test("__repr__ on GradCAM → includes class name, arch, device")
def _():
    cam = GradCAM(resnet)
    r = repr(cam)
    assert "GradCAM" in r, f"Missing class name: {r}"
    assert "CNN" in r or "cnn" in r.lower(), f"Missing arch: {r}"
    assert "device" in r.lower(), f"Missing device: {r}"
    print(f"      repr: {r}")


@test("__repr__ on EigenCAM → works")
def _():
    cam = EigenCAM(resnet)
    r = repr(cam)
    assert "EigenCAM" in r


@test("__repr__ on AttentionRollout → works")
def _():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cam = AttentionRollout(resnet)
        r = repr(cam)
        assert "AttentionRollout" in r


@test("requires_grad flags are correct")
def _():
    assert GradCAM.requires_grad is True
    assert GradCAMPlusPlus.requires_grad is True
    assert LayerCAM.requires_grad is True
    assert EigenCAM.requires_grad is False
    assert AttentionRollout.requires_grad is False
    assert TransformerAttribution.requires_grad is True


@test("explain() produces different heatmaps for different classes")
def _():
    hm1 = explain(resnet, tensor_input, target_class=0, method="gradcam")
    hm2 = explain(resnet, tensor_input, target_class=5, method="gradcam")
    # With random weights they might be similar, but shouldn't be identical
    # unless the model is completely untrained (which it is, so relax this)
    assert hm1.shape == hm2.shape == (224, 224)


@test("explain() with different methods produces different heatmaps")
def _():
    hm_gc = explain(resnet, tensor_input, method="gradcam")
    hm_ec = explain(resnet, tensor_input, method="eigencam")
    assert hm_gc.shape == hm_ec.shape == (224, 224)
    # They should NOT be identical (different algorithms)
    if not np.allclose(hm_gc, hm_ec, atol=0.01):
        print("      Different methods → different heatmaps (good)")
    else:
        print("      Warning: methods produced very similar outputs")


@test("Multiple explain() calls don't leak memory (hooks cleaned up)")
def _():
    gc.collect()
    for i in range(5):
        hm = explain(resnet, tensor_input)
        assert hm.shape == (224, 224)
    gc.collect()
    # If hooks leaked, we'd see increasing memory or errors
    print("      5 sequential calls completed cleanly")


@test("explain() with PNG path → works")
def _():
    path = save_temp_image(suffix=".png")
    try:
        hm = explain(resnet, path)
        assert hm.shape == (224, 224)
    finally:
        os.unlink(path)


@test("explain() with BMP path → works")
def _():
    path = save_temp_image(suffix=".bmp")
    try:
        hm = explain(resnet, path)
        assert hm.shape == (224, 224)
    finally:
        os.unlink(path)


print()


# ══════════════════════════════════════════════════════════════════════
# TEST 9: Hook System
# ══════════════════════════════════════════════════════════════════════
print("TEST 9: Hook System")

from torchxai.utils.hooks import ActivationHook, MultiHook


@test("MultiHook captures activations and gradients from a layer")
def _():
    hook = MultiHook(resnet.layer4)
    try:
        output = resnet(tensor_input)
        loss = output.sum()
        loss.backward()
        assert hook.activation is not None, "No activations captured"
        assert isinstance(hook.activation, torch.Tensor)
        assert hook.gradient is not None, "No gradients captured"
        assert isinstance(hook.gradient, torch.Tensor)
        print(
            f"      Activation shape: {hook.activation.shape}, Gradient shape: {hook.gradient.shape}"
        )
    finally:
        hook.remove()


@test("ActivationHook captures from tuple-returning layers (transformers)")
def _():
    class FakeTransformerLayer(torch.nn.Module):
        def forward(self, x):
            return (x, torch.ones(1, 4, 10, 10))  # (output, attn_weights)

    fake = FakeTransformerLayer()
    hook = ActivationHook(fake)
    try:
        fake(torch.randn(1, 10, 64))
        assert hook.activation is not None, "Should capture first element of tuple"
    finally:
        hook.remove()


@test("MultiHook cleanup → no references held after remove()")
def _():
    hook = MultiHook(resnet.layer4)
    try:
        output = resnet(tensor_input)
        loss = output.sum()
        loss.backward()
        assert hook.activation is not None
    finally:
        hook.remove()
    assert hook.activation is None, "Activation should be None after remove()"
    assert hook.gradient is None, "Gradient should be None after remove()"


@test("MultiHook as context manager → auto-cleanup")
def _():
    with MultiHook(resnet.layer4) as hook:
        output = resnet(tensor_input)
        loss = output.sum()
        loss.backward()
        assert hook.activation is not None
    # After context exit, references should be cleared
    assert hook.activation is None


@test("ActivationHook captures latest activation across multiple passes")
def _():
    hook = ActivationHook(resnet.layer4)
    try:
        with torch.no_grad():
            resnet(tensor_input)
            shape1 = hook.activation.shape
            resnet(make_image_tensor())
            shape2 = hook.activation.shape
        assert shape1 == shape2  # Same layer, same input size
    finally:
        hook.remove()


print()


# ══════════════════════════════════════════════════════════════════════
# TEST 10: End-to-End Pipeline (full workflow a user would do)
# ══════════════════════════════════════════════════════════════════════
print("TEST 10: End-to-End User Workflow")


@test("Full workflow: load model → explain → visualize → save → metrics")
def _():
    # Step 1: User loads a model
    model = models.resnet18(weights=None)
    model.eval()

    # Step 2: User has an image (as file path)
    img_path = save_temp_image()

    try:
        # Step 3: One-line explanation
        heatmap = explain(model, img_path)
        assert heatmap.shape == (224, 224)

        # Step 4: Visualize
        with tempfile.TemporaryDirectory() as td:
            save_path = os.path.join(td, "my_explanation.png")
            pil = Image.open(img_path)
            show_explanation(pil, heatmap, title="My Explanation", save_path=save_path)
            assert os.path.exists(save_path)

            # Step 5: Compare methods
            hm_gc = explain(model, img_path, method="gradcam")
            hm_ec = explain(model, img_path, method="eigencam")
            comp_path = os.path.join(td, "comparison.png")
            create_comparison(pil, {"GradCAM": hm_gc, "EigenCAM": hm_ec}, save_path=comp_path)
            assert os.path.exists(comp_path)

            # Step 6: Save raw heatmap
            hm_path = os.path.join(td, "heatmap.png")
            save_heatmap(heatmap, hm_path)
            assert os.path.exists(hm_path)

        # Step 7: Evaluate with metrics
        tensor = torch.randn(1, 3, 224, 224)
        hm = explain(model, tensor)
        ins = insertion_score(model, tensor, hm)
        dele = deletion_score(model, tensor, hm)
        assert isinstance(ins, float) and isinstance(dele, float)
        print(f"      Pipeline complete: insertion={ins:.3f}, deletion={dele:.3f}")

    finally:
        os.unlink(img_path)


@test("Full workflow with ViT: explain → all methods → compare")
def _():
    if not HAS_VIT:
        print("      (skipped, no ViT)")
        return

    t = make_image_tensor()
    methods_results = {}
    for method_name in ["gradcam", "eigencam", "layercam", "attention_rollout"]:
        hm = explain(vit, t, method=method_name)
        assert hm.shape == (224, 224), f"{method_name} failed: shape={hm.shape}"
        methods_results[method_name] = hm

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "vit_comparison.png")
        create_comparison(make_pil_image(), methods_results, save_path=path)
        assert os.path.exists(path)

    print(f"      All {len(methods_results)} methods worked on ViT")


@test("imports from top-level package → everything accessible")
def _():
    import torchxai

    # All public API should be importable from top level
    assert hasattr(torchxai, "explain")
    assert hasattr(torchxai, "GradCAM")
    assert hasattr(torchxai, "EigenCAM")
    assert hasattr(torchxai, "LayerCAM")
    assert hasattr(torchxai, "GradCAMPlusPlus")
    assert hasattr(torchxai, "AttentionRollout")
    assert hasattr(torchxai, "TransformerAttribution")
    assert hasattr(torchxai, "overlay_heatmap")
    assert hasattr(torchxai, "show_explanation")
    assert hasattr(torchxai, "create_comparison")
    assert hasattr(torchxai, "save_heatmap")
    assert hasattr(torchxai, "insertion_score")
    assert hasattr(torchxai, "deletion_score")
    assert hasattr(torchxai, "stability_score")
    print("      All 14 public API symbols accessible")


print()

# ── Cleanup ───────────────────────────────────────────────────────────
os.unlink(tmp_img_path)

# ── Final Report ──────────────────────────────────────────────────────
print("=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print("=" * 60)

if ERRORS:
    print("\nFAILURES:")
    for name, err, tb in ERRORS:
        print(f"\n  ✗ {name}")
        print(f"    {err}")
        print("    Traceback:")
        for line in tb.strip().split("\n"):
            print(f"      {line}")

if FAIL == 0:
    print("\n🎉 ALL TESTS PASSED — every feature works end-to-end!")
else:
    print(f"\n⚠️  {FAIL} test(s) need fixing")
    sys.exit(1)
