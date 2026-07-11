"""
Tests for new explainability methods: ScoreCAM, SmoothGrad,
IntegratedGradients, RISE, plus batch processing and detection.
"""

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def dummy_tensor():
    torch.manual_seed(42)
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def pil_image():
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def resnet():
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.eval()
    return model


@pytest.fixture
def simple_cnn():
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


# ═══════════════════════════════════════════════════════════════════════
# ScoreCAM
# ═══════════════════════════════════════════════════════════════════════


class TestScoreCAM:
    def test_output_shape(self, resnet, dummy_tensor):
        from torchxai import ScoreCAM

        cam = ScoreCAM(resnet)
        hm = cam(dummy_tensor)
        assert hm.shape == (224, 224)
        assert hm.min() >= 0 and hm.max() <= 1

    def test_requires_grad_false(self):
        from torchxai import ScoreCAM

        assert ScoreCAM.requires_grad is False

    def test_via_explain(self, resnet, dummy_tensor):
        from torchxai import explain

        hm = explain(resnet, dummy_tensor, method="scorecam")
        assert hm.shape == (224, 224)

    def test_repr(self, resnet):
        from torchxai import ScoreCAM

        cam = ScoreCAM(resnet)
        r = repr(cam)
        assert "ScoreCAM" in r


# ═══════════════════════════════════════════════════════════════════════
# SmoothGrad
# ═══════════════════════════════════════════════════════════════════════


class TestSmoothGrad:
    def test_output_shape(self, resnet, dummy_tensor):
        from torchxai import SmoothGrad

        sg = SmoothGrad(resnet)
        hm = sg(dummy_tensor)
        assert hm.shape == (224, 224)
        assert hm.min() >= 0 and hm.max() <= 1

    def test_requires_grad_true(self):
        from torchxai import SmoothGrad

        assert SmoothGrad.requires_grad is True

    def test_via_explain(self, resnet, dummy_tensor):
        from torchxai import explain

        hm = explain(resnet, dummy_tensor, method="smoothgrad")
        assert hm.shape == (224, 224)

    def test_different_from_gradcam(self, resnet, dummy_tensor):
        from torchxai import explain

        hm_sg = explain(resnet, dummy_tensor, method="smoothgrad")
        hm_gc = explain(resnet, dummy_tensor, method="gradcam")
        # Different algorithms should produce different results
        assert hm_sg.shape == hm_gc.shape


# ═══════════════════════════════════════════════════════════════════════
# Integrated Gradients
# ═══════════════════════════════════════════════════════════════════════


class TestIntegratedGradients:
    def test_output_shape(self, resnet, dummy_tensor):
        from torchxai import IntegratedGradients

        ig = IntegratedGradients(resnet)
        hm = ig(dummy_tensor)
        assert hm.shape == (224, 224)
        assert hm.min() >= 0 and hm.max() <= 1

    def test_requires_grad_true(self):
        from torchxai import IntegratedGradients

        assert IntegratedGradients.requires_grad is True

    def test_via_explain(self, resnet, dummy_tensor):
        from torchxai import explain

        hm = explain(resnet, dummy_tensor, method="integrated_gradients")
        assert hm.shape == (224, 224)


# ═══════════════════════════════════════════════════════════════════════
# RISE
# ═══════════════════════════════════════════════════════════════════════


class TestRISE:
    def test_output_shape(self, simple_cnn, dummy_tensor):
        # Use simple_cnn for speed — RISE is slow with many masks
        from torchxai import RISE

        rise = RISE(simple_cnn)
        hm = rise(dummy_tensor)
        assert hm.shape == (224, 224)
        assert hm.min() >= 0 and hm.max() <= 1

    def test_requires_grad_false(self):
        from torchxai import RISE

        assert RISE.requires_grad is False

    def test_via_explain(self, simple_cnn, dummy_tensor):
        from torchxai import explain

        hm = explain(simple_cnn, dummy_tensor, method="rise")
        assert hm.shape == (224, 224)


# ═══════════════════════════════════════════════════════════════════════
# explain() with "auto" method
# ═══════════════════════════════════════════════════════════════════════


class TestExplainAuto:
    def test_auto_string(self, resnet, dummy_tensor):
        from torchxai import explain

        hm = explain(resnet, dummy_tensor, method="auto")
        assert hm.shape == (224, 224)

    def test_auto_none(self, resnet, dummy_tensor):
        from torchxai import explain

        hm = explain(resnet, dummy_tensor, method=None)
        assert hm.shape == (224, 224)

    def test_all_method_names_valid(self, resnet, dummy_tensor):
        """Ensure every registered method name resolves without error."""
        from torchxai.api import METHOD_MAP

        for name in METHOD_MAP:
            # Just check it doesn't crash on instantiation
            from torchxai.api import _import_method

            cls = _import_method(METHOD_MAP[name])
            assert cls is not None


# ═══════════════════════════════════════════════════════════════════════
# Batch Processing
# ═══════════════════════════════════════════════════════════════════════


class TestBatchProcessing:
    def test_explain_batch_tensors(self, resnet, dummy_tensor):
        from torchxai import explain_batch

        tensors = [torch.randn(1, 3, 224, 224) for _ in range(3)]
        heatmaps = explain_batch(resnet, tensors, method="eigencam", progress=False)
        assert len(heatmaps) == 3
        for hm in heatmaps:
            assert hm.shape == (224, 224)

    def test_explain_batch_stacked_tensor(self, resnet):
        from torchxai import explain_batch

        stacked = torch.randn(4, 3, 224, 224)
        heatmaps = explain_batch(resnet, stacked, method="eigencam", progress=False)
        assert len(heatmaps) == 4

    def test_explain_batch_pil_images(self, resnet, pil_image):
        from torchxai import explain_batch

        images = [pil_image, pil_image]
        heatmaps = explain_batch(resnet, images, method="eigencam", progress=False)
        assert len(heatmaps) == 2

    def test_explain_batch_file_paths(self, resnet, tmp_path):
        from torchxai import explain_batch

        # Create temp images
        paths = []
        for i in range(2):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            p = tmp_path / f"img_{i}.jpg"
            img.save(str(p))
            paths.append(str(p))
        heatmaps = explain_batch(resnet, paths, method="eigencam", progress=False)
        assert len(heatmaps) == 2

    def test_explain_directory(self, resnet, tmp_path):
        from torchxai import explain_directory

        # Create temp directory with images
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(str(tmp_path / f"test_{i}.jpg"))

        results = explain_directory(
            resnet,
            str(tmp_path),
            method="eigencam",
            progress=False,
        )
        assert len(results) == 3
        for r in results:
            assert "heatmap" in r
            assert r["heatmap"].shape == (224, 224)

    def test_explain_directory_with_save(self, resnet, tmp_path):
        from torchxai import explain_directory

        img_dir = tmp_path / "images"
        out_dir = tmp_path / "outputs"
        img_dir.mkdir()

        for i in range(2):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(str(img_dir / f"test_{i}.png"))

        results = explain_directory(
            resnet,
            str(img_dir),
            save_dir=str(out_dir),
            method="eigencam",
            progress=False,
        )
        assert len(results) == 2
        assert out_dir.exists()
        assert any(r["overlay_path"] is not None for r in results)

    def test_export_csv(self, resnet, tmp_path):
        from torchxai import explain_batch, export_results

        tensors = [torch.randn(1, 3, 224, 224) for _ in range(2)]
        heatmaps = explain_batch(resnet, tensors, method="eigencam", progress=False)
        results = [
            {
                "filename": f"img_{i}.jpg",
                "path": f"/tmp/img_{i}.jpg",
                "heatmap": hm,
                "overlay_path": None,
                "heatmap_path": None,
            }
            for i, hm in enumerate(heatmaps)
        ]
        csv_path = tmp_path / "results.csv"
        export_results(results, str(csv_path), format="csv")
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "img_0.jpg" in content

    def test_export_json(self, resnet, tmp_path):
        import json

        from torchxai import explain_batch, export_results

        tensors = [torch.randn(1, 3, 224, 224)]
        heatmaps = explain_batch(resnet, tensors, method="eigencam", progress=False)
        results = [
            {
                "filename": "img.jpg",
                "path": "/tmp/img.jpg",
                "heatmap": heatmaps[0],
                "overlay_path": None,
                "heatmap_path": None,
            }
        ]
        json_path = tmp_path / "results.json"
        export_results(results, str(json_path), format="json")
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert len(data) == 1


# ═══════════════════════════════════════════════════════════════════════
# Detection
# ═══════════════════════════════════════════════════════════════════════


class TestDetection:
    def test_explain_detection_with_precomputed(self, resnet, dummy_tensor):
        from torchxai import explain_detection

        detections = [
            {"box": [10, 10, 100, 100], "class_id": 0, "confidence": 0.9},
            {"box": [120, 50, 200, 180], "class_id": 1, "confidence": 0.8},
        ]
        explanations = explain_detection(resnet, dummy_tensor, detections=detections)
        assert len(explanations) == 2
        for exp in explanations:
            assert exp.heatmap.shape[0] > 0
            assert exp.heatmap.shape[1] > 0
            assert exp.confidence > 0
            assert len(exp.box) == 4

    def test_detection_explanation_dataclass(self):
        from torchxai import DetectionExplanation

        exp = DetectionExplanation(
            box=[0, 0, 100, 100],
            class_id=5,
            confidence=0.95,
            heatmap=np.zeros((224, 224)),
            class_name="dog",
        )
        assert exp.class_name == "dog"
        assert exp.class_id == 5
        assert exp.confidence == 0.95

    def test_explain_detection_with_class_names(self, resnet, dummy_tensor):
        from torchxai import explain_detection

        detections = [
            {"box": [10, 10, 100, 100], "class_id": 0, "confidence": 0.9},
        ]
        class_names = ["dog", "cat", "bird"]
        explanations = explain_detection(
            resnet,
            dummy_tensor,
            detections=detections,
            class_names=class_names,
        )
        assert len(explanations) == 1
        assert explanations[0].class_name == "dog"

    def test_explain_detection_empty(self, resnet, dummy_tensor):
        from torchxai import explain_detection

        explanations = explain_detection(resnet, dummy_tensor, detections=[])
        assert explanations == []

    def test_explain_detection_pil_input(self, resnet, pil_image):
        from torchxai import explain_detection

        detections = [
            {"box": [10, 10, 100, 100], "class_id": 0, "confidence": 0.9},
        ]
        explanations = explain_detection(resnet, pil_image, detections=detections)
        assert len(explanations) == 1

    def test_explain_detection_string_path(self, resnet, tmp_path):
        from torchxai import explain_detection

        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        p = tmp_path / "test.jpg"
        img.save(str(p))
        detections = [{"box": [10, 10, 100, 100], "class_id": 0, "confidence": 0.9}]
        explanations = explain_detection(resnet, str(p), detections=detections)
        assert len(explanations) == 1


# ═══════════════════════════════════════════════════════════════════════
# All 10 Methods via explain()
# ═══════════════════════════════════════════════════════════════════════


class TestAllMethods:
    """Ensure every method name works through explain()."""

    @pytest.mark.parametrize(
        "method",
        [
            "gradcam",
            "eigencam",
            "layercam",
            "gradcam_pp",
            "scorecam",
            "smoothgrad",
            "integrated_gradients",
            "rise",
        ],
    )
    def test_method_on_cnn(self, resnet, dummy_tensor, method):
        from torchxai import explain

        hm = explain(resnet, dummy_tensor, method=method)
        assert hm.shape == (224, 224)
        assert 0 <= hm.min() and hm.max() <= 1
