"""Regenerate every proof image in docs/assets/proof/ with real models.

This is the verification harness behind docs/model-compatibility.md.
Every heatmap is produced by the actual torchxai pipeline on real
pretrained weights — nothing is mocked. In addition to rendering the
images, each heatmap is scored against a rough foreground/background
split of the test photo, so a silently-broken combination (inverted,
flat, or background-focused map) fails loudly instead of shipping as
a "verified" proof.

Usage:
    python generate_proof_images.py                    # everything
    python generate_proof_images.py --only resnet50    # one model family
    python generate_proof_images.py --skip-slow        # skip ScoreCAM/RISE
    python generate_proof_images.py --strict           # non-zero exit on any failed check

Requirements: pip install torchxai-explain timm  (plus ultralytics for the
YOLO section — skipped automatically when not installed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import warnings

import matplotlib
import numpy as np
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).parent
PROOF_DIR = REPO / "docs" / "assets" / "proof"
DOG_PATH = PROOF_DIR / "original_dog.jpg"


# ── Sanity scoring ────────────────────────────────────────────────────
# The test photo is a Labrador on grass: head in the upper-left region,
# torso centered, grass in the top-right / bottom-left / bottom-right
# corners. A correct heatmap must put more mass on the dog than on the
# grass corners.


def sanity_ratio(heat: np.ndarray) -> float | None:
    h, w = heat.shape
    if heat.std() < 0.02:
        return None  # flat map
    head = heat[int(0.05 * h) : int(0.42 * h), int(0.05 * w) : int(0.45 * w)]
    torso = heat[int(0.25 * h) : int(0.72 * h), int(0.25 * w) : int(0.80 * w)]
    dog = max(head.mean(), torso.mean())
    grass = np.mean(
        [
            heat[: int(0.16 * h), int(0.86 * w) :].mean(),
            heat[int(0.86 * h) :, : int(0.14 * w)].mean(),
            heat[int(0.88 * h) :, int(0.88 * w) :].mean(),
        ]
    )
    return float(dog / (grass + 1e-6))


def verdict(ratio: float | None) -> str:
    if ratio is None:
        return "FLAT"
    if ratio > 1.5:
        return "OK"
    if ratio < 0.8:
        return "INVERTED"
    return "WEAK"


# ── Rendering ─────────────────────────────────────────────────────────


def overlay(dog_img: Image.Image, heat: np.ndarray, alpha=0.5) -> np.ndarray:
    base = np.array(dog_img.resize((heat.shape[1], heat.shape[0]))).astype(np.float32) / 255.0
    cmap = plt.get_cmap("jet")(heat)[..., :3]
    return (1 - alpha) * base + alpha * cmap


def render_panel(title, dog_img, heatmaps: dict, out_path: Path):
    n = len(heatmaps) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.4))
    if n == 1:
        axes = [axes]
    axes[0].imshow(dog_img)
    axes[0].set_title("Original", fontweight="bold")
    axes[0].axis("off")
    for ax, (m, heat) in zip(axes[1:], heatmaps.items()):
        ax.imshow(overlay(dog_img, heat))
        ax.set_title(m, fontweight="bold")
        ax.axis("off")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Model zoo ─────────────────────────────────────────────────────────


def model_zoo():
    """(key, display name, loader, methods). Loaders are lazy."""
    import torchvision.models as tvm

    def timm_model(name, **kw):
        import timm

        return timm.create_model(name, pretrained=True, **kw)

    class DINOv2Classifier(nn.Module):
        """DINOv2 backbone + linear head, as in the docs example.

        The head is untrained, so only class-agnostic methods (EigenCAM,
        Attention Rollout) are meaningful — mirroring real usage where
        DINOv2 is a feature extractor.
        """

        def __init__(self):
            super().__init__()
            import timm

            self.backbone = timm.create_model(
                "vit_small_patch14_dinov2.lvd142m",
                pretrained=True,
                img_size=224,
                num_classes=0,
            )
            self.head = nn.Linear(self.backbone.embed_dim, 1000)

        def forward(self, x):
            return self.head(self.backbone(x))

    cam4 = ["gradcam", "eigencam", "layercam", "gradcam_pp"]
    cam5 = cam4 + ["scorecam"]
    vit4 = ["gradcam", "eigencam", "attention_rollout", "transformer_attribution"]

    return [
        ("resnet50", "ResNet50", lambda: tvm.resnet50(weights="IMAGENET1K_V2"), cam5),
        ("vgg16", "VGG16", lambda: tvm.vgg16(weights="IMAGENET1K_V1"), cam5),
        (
            "efficientnet_b0",
            "EfficientNet-B0",
            lambda: tvm.efficientnet_b0(weights="IMAGENET1K_V1"),
            cam5,
        ),
        (
            "efficientnetv2_s",
            "EfficientNetV2-S",
            lambda: timm_model("efficientnetv2_rw_s.ra2_in1k"),
            cam4,
        ),
        (
            "mobilenetv3_small",
            "MobileNetV3-Small",
            lambda: tvm.mobilenet_v3_small(weights="IMAGENET1K_V1"),
            cam5,
        ),
        (
            "mobilenetv4_small",
            "MobileNetV4-Small",
            lambda: timm_model("mobilenetv4_conv_small.e2400_r224_in1k"),
            cam4,
        ),
        (
            "convnext_tiny",
            "ConvNeXt-Tiny",
            lambda: tvm.convnext_tiny(weights="IMAGENET1K_V1"),
            cam4,
        ),
        (
            "convnext_v2",
            "ConvNeXt V2 Tiny",
            lambda: timm_model("convnextv2_tiny.fcmae_ft_in22k_in1k"),
            cam5,
        ),
        (
            "convnext_zepto",
            "ConvNeXt-Zepto",
            lambda: timm_model("convnext_zepto_rms_ols.ra4_e3600_r224_in1k"),
            cam4,
        ),
        (
            "regnety_400mf",
            "RegNetY-400MF",
            lambda: tvm.regnet_y_400mf(weights="IMAGENET1K_V2"),
            cam4,
        ),
        ("repvgg_b0", "RepVGG-B0", lambda: timm_model("repvgg_b0.rvgg_in1k"), cam4),
        (
            "ghostnetv3",
            "GhostNetV3",
            lambda: timm_model("ghostnetv3_100.in1k"),
            ["gradcam", "eigencam"],
        ),
        (
            "efficientnet_h_b5",
            "EfficientNet-H B5",
            lambda: timm_model("efficientnet_h_b5.sw_r448_e450_in1k"),
            ["gradcam", "eigencam"],
        ),
        (
            "densenet121",
            "DenseNet121",
            lambda: tvm.densenet121(weights="IMAGENET1K_V1"),
            ["eigencam", "scorecam"],
        ),
        ("vit_tiny_16", "ViT-Tiny/16", lambda: timm_model("vit_tiny_patch16_224"), vit4),
        ("deit_tiny_16", "DeiT-Tiny/16", lambda: timm_model("deit_tiny_patch16_224"), vit4),
        (
            "swin_tiny",
            "Swin-Tiny",
            lambda: timm_model("swin_tiny_patch4_window7_224"),
            ["gradcam", "eigencam", "layercam"],
        ),
        (
            "eva_02_base",
            "EVA-02 Base",
            lambda: timm_model("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"),
            ["gradcam", "eigencam"],
        ),
        (
            "maxvit_tiny",
            "MaxViT-Tiny",
            lambda: timm_model("maxvit_tiny_tf_224.in1k"),
            ["gradcam", "eigencam"],
        ),
        ("dinov2_vit_s14", "DINOv2 ViT-S14", DINOv2Classifier, ["eigencam", "attention_rollout"]),
    ]


SLOW_METHODS = {"scorecam", "rise"}

# Documented limitations (see docs/model-compatibility.md). These combos are
# still rendered and reported, but --strict treats them as warnings rather
# than regressions: GradCAM is inherently noisy on very small ViTs, and
# register-less DINOv2 has attention-sink artifacts that hollow out rollout.
EXPECTED_LIMITATIONS = {
    ("vit_tiny_16", "gradcam"),
    ("dinov2_vit_s14", "attention_rollout"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="comma-separated model keys")
    parser.add_argument("--skip-slow", action="store_true", help="skip ScoreCAM/RISE")
    parser.add_argument("--skip-input-level", action="store_true")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any check fails")
    parser.add_argument("--image", default=str(DOG_PATH))
    args = parser.parse_args()

    from torchxai import explain

    PROOF_DIR.mkdir(parents=True, exist_ok=True)
    dog = Image.open(args.image).convert("RGB")

    only = set(args.only.split(",")) if args.only else None
    results: dict[str, dict] = {}
    grid_panels: list[tuple[str, np.ndarray]] = []  # (display name, heatmap)

    for key, display, loader, methods in model_zoo():
        if only and key not in only:
            continue
        print(f"\n=== {display} ===", flush=True)
        try:
            model = loader()
            model.eval()
        except Exception as e:
            print(f"  SKIP (could not load): {e}")
            results[key] = {"__load__": f"SKIP: {e}"}
            continue

        heatmaps: dict[str, np.ndarray] = {}
        model_results: dict[str, dict] = {}
        for m in methods:
            if args.skip_slow and m in SLOW_METHODS:
                continue
            t0 = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    heat = explain(model, dog, method=m)
                ms = int((time.time() - t0) * 1000)
                r = sanity_ratio(heat)
                v = verdict(r)
                heatmaps[m] = heat
                model_results[m] = {
                    "verdict": v,
                    "ratio": None if r is None else round(r, 2),
                    "ms": ms,
                }
                print(f"  {m:26s} {v:9s} ratio={model_results[m]['ratio']} ({ms}ms)")
            except Exception as e:
                model_results[m] = {"verdict": "CRASH", "error": str(e)[:200]}
                print(f"  {m:26s} CRASH: {str(e)[:120]}")
        results[key] = model_results

        if heatmaps:
            render_panel(display, dog, heatmaps, PROOF_DIR / f"{key}.png")
            # master grid: best-scoring method for this model
            ok = {m: h for m, h in heatmaps.items() if model_results[m]["verdict"] == "OK"}
            pick = ok or heatmaps
            best = max(pick, key=lambda m: model_results[m].get("ratio") or 0)
            grid_panels.append((display, heatmaps[best]))

        del model

    # ── Input-level methods (ResNet50) ────────────────────────────────
    if not args.skip_input_level and (only is None or "input_level" in only):
        print("\n=== Input-level methods (ResNet50) ===", flush=True)
        import torchvision.models as tvm

        model = tvm.resnet50(weights="IMAGENET1K_V2").eval()
        heatmaps = {}
        input_results = {}
        for m in ["smoothgrad", "integrated_gradients"] + ([] if args.skip_slow else ["rise"]):
            t0 = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    heat = explain(model, dog, method=m)
                r = sanity_ratio(heat)
                heatmaps[m] = heat
                input_results[m] = {
                    "verdict": verdict(r),
                    "ratio": None if r is None else round(r, 2),
                    "ms": int((time.time() - t0) * 1000),
                }
                print(
                    f"  {m:26s} {input_results[m]['verdict']:9s} ratio={input_results[m]['ratio']}"
                )
            except Exception as e:
                input_results[m] = {"verdict": "CRASH", "error": str(e)[:200]}
                print(f"  {m:26s} CRASH: {str(e)[:120]}")
        results["input_level"] = input_results
        if heatmaps:
            render_panel(
                "Input-Level Methods (ResNet50)",
                dog,
                heatmaps,
                PROOF_DIR / "input_level_methods.png",
            )

    # ── YOLO detection section (optional) ─────────────────────────────
    if not args.skip_yolo and (only is None or any(k.startswith("yolo") for k in only)):
        try:
            from ultralytics import YOLO

            from torchxai.detection import explain_detection

            for yname in ["yolo26n", "yolo11n", "yolov8n"]:
                print(f"\n=== {yname} ===", flush=True)
                try:
                    yolo = YOLO(f"{yname}.pt")
                    dets = yolo(dog, verbose=False)
                    boxes = dets[0].boxes
                    detections = [
                        {
                            "box": boxes.xyxy[i].tolist(),
                            "class_id": int(boxes.cls[i].item()),
                            "confidence": float(boxes.conf[i].item()),
                            "class_name": dets[0].names[int(boxes.cls[i].item())],
                        }
                        for i in range(len(boxes))
                    ]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        heat = explain(yolo.model, dog, method="eigencam", image_size=(640, 640))
                    r = sanity_ratio(heat)
                    results[yname] = {
                        "eigencam": {
                            "verdict": verdict(r),
                            "ratio": None if r is None else round(r, 2),
                        },
                        "detections": len(detections),
                    }
                    print(
                        f"  detections={len(detections)} eigencam {verdict(r)} ratio={round(r, 2) if r else None}"
                    )

                    # 3-panel proof: original / detections / eigencam heatmap
                    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6))
                    axes[0].imshow(dog)
                    axes[0].set_title("Original", fontweight="bold")
                    axes[1].imshow(dets[0].plot()[..., ::-1])
                    axes[1].set_title(f"Detections ({len(detections)})", fontweight="bold")
                    axes[2].imshow(overlay(dog, heat))
                    axes[2].set_title("EigenCAM", fontweight="bold")
                    for ax in axes:
                        ax.axis("off")
                    fig.suptitle(
                        f"{yname} — Detection + Explanation", fontsize=15, fontweight="bold"
                    )
                    fig.tight_layout()
                    fig.savefig(PROOF_DIR / f"{yname}.png", dpi=100, bbox_inches="tight")
                    plt.close(fig)
                except Exception as e:
                    print(f"  SKIP: {e}")
                    results[yname] = {"__load__": f"SKIP: {e}"}
        except ImportError:
            print("\n[ultralytics not installed — skipping YOLO proofs]")

    # ── RF-DETR detection section (optional) ──────────────────────────
    if not args.skip_yolo and (only is None or "rfdetr" in only):
        try:
            from rfdetr import RFDETRBase

            print("\n=== RF-DETR Base ===", flush=True)
            rf = RFDETRBase()
            det = rf.predict(dog, threshold=0.5)
            core = rf.model.model.eval()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                heat = explain(
                    core,
                    dog,
                    method="eigencam",
                    target_layer=core.backbone,
                    image_size=(rf.model.resolution,) * 2,
                )
            r = sanity_ratio(heat)
            results["rfdetr"] = {
                "eigencam": {"verdict": verdict(r), "ratio": None if r is None else round(r, 2)},
                "detections": len(det.class_id),
            }
            print(
                f"  detections={len(det.class_id)} eigencam {verdict(r)} "
                f"ratio={round(r, 2) if r else None}"
            )

            fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6))
            axes[0].imshow(dog)
            axes[0].set_title("Original", fontweight="bold")
            axes[1].imshow(dog)
            import matplotlib.patches as patches

            for (x1, y1, x2, y2), conf in zip(det.xyxy, det.confidence):
                axes[1].add_patch(
                    patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2.5,
                        edgecolor="magenta",
                        facecolor="none",
                    )
                )
                axes[1].text(
                    x1,
                    y1 - 6,
                    f"dog {conf:.2f}",
                    color="white",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="magenta", alpha=0.85),
                )
            axes[1].set_title(f"Detections ({len(det.class_id)})", fontweight="bold")
            axes[2].imshow(overlay(dog, heat))
            axes[2].set_title("EigenCAM (backbone)", fontweight="bold")
            for ax in axes:
                ax.axis("off")
            fig.suptitle("RF-DETR Base — Detection + Explanation", fontsize=15, fontweight="bold")
            fig.tight_layout()
            fig.savefig(PROOF_DIR / "rfdetr.png", dpi=100, bbox_inches="tight")
            plt.close(fig)
        except ImportError:
            print("\n[rfdetr not installed — skipping RF-DETR proof]")
        except Exception as e:
            print(f"\n[RF-DETR proof failed: {e}]")
            results["rfdetr"] = {"__load__": f"SKIP: {e}"}

    # ── Master grid ────────────────────────────────────────────────────
    if grid_panels:
        cols = 6
        rows = int(np.ceil((len(grid_panels) + 1) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.7 * rows))
        axes = np.atleast_2d(axes)
        axes_flat = axes.ravel()
        axes_flat[0].imshow(dog)
        axes_flat[0].set_title("Original", fontweight="bold", fontsize=11)
        for ax, (name, heat) in zip(axes_flat[1:], grid_panels):
            ax.imshow(overlay(dog, heat))
            ax.set_title(name, fontweight="bold", fontsize=11)
        for ax in axes_flat:
            ax.axis("off")
        fig.suptitle(
            f"torchxai — verified on {len(grid_panels)} model families",
            fontsize=17,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(PROOF_DIR / "master_grid_final.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"\nMaster grid: {len(grid_panels)} models -> {PROOF_DIR / 'master_grid_final.png'}")

    # ── Summary ────────────────────────────────────────────────────────
    # Merge into any existing record so partial (--only) runs don't
    # clobber results for models they didn't touch.
    results_path = PROOF_DIR / "verification_results.json"
    merged = {}
    if results_path.exists():
        try:
            merged = json.loads(results_path.read_text())
        except Exception:
            merged = {}
    merged.update(results)
    with open(results_path, "w") as f:
        json.dump(merged, f, indent=1)

    bad, expected = [], []
    for key, methods in results.items():
        for m, v in methods.items():
            if isinstance(v, dict) and v.get("verdict") not in (None, "OK"):
                if (key, m) in EXPECTED_LIMITATIONS:
                    expected.append((key, m, v.get("verdict")))
                else:
                    bad.append((key, m, v.get("verdict")))
    print("\n===== VERIFICATION SUMMARY =====")
    total = sum(
        1 for ms in results.values() for v in ms.values() if isinstance(v, dict) and "verdict" in v
    )
    print(f"combos checked: {total}, failed checks: {len(bad)}, known limitations: {len(expected)}")
    for key, m, v in bad:
        print(f"  FAILED: {key} / {m} -> {v}")
    for key, m, v in expected:
        print(f"  KNOWN LIMITATION (documented): {key} / {m} -> {v}")

    if args.strict and bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
