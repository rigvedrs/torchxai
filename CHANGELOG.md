# Changelog

All notable changes to `torchxai-explain` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] — 2026-07-11

### Fixed — correctness of heatmaps across model families

Verified every model × method combination end-to-end on real pretrained
weights with an automated foreground/background check
(`generate_proof_images.py`); the fixes below turn previously silent
failures (inverted, blank, or striped heatmaps) into correct ones.

- **Faithful polarity resolution** (`BaseExplainer._resolve_polarity`):
  EigenCAM's SVD projection and gradient-weighted CAMs on models with
  LayerNorm/GRN before the classifier (ViT, ConvNeXt V2, MaxViT, …) are
  only defined up to sign. The sign is now chosen with an
  insertion/deletion confidence test instead of a magnitude heuristic,
  which used to invert heatmaps onto the background.
- **Attention capture on timm models**: timm ≥ 0.9 uses fused SDPA, so
  Attention Rollout and Transformer Attribution never saw attention
  weights and silently fell back to EigenCAM. They now temporarily unfuse
  attention and hook `attn_drop`, capturing real attention matrices.
  Rollout defaults changed to `head_fusion="max"`, `discard_ratio=0.9`.
- **Correct target layers**: timm MobileNetV3/V4 and GhostNet no longer
  hook the post-global-pool `conv_head` (which produced blank 1×1 maps);
  VGG now hooks the ReLU after the last conv (post-activation, 14×14);
  MaxViT hooks the full final block; Ultralytics `DetectionModel` is now
  recognized as YOLO via module type names (previously mis-detected as a
  generic CNN, producing striped artifacts from the DFL head).
- **Dynamic token reshaping**: the ViT reshape transform derives the grid
  from the actual token count (and the model's `num_prefix_tokens`), fixing
  crashes on wrapped backbones (DINOv2 + linear head) and register tokens.
- **Model-aware input size**: `explain()` resolves the expected input size
  from the model (EVA-02 @ 448 no longer crashes with the 224 default).
- **GradCAM++ formula**: fixed the alpha denominator
  (`sum_spatial(A)·grad³`, per the paper) and a doubled `relu(grad)`
  factor that flattened or inverted maps on VGG16 and ConvNeXt.
- **ScoreCAM**: combines min-max-normalized channels (pre-activation
  target layers produced all-negative maps that the final ReLU erased) and
  clamps signed activations to the propagated positive side.
- **DINO-family default method** is now EigenCAM (register-less DINOv2's
  attention-sink artifacts hollow out rollout maps).
- Hooks unwrap nested list/tuple/NestedTensor outputs (RF-DETR backbone).

### Added
- `generate_proof_images.py` — reproducible proof-image suite with
  automated heatmap verification (`--strict` fails on regressions).
- RF-DETR proof with a real EigenCAM backbone heatmap.
- `_explain_all` now warns (instead of silently skipping) when a method
  fails for a model.

---

## [0.2.0] — 2026-04-09

### Added

#### New Explainability Methods (4)
- **ScoreCAM** — perturbation-based, gradient-free CAM (Wang et al., CVPR 2020)
- **SmoothGrad** — smoothed input gradient saliency (Smilkov et al., 2017)
- **Integrated Gradients** — axiomatic attribution with baseline interpolation (Sundararajan et al., ICML 2017)
- **RISE** — randomized input sampling for explanation (Petsiuk et al., BMVC 2018)

#### Object Detection Explainability
- `explain_detection()` — per-bounding-box explanations for YOLO, DETR, Faster R-CNN
- `DetectionExplanation` dataclass with box, class_id, confidence, heatmap
- `visualize_detections()` — annotated overlay with detection boxes and heatmaps

#### Batch Processing
- `explain_batch()` — explain multiple images with progress tracking
- `explain_directory()` — process all images in a folder with optional save
- `export_results()` — export results to CSV or JSON

#### Model Compatibility (21 architectures verified with proof images)
- **CNNs**: ResNet50, VGG16, EfficientNet-B0, EfficientNetV2-S, MobileNetV3, MobileNetV4, ConvNeXt, ConvNeXt V2, ConvNeXt-Zepto, RegNet, RepVGG, EfficientNet-H, DenseNet121, GhostNetV3
- **Transformers**: ViT-Tiny, DeiT-Tiny, Swin-Tiny, EVA-02, MaxViT, DINOv2
- **Detection**: YOLO26, YOLO11, YOLOv8, RF-DETR

#### CI/CD & Tooling
- GitHub Actions: CI (lint + test matrix Python 3.9–3.12), PyPI publish (OIDC), docs deploy
- `ruff.toml`, `mypy.ini`, `.pre-commit-config.yaml` for code quality
- PEP 561 `py.typed` marker for type checker support

#### Community Infrastructure
- CODE_OF_CONDUCT.md (Contributor Covenant v2.1)
- SECURITY.md with vulnerability reporting policy
- PULL_REQUEST_TEMPLATE.md with checklist
- Enhanced issue templates (bug report with system info, feature request, model support)
- FUNDING.yml for GitHub Sponsors

#### Documentation Website
- 6-page documentation site with dark mode, syntax highlighting, real heatmap images
- Pages: Home, Getting Started, Methods, API Reference, Examples, Model Compatibility
- 29 proof images from real pretrained models

#### Benchmark Suite
- `benchmarks/benchmark_methods.py` — speed and quality benchmarks across models and methods

### Changed
- `explain()` now accepts `method="auto"` as equivalent to `method=None`
- `np.trapz` replaced with `np.trapezoid` for NumPy 2.0+ compatibility
- `tostring_rgb()` replaced with `buffer_rgba()` for matplotlib 3.8+ compatibility
- Improved error message for unsupported image types (now lists all accepted formats)

### Fixed
- `stability_score()` crashed with "Can't call numpy() on Tensor that requires grad"
- `matplotlib.pyplot.show()` crashed on headless servers (auto Agg backend detection)

---

## [0.1.0] — 2026-04-09

Initial public release.

### Added

#### Core API
- `explain()` — one-line explainability with auto architecture detection
- Accepts: torch.Tensor, PIL.Image, numpy array, file path (str/Path)
- Auto-selects best method based on model architecture

#### Explainability Methods (6)
- **GradCAM** — gradient-weighted class activation mapping
- **GradCAM++** — improved localization for multiple objects
- **EigenCAM** — PCA-based, gradient-free
- **LayerCAM** — fine-grained spatial detail
- **Attention Rollout** — attention flow through transformer layers
- **Transformer Attribution** — class-specific ViT explanation

#### Architecture Detection
- Auto-detects: CNN, ViT, Swin, CLIP, DETR, YOLO, DINO
- Auto-resolves optimal target layer for each architecture

#### Visualization
- `overlay_heatmap()`, `show_explanation()`, `create_comparison()`, `save_heatmap()`
- Headless server support (auto Agg backend)

#### Metrics
- `insertion_score()`, `deletion_score()`, `stability_score()`

#### Utilities
- `preprocess_image()`, `load_image()`, `tensor_to_numpy()`, `denormalize()`, `normalize_heatmap()`
- Hook classes: `ActivationHook`, `GradientHook`, `MultiHook`, `AttentionHook`

### Dependencies
- **Required**: `torch>=1.9`, `torchvision>=0.10`, `numpy>=1.21`, `Pillow>=8.0`, `matplotlib>=3.4`
- **Optional**: `timm>=0.9`, `ultralytics>=8.0`

---

[0.3.0]: https://github.com/rigvedrs/torchxai/releases/tag/v0.3.0
[0.2.0]: https://github.com/rigvedrs/torchxai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rigvedrs/torchxai/releases/tag/v0.1.0
