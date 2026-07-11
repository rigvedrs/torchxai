"""
Benchmark suite for torchxai methods.

Measures speed, memory usage, and quality metrics across all methods
on different model architectures and input sizes.

Usage:
    python benchmarks/benchmark_methods.py
    python benchmarks/benchmark_methods.py --methods gradcam eigencam --sizes 224 384
"""

import argparse
import gc
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torchvision.models as models

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torchxai import explain
from torchxai.metrics.fidelity import deletion_score, insertion_score


def measure_time(fn, n_runs=5, warmup=2):
    """Measure average execution time in milliseconds."""
    # Warmup
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(n_runs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        fn()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


def measure_memory(fn):
    """Measure peak GPU memory usage in MB (if CUDA available)."""
    if not torch.cuda.is_available():
        fn()
        return {"peak_mb": 0, "device": "cpu"}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {"peak_mb": peak, "device": "cuda"}


def run_benchmark(args):
    """Run the full benchmark suite."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Models to benchmark
    model_configs = [
        ("ResNet50", lambda: models.resnet50(weights=None).eval().to(device)),
    ]

    try:
        import timm

        model_configs.append(
            (
                "ViT-Tiny",
                lambda: (
                    timm.create_model("vit_tiny_patch16_224", pretrained=False).eval().to(device)
                ),
            )
        )
    except ImportError:
        print("timm not installed — skipping ViT benchmarks\n")

    # Methods to benchmark
    all_methods = ["gradcam", "eigencam", "layercam", "gradcam_pp"]
    methods = args.methods if args.methods else all_methods

    # Input sizes
    sizes = args.sizes if args.sizes else [224]

    print("=" * 80)
    print(f"{'SPEED BENCHMARK':^80}")
    print("=" * 80)

    for model_name, model_fn in model_configs:
        model = model_fn()
        print(f"\n  Model: {model_name}")
        print(f"  {'Method':<20} {'Size':>6} {'Mean (ms)':>12} {'Std':>8} {'Min':>8} {'Max':>8}")
        print("  " + "-" * 60)

        for size in sizes:
            tensor = torch.randn(1, 3, size, size).to(device)
            for method in methods:
                try:
                    fn = lambda: explain(model, tensor, method=method)
                    result = measure_time(fn, n_runs=args.runs)
                    print(
                        f"  {method:<20} {size:>6} "
                        f"{result['mean_ms']:>10.1f}ms "
                        f"{result['std_ms']:>6.1f} "
                        f"{result['min_ms']:>6.1f} "
                        f"{result['max_ms']:>6.1f}"
                    )
                except Exception as e:
                    print(f"  {method:<20} {size:>6} {'ERROR':>12} {str(e)[:30]}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Quality metrics
    if not args.speed_only:
        print("\n" + "=" * 80)
        print(f"{'QUALITY METRICS':^80}")
        print("=" * 80)

        model = models.resnet50(weights=None).eval().to(device)
        tensor = torch.randn(1, 3, 224, 224).to(device)

        print("\n  Model: ResNet50 (random weights), Input: 224x224")
        print(f"  {'Method':<20} {'Insertion ↑':>12} {'Deletion ↓':>12}")
        print("  " + "-" * 46)

        for method in methods:
            try:
                hm = explain(model, tensor, method=method)
                ins = insertion_score(model, tensor, hm, steps=20)
                dele = deletion_score(model, tensor, hm, steps=20)
                print(f"  {method:<20} {ins:>12.4f} {dele:>12.4f}")
            except Exception as e:
                print(f"  {method:<20} {'ERROR':>12} {str(e)[:30]}")

    print("\n" + "=" * 80)
    print("Benchmark complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark torchxai methods")
    parser.add_argument("--methods", nargs="+", help="Methods to benchmark")
    parser.add_argument("--sizes", nargs="+", type=int, help="Input sizes")
    parser.add_argument("--runs", type=int, default=5, help="Number of timing runs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--speed-only", action="store_true", help="Skip quality metrics")
    args = parser.parse_args()
    run_benchmark(args)
