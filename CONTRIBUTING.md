# Contributing to torchxai

Thank you for your interest in contributing to torchxai.

## Development Setup

```bash
git clone https://github.com/rigvedrs/torchxai.git
cd torchxai
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check torchxai/
ruff format torchxai/
```

## Adding a New Explainability Method

1. Create a new file in `torchxai/methods/` (e.g., `your_method.py`)
2. Subclass `BaseExplainer` and implement `_compute_cam()`
3. Add the method to `METHOD_MAP` in `torchxai/api.py`
4. Add tests in `tests/test_core.py`
5. Update the README method table

### Example skeleton

```python
from torchxai.methods.base import BaseExplainer

class YourMethod(BaseExplainer):
    def _compute_cam(self, input_tensor, target_class):
        # Your implementation here
        # Return numpy array (h, w) — not necessarily normalized
        ...
```

## Adding a New Architecture

1. Add a new `ArchType` variant in `torchxai/models/registry.py`
2. Add detection logic in `detect_architecture()`
3. Add layer resolution in `resolve_target_layer()`
4. Add auto-method selection in `AUTO_METHOD_MAP` in `torchxai/api.py`
5. Add tests

## Pull Request Guidelines

- One feature or fix per PR
- Include tests for new functionality
- Update README if user-facing behavior changes
- Keep commits focused and well-described

## Reporting Issues

Open an issue on GitHub with:
- Your Python/PyTorch version
- The model you're using
- A minimal reproducible example
- The full error traceback
