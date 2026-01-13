# AGENTS.md - TinyTorch Development Guide for AI Agents

This document provides essential information for AI coding agents working on the TinyTorch codebase.

## Project Overview

TinyTorch is an **educational deep learning framework** that teaches ML systems engineering through hands-on implementation. It prioritizes **clarity over optimization** and **pedagogical flow**.

**Key directories:**
- `src/XX_module/` - Source Python files (jupytext format, source of truth)
- `tinytorch/core/` - Generated package files (auto-generated, do NOT edit directly)
- `tests/XX_module/` - Test files for each module
- `modules/` - Generated notebooks for learners
- `tito/` - CLI tool for development

## Build and Test Commands

### Setup
```bash
pip install -e ".[dev]"           # Install with dev dependencies
make setup                        # Alternative: also installs pytest, rich
```

### Running Tests

**Full test suite:**
```bash
make test                         # Main suite (excludes e2e/milestones)
pytest tests/ -v --ignore=tests/e2e --ignore=tests/milestones -q
```

**Single test file:**
```bash
pytest tests/01_tensor/test_tensor_core.py -v
```

**Single test by name pattern:**
```bash
pytest tests/ -k "test_tensor_addition" -v
pytest tests/ -k "TestTensorCreation" -v
```

**Tests for specific module:**
```bash
make test-module-01_tensor        # Using make
pytest tests/01_tensor/ -v        # Direct pytest
tito module test 01               # Using tito CLI
```

**Quick smoke tests (~30s):**
```bash
make test-quick
make test-e2e-quick
```

**E2E and milestone tests:**
```bash
make test-e2e-full                # Full E2E journey (~10 min)
make test-milestones              # ML learning verification (~90s)
```

### Preflight Checks
```bash
make preflight                    # Quick check (~1 min) - run before work
make preflight-quick              # Faster version
make release                      # Full release validation (~10 min)
```

### Linting
```bash
make lint                         # Syntax check (py_compile)
python -m py_compile path/to/file.py
```

Note: No active formatters (black/isort are commented out in `.pre-commit-config.yaml`).

## Code Style Guidelines

### File Format
- **Indentation:** 4 spaces (no tabs)
- **Line endings:** LF (Unix-style)
- **Encoding:** UTF-8
- **Final newline:** Required
- **Trailing whitespace:** Trimmed

### Imports
```python
# Order: standard library, third-party, local
import numpy as np
from typing import Optional, List

from tinytorch.core.tensor import Tensor
```

### Naming Conventions
| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `Tensor`, `ReLU`, `CrossEntropyLoss` |
| Functions/methods | snake_case | `forward`, `memory_footprint` |
| Constants | UPPER_SNAKE_CASE | `BYTES_PER_FLOAT32`, `KB_TO_BYTES` |
| Private members | Leading underscore | `_cached_value` |
| Module files | `XX_name.py` | `01_tensor.py`, `02_activations.py` |

### Type Hints
- Use on public API methods
- Return types with `-> Type`
- `Optional[Type]` for nullable parameters

```python
def forward(self, x: Tensor, dim: int = -1) -> Tensor:
    ...
```

### Docstrings (Google-style with educational extensions)
```python
def forward(self, x: Tensor) -> Tensor:
    """Apply activation element-wise.

    TODO: Implement function

    APPROACH:
    1. Step one description
    2. Step two description

    EXAMPLE:
    >>> activation = ReLU()
    >>> x = Tensor([1, -2, 3])
    >>> result = activation(x)

    HINT: Use np.maximum for this
    """
```

### Test Docstrings
```python
def test_tensor_from_list(self):
    """
    WHAT: What this test verifies.

    WHY: Why this matters for ML systems.

    STUDENT LEARNING: Key takeaway for students.
    """
```

### Class Structure Pattern
```python
class ActivationName:
    """Class docstring with description."""

    def parameters(self):
        """Return empty list (activations have no learnable parameters)."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        """Main implementation."""
        ### BEGIN SOLUTION
        # Implementation here
        ### END SOLUTION

    def __call__(self, x: Tensor) -> Tensor:
        """Allow callable syntax."""
        return self.forward(x)

    def backward(self, grad: Tensor) -> Tensor:
        """Gradient computation."""
        pass
```

### NBGrader Solution Markers
For educational code with solutions:
```python
### BEGIN SOLUTION
# actual implementation
### END SOLUTION
```

### Error Handling
- Provide helpful, educational error messages
- Include context about what went wrong and why
- Use assertions with descriptive messages:
```python
assert t1.shape == (3,), (
    f"1D tensor has wrong shape.\n"
    f"  Input: [1, 2, 3] (3 elements)\n"
    f"  Expected shape: (3,)\n"
    f"  Got: {t1.shape}"
)
```

## Key Policies

### Git Workflow
- **ALWAYS** use virtual environment (`.venv`)
- **ALWAYS** work on feature branches, never directly on `dev` or `main`
- **NEVER** add "Co-Authored-By" or "Generated with Claude Code" attribution
- Only project owner adds attribution when needed

### Educational Focus
- Prioritize **clarity over optimization**
- Every change should enhance student learning
- Include memory and performance analysis where relevant
- Maintain the pedagogical flow across modules

### Data Types
- Default to `np.float32` for tensors
- Be explicit about dtype when it matters

### Do NOT Edit
- `tinytorch/core/*.py` - These are auto-generated from `src/` files

## Useful Commands

```bash
# System health check
tito system health
tito module status

# Export module to package
tito module complete N
tito src export

# Clean generated files
make clean
```

## Python Version
Supports Python 3.8 - 3.13. Use features compatible with Python 3.8+.

## Dependencies
Core: `numpy>=1.24.0`, `rich>=13.0.0`, `PyYAML>=6.0`
Dev: `pytest>=8.0.0`, `pytest-cov>=4.0.0`, `jupytext>=1.16.0`
