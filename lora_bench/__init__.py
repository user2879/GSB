"""Local deterministic multi-task LoRA incremental training workbench.

The package is deliberately self-contained: NumPy plus the standard library,
CPU only, no network access and no real model weights. Every artifact is keyed
by content fingerprints so that incremental runs can be reconciled against
from-scratch rebuilds field by field.
"""

from .errors import WorkbenchError

__all__ = ["WorkbenchError"]
__version__ = "1.0.0"
