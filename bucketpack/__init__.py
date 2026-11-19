"""Deterministic bucket expansion and budgeted packing engine.

The package intentionally keeps its public surface small.  The CLI and the
library entry points share the same implementation so that reproducible demos
and tests exercise exactly the same code path.
"""

from .errors import BusinessError
from .spec import load_spec, validate_spec

__all__ = ["BusinessError", "load_spec", "validate_spec"]
__version__ = "1.0.0"
