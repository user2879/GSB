"""RoPE extrapolation diagnostics.

The package deliberately has a small public surface.  All numerical paths use
``numpy.float64`` and all identifiers are derived from canonical JSON so that
serial and concurrent runs produce byte-for-byte stable diagnostic payloads.
"""

from .errors import ErrorCode, Failure
from .pipeline import DiagnosticRunner

__all__ = ["DiagnosticRunner", "ErrorCode", "Failure"]
__version__ = "1.0.0"

