"""WAL crash-recovery diagnostics.

The package is intentionally standard-library-only.  Tkinter is imported
lazily by :mod:`walcrash.gui` so that the deterministic engine and CLI work in
headless environments as well.
"""

from .errors import BusinessError
from .validation import load_sample

__all__ = ["BusinessError", "load_sample"]
__version__ = "1.0.0"

