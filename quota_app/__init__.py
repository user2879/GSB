"""Local storage quota and rate-limit configuration editor.

The package intentionally depends only on the Python standard library.  The
Tkinter front end is imported lazily so the core domain and SQLite ledger can
be used and tested on machines without a Tk installation.
"""

from .errors import AppError

__all__ = ["AppError"]
__version__ = "1.0.0"
