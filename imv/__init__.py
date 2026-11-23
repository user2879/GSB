"""Incremental materialized-view workbench.

The package intentionally depends only on the Python standard library so the
same source tree runs on a stock Windows Python 3.10+ installation.
"""

from .errors import BusinessError

__all__ = ["BusinessError"]
__version__ = "1.0.0"
