"""Deterministic fixed-point multimodal renderer.

The package intentionally keeps every render decision content-addressed and
serialisable.  Public entry points live in :mod:`mmfm.cli`, while the modules
below are also usable directly from tests or embedding applications.
"""

from .errors import BusinessError

__all__ = ["BusinessError"]
__version__ = "0.1.0"
