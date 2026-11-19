"""Deterministic multi-band remote-sensing tile segmentation (incremental).

The package implements a fully deterministic, integer-preserving inference
pipeline: multi-band tile loading and index validation, sliding-window feature
extraction, pixel classification, cross-tile overlap fusion, connected-area
post-processing, dependency-fingerprint caching, budgeted recomputation and
atomic versioned snapshots.
"""

from .errors import BusinessError, ErrorCode

__all__ = ["BusinessError", "ErrorCode"]
__version__ = "1.0.0"
