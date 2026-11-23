"""kvdiag: key-value index shard and lease-expiry diagnostics."""

from .errors import (ChainError, DivergeError, KvDiagError, LeaseError,
                     SnapshotError, SpecError)

__all__ = ["KvDiagError", "SpecError", "LeaseError", "ChainError",
           "DivergeError", "SnapshotError"]
__version__ = "1.0.0"
