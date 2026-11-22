"""Deterministic local query-plan and index-selection diagnostics.

The package deliberately keeps the estimate model, executor and ledger in
separate modules.  The independent verifier uses only the declaration plus the
committed ledger (including archived statistics), never the generated data
file.
"""

from .errors import QPlanError

__all__ = ["QPlanError"]
__version__ = "0.1.0"

