"""Deterministic columnar block-group diagnostics pipeline."""

__version__ = "0.1.0"
FORMAT_VERSION = 1
LEDGER_FORMAT_VERSION = 1

from .errors import E_FALSE_SKIP, E_SPEC, GSBError, SpecError
from .hashing import canonical_dumps, canonical_hash, sha256_hex

__all__ = [
    "E_FALSE_SKIP",
    "E_SPEC",
    "FORMAT_VERSION",
    "LEDGER_FORMAT_VERSION",
    "GSBError",
    "SpecError",
    "canonical_dumps",
    "canonical_hash",
    "sha256_hex",
    "__version__",
]
