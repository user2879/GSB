"""Business error codes and exception hierarchy for the columnar store.

All expected, domain-level failures raise :class:`ColStoreError` with a stable
business code so that callers (CLI, GUI, tests, structured logs) can branch on
the code rather than on exception messages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class ColStoreError(Exception):
    """Base class for every business error.

    ``code`` is a stable string such as ``E_SEGMENT``; ``context`` carries
    machine-readable details (column name, row group id, block offset, ...).
    """

    code: str = "E_INTERNAL"

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context: Dict[str, Any] = dict(context or {})

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}

    def __str__(self) -> str:  # pragma: no cover - trivial
        ctx = ", ".join(f"{k}={v}" for k, v in self.context.items())
        return f"[{self.code}] {self.message}" + (f" ({ctx})" if ctx else "")


class SegmentError(ColStoreError):
    """Segment file is corrupt or structurally invalid (whole-segment reject)."""

    code = "E_SEGMENT"


class PayloadError(ColStoreError):
    """Injected payload truncation / payload length mismatch."""

    code = "E_PAYLOAD"


class EncodingError(ColStoreError):
    """Encoding marker does not match the actual payload shape."""

    code = "E_ENCODING"


class QueryError(ColStoreError):
    """Unsupported query operator or malformed query."""

    code = "E_QUERY"


class BudgetError(ColStoreError):
    """Memory/parallelism budget makes a scan impossible (e.g. zero budget)."""

    code = "E_BUDGET"


class DivergenceError(ColStoreError):
    """Two scan paths produced different results or statistics."""

    code = "E_DIVERGE"


class SnapshotError(ColStoreError):
    """Snapshot database or version mismatch problem."""

    code = "E_SNAPSHOT"


@dataclass
class ErrorReport:
    """Serializable error record used by structured logging."""

    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_error(cls, exc: ColStoreError) -> "ErrorReport":
        return cls(code=exc.code, message=exc.message, context=dict(exc.context))

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}
