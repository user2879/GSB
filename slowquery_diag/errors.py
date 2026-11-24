"""Diagnostic error codes and exception hierarchy.

All user-visible failures carry a stable ``code`` (``E_*``) plus the
reference identity (source/candidate id or record row number) so callers
can locate the first offending object deterministically.
"""

from __future__ import annotations


class DiagError(Exception):
    """Base class for every diagnostic failure."""

    code = "E_INTERNAL"

    def __init__(self, message: str, *, ref: str | None = None,
                 field: str | None = None, detail: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.ref = ref
        self.field = field
        self.detail = detail

    def __str__(self) -> str:
        head = self.code
        if self.ref is not None:
            head += f":{self.ref}"
        if self.field is not None:
            head += f".{self.field}"
        return f"{head} {self.message}"


class SourceDeclarationError(DiagError):
    """Raised on the first invalid observation source / candidate rule."""

    code = "E_SOURCE"


class FormatError(DiagError):
    """Raised on the first record that fails frame or version validation."""

    code = "E_FORMAT"

    def __init__(self, message: str, *, row: int | None = None,
                 source: str | None = None, detail: object | None = None) -> None:
        ref = source if source is not None else None
        field = f"row={row}" if row is not None else None
        super().__init__(message, ref=ref, field=field, detail=detail)
        self.row = row


class PortError(DiagError):
    """Raised when the loopback replay port cannot be bound or connected."""

    code = "E_PORT"


class ReplayTimeoutError(DiagError):
    """Raised when a real socket handshake / reply times out."""

    code = "E_TIMEOUT"


class AckError(DiagError):
    """Raised when a replay receipt is missing required fields."""

    code = "E_ACK"


class ReplayMismatchError(DiagError):
    """Raised when post-recovery replay disagrees with the ledger."""

    code = "E_REPLAY"

    def __init__(self, message: str, *, query: str | None = None,
                 field: str | None = None, detail: object | None = None) -> None:
        super().__init__(message, ref=query, field=field, detail=detail)
        self.query = query


class LedgerError(DiagError):
    """Raised when ledger format version / digest cannot be reconciled."""

    code = "E_LEDGER"
