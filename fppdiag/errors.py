"""Structured business error codes.

Every diagnostic failure raises :class:`DiagError` carrying a stable business
code so callers (CLI, tests) can branch on the code rather than message text.
"""

from __future__ import annotations


class DiagError(Exception):
    """Base error with a stable business code and a located offending field."""

    code = "E_UNKNOWN"

    def __init__(self, message: str, *, location: str = "", field: str = "",
                 detail: dict | None = None):
        super().__init__(message)
        self.message = message
        # location identifies the first offending stage id or data source id.
        self.location = location
        # field is the offending attribute name within that object.
        self.field = field
        self.detail = detail or {}

    def to_dict(self) -> dict:
        out = {"code": self.code, "message": self.message,
               "location": self.location, "field": self.field}
        if self.detail:
            out["detail"] = self.detail
        return out


class SpecError(DiagError):
    """Pipeline/source/window specification is invalid (business req. 1)."""

    code = "E_SPEC"


class DataError(DiagError):
    """Generated/cached data source content is unreadable or corrupt."""

    code = "E_DATA"


class ParamError(DiagError):
    """Runtime stage parameter violates the integer fixed-point envelope."""

    code = "E_PARAM"


class DivergeError(DiagError):
    """Incremental recompute differs from a from-scratch full recompute."""

    code = "E_DIVERGE"


class ReplayError(DiagError):
    """Recovered replay disagrees with a committed ledger record."""

    code = "E_REPLAY"


class SnapshotError(DiagError):
    """Snapshot/ledger content digest or format identity mismatch."""

    code = "E_SNAPSHOT"
