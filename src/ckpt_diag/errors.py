"""Diagnostic error codes for the training checkpoint ledger.

Every validation failure is reported with a stable machine-readable code and
the *first* offending location (a JSON path such as ``shards[0].port``) plus
the offending field name, so callers can reject the whole input deterministically.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional


class DiagError(Exception):
    """Base class carrying a stable code, the first bad location and field."""

    code = "E_INTERNAL"

    def __init__(self, message: str, location: str = "<root>", field: str = "",
                 *, detail: Optional[Any] = None) -> None:
        super().__init__(message)
        self.message = message
        self.location = location
        self.field = field
        self.detail = detail

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "field": self.field,
            "detail": self.detail,
        }

    def __str__(self) -> str:
        loc = f"{self.location}.{self.field}" if self.field else self.location
        return f"{self.code}: {self.message} (at {loc})"


class SpecError(DiagError):
    """E_SPEC: the declarative JSON specification is invalid."""

    code = "E_SPEC"


class FormatError(DiagError):
    """E_FORMAT: unknown receipt format version / unsupported envelope."""

    code = "E_FORMAT"


class FieldError(DiagError):
    """E_FIELD: required field missing, negative duration or width underflow."""

    code = "E_FIELD"


class LeaseError(DiagError):
    """E_LEASE: optimistic lease version mismatch or lease state violation."""

    code = "E_LEASE"


class DivergeError(DiagError):
    """E_DIVERGE: subset recompute disagrees with a blank full rebuild."""

    code = "E_DIVERGE"


class ReplayError(DiagError):
    """E_REPLAY: post-crash replay recompute disagrees with the durable ledger."""

    code = "E_REPLAY"


def first_error(errors: Iterable[DiagError]) -> Optional[DiagError]:
    for err in errors:
        return err
    return None
