from __future__ import annotations

from typing import Any


class DiagnosticError(Exception):
    """Base class carrying a stable platform error code."""

    code = "E_DIAG"

    def __init__(
        self,
        message: str,
        *,
        object: str | None = None,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.object = object
        self.field = field
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "object": self.object,
            "field": self.field,
            "details": self.details,
        }


class SpecError(DiagnosticError):
    code = "E_SPEC"


class LeaseError(DiagnosticError):
    code = "E_LEASE"


class DivergeError(DiagnosticError):
    code = "E_DIVERGE"


class ReplayError(DiagnosticError):
    code = "E_REPLAY"


class NodeProtocolError(DiagnosticError):
    code = "E_NODE"


class CapacityStaleError(DiagnosticError):
    code = "E_CAPACITY"
