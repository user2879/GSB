"""Typed diagnostic errors used by the CLI, runtime, and subprocesses."""

from __future__ import annotations

from typing import Any


class DiagnosticError(Exception):
    """An error with one of the stable ``E_*`` diagnostic codes."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        field: str | None = None,
        location: str | None = None,
        **details: Any,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.location = location
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        if self.field is not None:
            payload["field"] = self.field
        if self.location is not None:
            payload["location"] = self.location
        if self.details:
            payload["details"] = self.details
        return payload


class SpecError(DiagnosticError):
    def __init__(self, message: str, *, field: str, location: str, **details: Any) -> None:
        super().__init__("E_SPEC", message, field=field, location=location, **details)


class PortError(DiagnosticError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_PORT", message, **details)


class TimeoutDiagnosticError(DiagnosticError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_TIMEOUT", message, **details)


class ProtocolError(DiagnosticError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_PROTO", message, **details)


class LeaseError(DiagnosticError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_LEASE", message, **details)


class LedgerError(DiagnosticError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_LEDGER", message, **details)
