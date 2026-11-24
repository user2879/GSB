"""Structured exceptions with stable business codes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProbeLeaseError(Exception):
    code: str
    message: str
    field_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field_path is not None:
            result["field_path"] = self.field_path
        if self.details:
            result["details"] = self.details
        return result


class SessionValidationError(ProbeLeaseError):
    def __init__(
        self,
        message: str,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("E_SESSION", message, field_path, details or {})


class GrantError(ProbeLeaseError):
    def __init__(
        self,
        message: str,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("E_GRANT", message, field_path, details or {})


class DependencyError(SessionValidationError):
    """Dependency validation errors use E_DEPEND while remaining session errors."""

    def __init__(
        self,
        message: str,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        ProbeLeaseError.__init__(self, "E_DEPEND", message, field_path, details or {})
        self.outer_code = "E_SESSION"


class LeaseStateError(ProbeLeaseError):
    def __init__(
        self,
        message: str,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("E_LEASE", message, field_path, details or {})


class LedgerError(ProbeLeaseError):
    def __init__(
        self,
        message: str,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
        code: str = "E_LEDGER",
    ) -> None:
        super().__init__(code, message, field_path, details or {})


class ReplayError(ProbeLeaseError):
    def __init__(
        self,
        message: str,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("E_REPLAY", message, field_path, details or {})


class SpliceError(ProbeLeaseError):
    def __init__(
        self,
        message: str,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("E_SPLICE", message, field_path, details or {})

