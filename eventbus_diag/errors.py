"""Typed diagnostics with stable machine-readable error codes."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any


@dataclass
class BusDiagnosticError(Exception):
    code: str
    message: str
    field: str | None = None
    line: int | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.field is not None:
            payload["field"] = self.field
        if self.line is not None:
            payload["line"] = self.line
        if self.details:
            payload["details"] = self.details
        return payload

    def __str__(self) -> str:
        location = ""
        if self.field:
            location += f" field={self.field}"
        if self.line is not None:
            location += f" line={self.line}"
        return f"{self.code}:{location} {self.message}".strip()


class SpecError(BusDiagnosticError):
    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        line: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("E_SPEC", message, field, line, details or {})


class PortError(BusDiagnosticError):
    def __init__(
        self,
        message: str,
        *,
        subscriber_id: str | None = None,
        port: int | None = None,
        line: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        extra = dict(details or {})
        if subscriber_id is not None:
            extra["subscriber_id"] = subscriber_id
        if port is not None:
            extra["port"] = port
        super().__init__("E_PORT", message, None, line, extra)


class ProtocolTimeoutError(BusDiagnosticError):
    def __init__(
        self,
        message: str,
        *,
        subscriber_id: str | None = None,
        batch_id: str | None = None,
        port: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        extra = dict(details or {})
        if subscriber_id is not None:
            extra["subscriber_id"] = subscriber_id
        if batch_id is not None:
            extra["batch_id"] = batch_id
        if port is not None:
            extra["port"] = port
        super().__init__("E_TIMEOUT", message, None, None, extra)


class AckError(BusDiagnosticError):
    def __init__(
        self,
        message: str,
        *,
        subscriber_id: str | None = None,
        batch_id: str | None = None,
        missing_fields: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        extra = dict(details or {})
        if subscriber_id is not None:
            extra["subscriber_id"] = subscriber_id
        if batch_id is not None:
            extra["batch_id"] = batch_id
        if missing_fields:
            extra["missing_fields"] = missing_fields
        super().__init__("E_ACK", message, None, None, extra)


class LeaseError(BusDiagnosticError):
    def __init__(
        self,
        message: str,
        *,
        lease_id: str | None = None,
        batch_id: str | None = None,
        subscriber_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        extra = dict(details or {})
        if lease_id is not None:
            extra["lease_id"] = lease_id
        if batch_id is not None:
            extra["batch_id"] = batch_id
        if subscriber_id is not None:
            extra["subscriber_id"] = subscriber_id
        super().__init__("E_LEASE", message, None, None, extra)


class ReplayError(BusDiagnosticError):
    def __init__(
        self,
        message: str,
        *,
        batch_id: str | None = None,
        mismatch_field: str | None = None,
        ordinal: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        extra = dict(details or {})
        if batch_id is not None:
            extra["batch_id"] = batch_id
        if mismatch_field is not None:
            extra["mismatch_field"] = mismatch_field
        if ordinal is not None:
            extra["ordinal"] = ordinal
        super().__init__("E_REPLAY", message, None, None, extra)
