"""Structured errors used by every diagnostic surface."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any


ERROR_CODES = {
    "E_SCHEMA",
    "E_DEPEND",
    "E_LEASE",
    "E_BUDGET",
    "E_SNAPSHOT",
    "E_REPLAY",
    "E_IO",
}


@dataclass
class DiagnosticError(Exception):
    """A deterministic, serialisable application error.

    ``layer_id`` and ``field`` identify the first offending declaration
    location.  Details never contain object addresses or timestamps, so error
    reports are stable across processes.
    """

    code: str
    message: str
    layer_id: str | None = None
    field: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in ERROR_CODES:
            raise ValueError(f"unknown error code: {self.code}")
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        if self.layer_id is not None:
            result["layer_id"] = self.layer_id
        if self.field is not None:
            result["field"] = self.field
        if self.details:
            result["details"] = self.details
        return result


def schema_error(
    message: str,
    *,
    layer_id: str | None = None,
    field: str | None = None,
    **details: Any,
) -> DiagnosticError:
    return DiagnosticError(
        "E_SCHEMA", message, layer_id=layer_id, field=field, details=details
    )


def dependency_error(
    message: str,
    *,
    layer_id: str | None = None,
    field: str | None = None,
    **details: Any,
) -> DiagnosticError:
    return DiagnosticError(
        "E_DEPEND", message, layer_id=layer_id, field=field, details=details
    )


def lease_error(
    message: str,
    *,
    layer_id: str | None = None,
    field: str | None = None,
    **details: Any,
) -> DiagnosticError:
    return DiagnosticError(
        "E_LEASE", message, layer_id=layer_id, field=field, details=details
    )


def budget_error(message: str, **details: Any) -> DiagnosticError:
    return DiagnosticError("E_BUDGET", message, details=details)


def snapshot_error(message: str, **details: Any) -> DiagnosticError:
    return DiagnosticError("E_SNAPSHOT", message, details=details)


def replay_error(
    message: str,
    *,
    layer_id: str | None = None,
    field: str | None = None,
    **details: Any,
) -> DiagnosticError:
    return DiagnosticError(
        "E_REPLAY", message, layer_id=layer_id, field=field, details=details
    )
