"""Business errors with stable machine-readable codes."""

from __future__ import annotations

from typing import Any


class DiagnosticError(Exception):
    """An expected validation, reference, replay, or snapshot failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        item: Any = None,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.item = item
        self.field = field
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "item": self.item,
            "field": self.field,
            "details": self.details,
        }


def format_error(
    message: str,
    *,
    item: Any = None,
    field: str | None = None,
    details: dict[str, Any] | None = None,
) -> DiagnosticError:
    return DiagnosticError("E_FORMAT", message, item=item, field=field, details=details)


def event_error(
    message: str,
    *,
    item: Any = None,
    field: str | None = None,
    details: dict[str, Any] | None = None,
) -> DiagnosticError:
    return DiagnosticError("E_EVENT", message, item=item, field=field, details=details)


def reference_error(
    message: str,
    *,
    item: Any = None,
    field: str | None = None,
    details: dict[str, Any] | None = None,
) -> DiagnosticError:
    return DiagnosticError("E_REF", message, item=item, field=field, details=details)


def snapshot_error(
    message: str,
    *,
    item: Any = None,
    field: str | None = None,
    details: dict[str, Any] | None = None,
) -> DiagnosticError:
    return DiagnosticError("E_SNAPSHOT", message, item=item, field=field, details=details)


def replay_error(
    message: str,
    *,
    item: Any = None,
    field: str | None = None,
    details: dict[str, Any] | None = None,
) -> DiagnosticError:
    return DiagnosticError("E_REPLAY", message, item=item, field=field, details=details)
