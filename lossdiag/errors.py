"""Business errors with stable codes and structured details."""

from __future__ import annotations

from typing import Any


class DiagnosticError(Exception):
    """A failure that belongs to the workbench's public error contract."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


def wire_error(message: str, *, byte_offset: int, frame_index: int, **details: Any) -> DiagnosticError:
    return DiagnosticError(
        "E_WIRE",
        message,
        byte_offset=byte_offset,
        frame_index=frame_index,
        **details,
    )


def format_error(message: str, *, record_number: int, field: str, **details: Any) -> DiagnosticError:
    return DiagnosticError(
        "E_FORMAT",
        message,
        record_number=record_number,
        field=field,
        **details,
    )


def replay_error(
    message: str,
    *,
    flow_id: str | None,
    gap_interval: dict[str, int] | None,
    field: str,
    expected: Any,
    actual: Any,
) -> DiagnosticError:
    return DiagnosticError(
        "E_REPLAY",
        message,
        flow_id=flow_id,
        gap_interval=gap_interval,
        field=field,
        expected=expected,
        actual=actual,
    )


def echo_error(
    message: str,
    *,
    flow_id: str | None,
    gap_interval: dict[str, int] | None,
    field: str,
    expected: Any,
    actual: Any,
) -> DiagnosticError:
    return DiagnosticError(
        "E_ECHO",
        message,
        flow_id=flow_id,
        gap_interval=gap_interval,
        field=field,
        expected=expected,
        actual=actual,
    )


def snapshot_error(message: str, **details: Any) -> DiagnosticError:
    return DiagnosticError("E_SNAPSHOT", message, **details)

