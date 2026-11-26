from __future__ import annotations

from typing import Any, Optional


class DiagnosticError(Exception):
    """Structured failure carrying one of the task's stable error codes."""

    code = "E_INTERNAL"

    def __init__(
        self,
        message: str,
        *,
        field: Optional[str] = None,
        object_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        self.field = field
        self.object_id = object_id
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
            },
        }
        if self.field is not None:
            payload["error"]["field"] = self.field
        if self.object_id is not None:
            payload["error"]["object_id"] = self.object_id
        if self.details:
            payload["error"]["details"] = self.details
        return payload


class SpecError(DiagnosticError):
    code = "E_SPEC"


class PortError(DiagnosticError):
    code = "E_PORT"


class ProtocolTimeout(DiagnosticError):
    code = "E_TIMEOUT"


class FieldError(DiagnosticError):
    code = "E_FIELD"


class LeaseError(DiagnosticError):
    code = "E_LEASE"


class VerifyError(DiagnosticError):
    code = "E_VERIFY"
