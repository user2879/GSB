from __future__ import annotations

from typing import Any, Optional


OK = "OK"
E_SPEC = "E_SPEC"
E_PORT = "E_PORT"
E_TIMEOUT = "E_TIMEOUT"
E_FIELD = "E_FIELD"
E_LEASE = "E_LEASE"
E_STATE = "E_STATE"
E_LIMIT = "E_LIMIT"
E_VERIFY = "E_VERIFY"
E_IO = "E_IO"


class MeshError(Exception):
    """Operational error carrying one of the platform's stable codes."""

    code = "E_IO"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        field: Optional[str] = None,
        line: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        if code:
            self.code = code
        self.message = message
        self.field = field
        self.line = line
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "error": self.message,
        }
        if self.field is not None:
            payload["field"] = self.field
        if self.line is not None:
            payload["line"] = self.line
        if self.details:
            payload["details"] = self.details
        return payload


class SpecError(MeshError):
    code = E_SPEC


class LeaseError(MeshError):
    code = E_LEASE


class VerifyError(MeshError):
    code = E_VERIFY


def error_result(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, MeshError):
        return exc.to_dict()
    return {"ok": False, "code": E_IO, "error": str(exc)}
