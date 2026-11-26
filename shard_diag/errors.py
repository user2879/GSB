from __future__ import annotations

from typing import Any, Optional


class DiagnosticError(Exception):
    code = "E_DIAG"

    def __init__(
        self,
        message: str,
        *,
        entry: Optional[str] = None,
        field: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.entry = entry
        self.field = field
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "error": self.message,
        }
        if self.entry is not None:
            payload["entry"] = self.entry
        if self.field is not None:
            payload["field"] = self.field
        if self.details:
            payload["details"] = self.details
        return payload


class SpecError(DiagnosticError):
    code = "E_SPEC"


class PortInUseError(DiagnosticError):
    code = "E_PORT"


class SocketTimeoutError(DiagnosticError):
    code = "E_TIMEOUT"


class LeaseConflictError(DiagnosticError):
    code = "E_LEASE"


class ReplayConsistencyError(DiagnosticError):
    code = "E_REPLAY"


class ProtocolError(DiagnosticError):
    code = "E_PROTO"


class NodeUnavailableError(DiagnosticError):
    code = "E_NODE"
