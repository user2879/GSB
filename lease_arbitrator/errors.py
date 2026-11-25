from __future__ import annotations

from typing import Any


class ArbitrationError(Exception):
    """Structured failure carrying one of the task's stable error codes."""

    code = "E_ARBITRATION"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        entry_id: str | None = None,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        if code:
            self.code = code
        self.message = message
        self.entry_id = entry_id
        self.field = field
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.entry_id is not None:
            result["entry_id"] = self.entry_id
        if self.field is not None:
            result["field"] = self.field
        if self.details:
            result["details"] = self.details
        return result


class SpecError(ArbitrationError):
    code = "E_SPEC"


class PortError(ArbitrationError):
    code = "E_PORT"


class HandshakeError(ArbitrationError):
    code = "E_HANDSHAKE"


class FieldError(ArbitrationError):
    code = "E_FIELD"


class LeaseConflictError(ArbitrationError):
    code = "E_LEASE"


class VerificationError(ArbitrationError):
    code = "E_VERIFY"


class UnknownBatchError(ArbitrationError):
    code = "E_UNKNOWN"
