"""Business error types shared by the storage service and clients."""

from __future__ import annotations

from typing import Any


class BusinessError(Exception):
    """A deterministic, user-facing failure with a stable business code."""

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


E_CHUNK = "E_CHUNK"
E_STATE = "E_STATE"
E_INTEGRITY = "E_INTEGRITY"
E_VERSION = "E_VERSION"
E_AUDIT = "E_AUDIT"
E_NOT_FOUND = "E_NOT_FOUND"
E_INPUT = "E_INPUT"
E_CONFLICT = "E_CONFLICT"
E_INTERNAL = "E_INTERNAL"


EXIT_CODES = {
    E_CHUNK: 20,
    E_INTEGRITY: 30,
    E_AUDIT: 40,
    E_VERSION: 50,
    E_STATE: 60,
    E_CONFLICT: 61,
    E_NOT_FOUND: 64,
    E_INPUT: 65,
    E_INTERNAL: 70,
}

