"""Business error types used at every public boundary."""

from __future__ import annotations

from typing import Any


ERROR_CODES = {
    "E_SCHEMA",
    "E_JSON",
    "E_RATE",
    "E_WAV",
    "E_TRUNC",
    "E_SILENT",
    "E_BAND",
    "E_IO",
    "E_WR",
    "E_INTERRUPTED",
    "E_CANCELLED",
    "E_INTERNAL",
    "E_PARTIAL",
    "E_OK",
}


class BusinessError(Exception):
    """An error that maps to a stable, machine-readable business code."""

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        **extra_details: Any,
    ) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown business code: {code}")
        self.code = code
        self.message = message
        self.details = dict(details or {})
        self.details.update(extra_details)
        Exception.__init__(self, f"{code}: {message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "business_code": self.code,
            "status": "error",
            "message": self.message,
            "details": self.details,
        }


def schema_error(message: str, **details: Any) -> BusinessError:
    return BusinessError("E_SCHEMA", message, details)
