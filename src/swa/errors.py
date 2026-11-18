"""Business error codes and structured error type for the aggregation middleware."""

from __future__ import annotations

from typing import Any


class BusinessError(Exception):
    """Error carrying a stable business code and a JSON-serialisable payload."""

    def __init__(self, code: str, message: str, path: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path
        self.details: dict[str, Any] = details or {}

    def to_payload(self, triggered_monotonic_ms: float | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "error",
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "details": self.details,
        }
        if triggered_monotonic_ms is not None:
            payload["first_triggered_monotonic_ms"] = triggered_monotonic_ms
        return payload


SUPPORTED_AGGREGATIONS = ("count", "sum", "max")
LEDGER_FORMAT_VERSION = 1
