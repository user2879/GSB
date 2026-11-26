from __future__ import annotations

from typing import Any


class QuotaLabError(Exception):
    """Structured error with a stable public code and field path."""

    code = "E_INTERNAL"

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        details: dict[str, Any] | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.field = field
        self.details = details or {}
        if code:
            self.code = code

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
            },
        }
        if self.field is not None:
            payload["error"]["first_error_field"] = self.field
        if self.details:
            payload["error"]["details"] = self.details
        return payload


class SpecError(QuotaLabError):
    code = "E_SPEC"


class PortError(QuotaLabError):
    code = "E_PORT"


class TimeoutError(QuotaLabError):
    code = "E_TIMEOUT"


class LeaseError(QuotaLabError):
    code = "E_LEASE"


class BudgetError(QuotaLabError):
    code = "E_BUDGET"


class LedgerError(QuotaLabError):
    code = "E_LEDGER"
