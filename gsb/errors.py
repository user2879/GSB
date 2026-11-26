from __future__ import annotations

from typing import Any


E_INPUT_INVALID = "E_INPUT_INVALID"
E_DOMAIN_QUOTA_EXCEEDED = "E_DOMAIN_QUOTA_EXCEEDED"
E_BUDGET_PENDING = "E_BUDGET_PENDING"
E_SHARD_COMPARISON_LIMIT = "E_SHARD_COMPARISON_LIMIT"
E_VERSION_CONFLICT = "E_VERSION_CONFLICT"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"
E_INTERNAL = "E_INTERNAL"

BUSINESS_CODES = frozenset(
    {
        E_INPUT_INVALID,
        E_DOMAIN_QUOTA_EXCEEDED,
        E_BUDGET_PENDING,
        E_SHARD_COMPARISON_LIMIT,
        E_VERSION_CONFLICT,
        E_SNAPSHOT_CORRUPT,
        E_INTERNAL,
    }
)


class PipelineError(Exception):
    """Structured error carrying a stable business code and location."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        field: str | None = None,
        batch_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.batch_id = batch_id
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "error_code": self.code,
            "message": self.message,
        }
        if self.field is not None:
            payload["field"] = self.field
        if self.batch_id is not None:
            payload["batch_id"] = self.batch_id
        if self.details:
            payload["details"] = self.details
        return payload


class InputValidationError(PipelineError):
    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        batch_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            E_INPUT_INVALID,
            message,
            field=field,
            batch_id=batch_id,
            details=details,
        )
