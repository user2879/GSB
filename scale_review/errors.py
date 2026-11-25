from __future__ import annotations

from typing import Any


class ErrorCode:
    SPEC = "E_SPEC"
    DIVERGE = "E_DIVERGE"
    LEASE = "E_LEASE"
    REPLAY = "E_REPLAY"


class ScaleReviewError(Exception):
    """Structured error with a stable machine-readable code."""

    code = "E_REVIEW"

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.field = field
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
            },
        }
        if self.field is not None:
            payload["error"]["field"] = self.field
        if self.details:
            payload["error"]["details"] = self.details
        return payload


class SpecError(ScaleReviewError):
    code = ErrorCode.SPEC


class DivergenceError(ScaleReviewError):
    code = ErrorCode.DIVERGE


class LeaseError(ScaleReviewError):
    code = ErrorCode.LEASE


class ReplayError(ScaleReviewError):
    code = ErrorCode.REPLAY
