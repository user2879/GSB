"""Stable business errors used by the package and command line."""

from __future__ import annotations

from typing import Any


class LeaseMiddlewareError(Exception):
    code = "E_INTERNAL"

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
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        if self.field is not None:
            payload["field"] = self.field
        if self.details:
            payload["details"] = self.details
        return payload


class SpecError(LeaseMiddlewareError):
    code = "E_SPEC"


class PortError(LeaseMiddlewareError):
    code = "E_PORT"


class TimeoutError(LeaseMiddlewareError):
    code = "E_TIMEOUT"


class ReplyError(LeaseMiddlewareError):
    code = "E_REPLY"


class LeaseError(LeaseMiddlewareError):
    code = "E_LEASE"


class VerifyError(LeaseMiddlewareError):
    code = "E_VERIFY"


class SnapshotError(LeaseMiddlewareError):
    code = "E_SNAPSHOT"
