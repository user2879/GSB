"""Typed, stable error codes used across the package and CLI."""

from __future__ import annotations

from typing import Any


class GSBError(Exception):
    """Base error with a machine-readable code and a deterministic payload."""

    code = "E_GSB"

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.path = path
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.path is not None:
            out["path"] = self.path
        if self.details:
            out["details"] = self.details
        return out


class SpecError(GSBError):
    code = "E_SPEC"


class DependencyError(GSBError):
    code = "E_DEPEND"


class PortError(GSBError):
    code = "E_PORT"


class GSBTimeoutError(GSBError):
    code = "E_TIMEOUT"


class ProtocolError(GSBError):
    code = "E_PROTO"


class DivergenceError(GSBError):
    code = "E_DIVERGE"

    def __init__(
        self,
        first_shard: str,
        first_field: str,
        expected: Any,
        actual: Any,
    ) -> None:
        super().__init__(
            f"first divergence at shard {first_shard!r}, field {first_field!r}",
            details={
                "first_shard": first_shard,
                "first_field": first_field,
                "expected": expected,
                "actual": actual,
            },
        )
        self.first_shard = first_shard
        self.first_field = first_field
        self.expected = expected
        self.actual = actual


class SnapshotError(GSBError):
    code = "E_SNAPSHOT"


class ScheduleError(GSBError):
    code = "E_SCHEDULE"


class LeaseStateError(GSBError):
    code = "E_LEASE"

