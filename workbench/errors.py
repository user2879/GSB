from __future__ import annotations

from typing import Any


class WorkbenchError(Exception):
    """Base error carrying the platform business code."""

    code = "E_WORKBENCH"

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        **details: Any,
    ) -> None:
        super().__init__(message or self.code)
        self.message = message or self.code
        if code:
            self.code = code
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        payload = {"code": self.code, "message": self.message}
        payload.update(self.details)
        return payload


class SpecError(WorkbenchError):
    code = "E_SPEC"

    def __init__(self, message: str, *, field: str, location: str) -> None:
        super().__init__(message, field=field, location=location)
        self.field = field
        self.location = location


class PortError(WorkbenchError):
    code = "E_PORT"


class TimeoutError(WorkbenchError):
    code = "E_TIMEOUT"


class FieldError(WorkbenchError):
    code = "E_FIELD"

    def __init__(self, message: str = "", *, field: str = "$", **details: Any) -> None:
        super().__init__(message, field=field, **details)
        self.field = field


class LeaseError(WorkbenchError):
    code = "E_LEASE"


class ReplayError(WorkbenchError):
    code = "E_REPLAY"


class NodeGoneError(WorkbenchError):
    code = "E_NODE_GONE"


class SchedulingError(WorkbenchError):
    code = "E_SCHED"
