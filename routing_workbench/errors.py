"""Business error codes used by all public entry points."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    ROUTE_INVALID = "E_ROUTE_INVALID"
    ROUTE_BUDGET_PENDING = "E_ROUTE_BUDGET_PENDING"
    ROUTE_INJECTED_RUNTIME = "E_ROUTE_INJECTED_RUNTIME"
    SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
    SNAPSHOT_IO = "E_SNAPSHOT_IO"
    COMMAND_INVALID = "E_COMMAND_INVALID"


@dataclass(frozen=True)
class Failure:
    business_code: str
    utterance_id: str
    stage: str
    message: str
    field: str | None = None
    command_index: int | None = None
    injection_id: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "business_code": self.business_code,
            "utterance_id": self.utterance_id,
            "stage": self.stage,
            "message": self.message,
        }
        if self.field is not None:
            result["field"] = self.field
        if self.command_index is not None:
            result["command_index"] = self.command_index
        if self.injection_id is not None:
            result["injection_id"] = self.injection_id
        if self.details:
            result["details"] = self.details
        return result


class BusinessError(Exception):
    """An expected business failure with an explicit code."""

    def __init__(self, code: str | ErrorCode, message: str, **details: Any) -> None:
        self.code = code.value if isinstance(code, ErrorCode) else str(code)
        self.message = message
        self.details = details
        super().__init__(f"{self.code}: {message}")


class SimulatedCrash(RuntimeError):
    """Raised by the test hook that simulates a forced process termination."""
