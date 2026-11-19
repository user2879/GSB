"""Business error codes and exception plumbing for the tactical router."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class ErrorCode(str, Enum):
    MAP_SHAPE = "E_MAP_SHAPE"
    TERRAIN_RANGE = "E_TERRAIN_RANGE"
    DEPENDENCY_STALE = "E_DEPENDENCY_STALE"
    CLOSURE_MISMATCH = "E_CLOSURE_MISMATCH"
    BUDGET_DENIED = "E_BUDGET_DENIED"
    UNIT_INVALID = "E_UNIT_INVALID"
    SNAPSHOT_VERSION = "E_SNAPSHOT_VERSION"


class TacticalError(RuntimeError):
    """A deterministic, machine-readable tactical routing error."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code.value}: {message}")
        self.code = code
        self.message = message
        self.context = dict(context or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": dict(sorted(self.context.items())),
        }


def fail(
    code: ErrorCode,
    message: str,
    *,
    context: Mapping[str, Any] | None = None,
) -> "NoReturn":
    raise TacticalError(code, message, context=context)

