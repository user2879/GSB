"""Business error codes with deterministic, serializable context."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    E_FRAME_MISSING = "E_FRAME_MISSING"
    E_THRESHOLD_RANGE = "E_THRESHOLD_RANGE"
    E_KEYFRAME_INVALID = "E_KEYFRAME_INVALID"
    E_QUAT_INVALID = "E_QUAT_INVALID"
    E_CLOSURE_MISMATCH = "E_CLOSURE_MISMATCH"
    E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
    E_BUDGET_EXCEEDED = "E_BUDGET_EXCEEDED"


@dataclass
class BusinessError(Exception):
    code: ErrorCode
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code.value}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code.value,
            "message": self.message,
            "context": self.context,
        }


def require(condition: bool, code: ErrorCode, message: str, **context: Any) -> None:
    if not condition:
        raise BusinessError(code, message, context)

