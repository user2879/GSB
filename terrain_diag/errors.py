from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    E_HEIGHT_RANGE = "E_HEIGHT_RANGE"
    E_BLOCK_MISSING = "E_BLOCK_MISSING"
    E_HIZ_LEVEL = "E_HIZ_LEVEL"
    E_DEPTH_BIAS = "E_DEPTH_BIAS"
    E_BUDGET_DENIED = "E_BUDGET_DENIED"
    E_RESIDENT_LEAK = "E_RESIDENT_LEAK"
    E_TRACK_DIVERGE = "E_TRACK_DIVERGE"


@dataclass
class WorkbenchError(RuntimeError):
    code: ErrorCode
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.code, ErrorCode):
            self.code = ErrorCode(str(self.code))
        RuntimeError.__init__(self, f"{self.code.value}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": self.context,
        }


def fail(code: ErrorCode | str, message: str, **context: Any) -> None:
    normalized = code if isinstance(code, ErrorCode) else ErrorCode(str(code))
    raise WorkbenchError(normalized, message, context)
