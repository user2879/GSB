"""Business error codes and exceptions used by the geometry component."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    OUTLINE_INVALID = "E_OUTLINE_INVALID"
    BOOLEAN_DEGENERATE = "E_BOOLEAN_DEGENERATE"
    STROKE_WIDTH = "E_STROKE_WIDTH"
    CAP_JOIN = "E_CAP_JOIN"
    CONSISTENCY_DRIFT = "E_CONSISTENCY_DRIFT"
    CLOSURE_MISMATCH = "E_CLOSURE_MISMATCH"
    BITMAP_HASH = "E_BITMAP_HASH"


class GlyphError(Exception):
    """A deterministic, serializable business exception."""

    def __init__(self, code: ErrorCode | str, message: str, details: Any | None = None, **extra_details: Any):
        self.code = ErrorCode(code)
        self.message = message
        if details is None:
            self.details = dict(extra_details)
        elif extra_details:
            if not isinstance(details, dict):
                self.details = {"detail": details, **extra_details}
            else:
                self.details = {**details, **extra_details}
        else:
            self.details = details
        super().__init__(f"{self.code.value}: {message}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }


def fail(code: ErrorCode | str, message: str, **details: Any) -> None:
    raise GlyphError(code, message, details)
