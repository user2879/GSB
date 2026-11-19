"""Stable business error codes shared by the CLI, scheduler, and validators."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    TILE_SIZE = "E_TILE_SIZE"
    THREAD_BOUNDS = "E_THREAD_BOUNDS"
    IOR_RANGE = "E_IOR_RANGE"
    MEDIUM_COEFF = "E_MEDIUM_COEFF"
    BUDGET_DENIED = "E_BUDGET_DENIED"


@dataclass
class BusinessError(Exception):
    code: ErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": str(self.code),
            "message": self.message,
            "details": self.details,
        }


def fail(code: ErrorCode, message: str, **details: Any) -> BusinessError:
    return BusinessError(code=code, message=message, details=details)
