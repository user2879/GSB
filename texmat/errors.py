"""Stable error codes and typed failures for the texmat pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class TexmatError(Exception):
    """Base class carrying a stable machine-readable code."""

    code = "E_INTERNAL"

    def __init__(self, message: str = "", *, details: dict[str, Any] | None = None):
        super().__init__(message or self.code)
        self.message = message or self.code
        self.details = details or {}


class ConfigInvalid(TexmatError):
    code = "E_CONFIG_INVALID"


class TextureDecodeError(TexmatError):
    code = "E_TEXTURE_DECODE"


class ConfigMismatch(TexmatError):
    code = "E_CONFIG_MISMATCH"


class PointBudgetPending(Exception):
    """Control-flow marker: a task is retained for a later run (not a hard error)."""

    code = "E_POINT_BUDGET_PENDING"

    def __init__(self, task_key: str, points: int):
        super().__init__(f"task {task_key} pending ({points} points over budget)")
        self.task_key = task_key
        self.points = points
        self.code = PointBudgetPending.code


@dataclass(frozen=True)
class Failure:
    """A single failure-list entry with a deterministic ordering key."""

    code: str
    target: str
    message: str
    order_key: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "target": self.target,
            "message": self.message,
            "order_key": self.order_key,
            "details": self.details,
        }
