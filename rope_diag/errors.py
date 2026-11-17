"""Typed workbench errors and business codes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_CONFIG_MISMATCH = "E_CONFIG_MISMATCH"
E_NONFINITE_INPUT = "E_NONFINITE_INPUT"
E_POSITION_OUT_OF_RANGE = "E_POSITION_OUT_OF_RANGE"
E_BASE_ZERO = "E_BASE_ZERO"
E_SCALE_INVALID = "E_SCALE_INVALID"
E_NUMERIC_OVERFLOW = "E_NUMERIC_OVERFLOW"
E_INTERNAL = "E_INTERNAL"


@dataclass
class WorkbenchError(Exception):
    """An error that can be rendered into the report failure list."""

    code: str
    message: str
    stage: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(f"{self.code}: {self.message}")

    def as_failure(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "details": self.details,
        }


class ConfigError(WorkbenchError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(E_CONFIG_INVALID, message, details=details or {})


class BoundaryError(WorkbenchError):
    """A numeric boundary condition discovered while encoding a position."""

    def __init__(self, message: str, info: dict[str, Any]) -> None:
        super().__init__(E_CONFIG_INVALID, message, details=info)
        self.info = info
