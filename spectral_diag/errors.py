"""Typed error vocabulary used by every layer of the application."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    MATRIX_SINGULAR = "E_MATRIX_SINGULAR"
    QUANT_RANGE = "E_QUANT_RANGE"
    WHITE_POINT = "E_WHITE_POINT"
    OBSERVER_INVALID = "E_OBSERVER_INVALID"
    INTEGRATION_DENSITY = "E_INTEGRATION_DENSITY"
    ATTRIB_MISMATCH = "E_ATTRIB_MISMATCH"
    SNAPSHOT_VERSION = "E_SNAPSHOT_VERSION"
    CONFIG_INVALID = "E_CONFIG_INVALID"
    INPUT_INVALID = "E_INPUT_INVALID"


EXIT_CODES: dict[ErrorCode, int] = {
    ErrorCode.CONFIG_INVALID: 2,
    ErrorCode.INPUT_INVALID: 3,
    ErrorCode.MATRIX_SINGULAR: 10,
    ErrorCode.QUANT_RANGE: 11,
    ErrorCode.WHITE_POINT: 12,
    ErrorCode.OBSERVER_INVALID: 13,
    ErrorCode.INTEGRATION_DENSITY: 16,
    ErrorCode.ATTRIB_MISMATCH: 14,
    ErrorCode.SNAPSHOT_VERSION: 15,
}


@dataclass
class DiagnosticError(Exception):
    """An error with a stable machine-readable code and JSON details."""

    code: ErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, f"{self.code}: {self.message}")

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.code, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": str(self.code),
            "message": self.message,
            "details": self.details,
        }


def fail(code: ErrorCode, message: str, **details: Any) -> None:
    raise DiagnosticError(code=code, message=message, details=details)
