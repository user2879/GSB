"""Stable business errors and deterministic exit codes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    E_OBS_MISSING = "E_OBS_MISSING"
    E_RETARGET_TOL = "E_RETARGET_TOL"
    E_CONSTRAINT_UNSAT = "E_CONSTRAINT_UNSAT"
    E_STRATEGY_BUDGET = "E_STRATEGY_BUDGET"
    E_CLOSURE_MISMATCH = "E_CLOSURE_MISMATCH"
    E_INPUT = "E_INPUT"
    E_IO = "E_IO"


EXIT_CODES: dict[ErrorCode, int] = {
    ErrorCode.E_OBS_MISSING: 10,
    ErrorCode.E_RETARGET_TOL: 11,
    ErrorCode.E_CONSTRAINT_UNSAT: 12,
    ErrorCode.E_STRATEGY_BUDGET: 13,
    ErrorCode.E_CLOSURE_MISMATCH: 14,
    ErrorCode.E_INPUT: 20,
    ErrorCode.E_IO: 21,
}


class BusinessError(RuntimeError):
    def __init__(self, code: ErrorCode, message: str, **details: Any):
        super().__init__(f"{code.value}: {message}")
        self.code = code
        self.message = message
        self.details = details

    def __str__(self) -> str:
        return f"{self.code.value}: {self.message}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error_code": self.code.value,
            "message": self.message,
            "details": self.details,
            "exit_code": EXIT_CODES[self.code],
        }


def ensure(condition: bool, code: ErrorCode, message: str, **details: Any) -> None:
    if not condition:
        raise BusinessError(code, message, details)
