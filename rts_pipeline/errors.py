"""Business error type with stable codes and deterministic payloads."""

from __future__ import annotations

from typing import Any

from .constants import ERROR_EXIT_CODES


class PipelineError(Exception):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        if code not in ERROR_EXIT_CODES:
            code = "E_INTERNAL"
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    @property
    def exit_code(self) -> int:
        return ERROR_EXIT_CODES[self.code]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


def require(condition: bool, code: str, message: str, **details: Any) -> None:
    if not condition:
        raise PipelineError(code, message, **details)

