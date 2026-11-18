"""Structured business errors used by every workbench entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ERROR_CODES = {
    "E_PARSE",
    "E_SESSION",
    "E_DATA",
    "E_VERSION",
    "E_TIMEOUT",
    "E_EQUIV",
    "E_REPLAY",
}


@dataclass
class WorkbenchError(Exception):
    """An expected domain failure with a stable business code."""

    code: str
    message: str
    details: Any = None

    def __post_init__(self) -> None:
        if self.code not in ERROR_CODES:
            raise ValueError(f"unknown workbench error code: {self.code}")
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            payload["details"] = self.details
        return payload


def fail(code: str, message: str, details: Any = None) -> WorkbenchError:
    return WorkbenchError(code=code, message=message, details=details)
