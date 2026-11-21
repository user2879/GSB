from __future__ import annotations

from typing import Any


class DiagnosticError(Exception):
    def __init__(
        self,
        code: str,
        location: dict[str, Any],
        message: str,
        details: dict[str, Any] | None = None,
        **extra: Any,
    ):
        super().__init__(message)
        self.code = code
        self.location = location
        self.message = message
        self.details = dict(details or {})
        self.details.update(extra)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "details": self.details,
        }


class TaskTimedOut(Exception):
    pass


class TaskExecutionError(Exception):
    pass
