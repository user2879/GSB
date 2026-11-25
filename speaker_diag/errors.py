"""Business errors used across the speaker-diag component.

Every public failure has a stable code so callers can assert on a code rather
than parsing an English message.  The line/field pair is deliberately kept on
the exception: the CLI and the Tk viewer both render it verbatim.
"""

from __future__ import annotations

from typing import Any


class DiagnosticError(Exception):
    """A deterministic, machine-readable business error."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        field: str | None = None,
        line: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.field = field
        self.line = line
        self.details = details or {}
        super().__init__(self.__str__())

    def __str__(self) -> str:
        location = ""
        if self.field:
            location += f" field={self.field}"
        if self.line is not None:
            location += f" line={self.line}"
        return f"{self.code}{location}: {self.message}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "field": self.field,
            "line": self.line,
            "message": self.message,
            "details": self.details,
        }


def require(condition: bool, code: str, message: str, **kwargs: Any) -> None:
    if not condition:
        raise DiagnosticError(code, message, **kwargs)

