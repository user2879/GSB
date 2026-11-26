from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DiagnosticError(Exception):
    """Structured error carrying one of the public E_* diagnostic codes."""

    code: str
    message: str
    field: str | None = None
    line: int | None = None
    source: str | None = None
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        location = ""
        if self.source:
            location += f"{self.source}"
            if self.line is not None:
                location += f":{self.line}"
            location += " "
        field = f"field={self.field} " if self.field else ""
        return f"[{self.code}] {location}{field}{self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "line": self.line,
            "source": self.source,
            "details": self.details or {},
        }


class Codes:
    SPEC = "E_SPEC"
    FORMAT = "E_FORMAT"
    FIELD = "E_FIELD"
    DUP = "E_DUP"
    LEASE = "E_LEASE"
    PORT = "E_PORT"
    TIMEOUT = "E_TIMEOUT"
    DIVERGE = "E_DIVERGE"
    REPLAY = "E_REPLAY"
    BUDGET = "E_BUDGET"
    STATE = "E_STATE"
