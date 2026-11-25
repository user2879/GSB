"""Structured diagnostics shared by the package and command line interface."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any


@dataclass
class DiagnosticError(Exception):
    code: str
    message: str
    object: str | None = None
    field: str | None = None
    line: int | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.object is not None:
            result["object"] = self.object
        if self.field is not None:
            result["field"] = self.field
        if self.line is not None:
            result["line"] = self.line
        if self.details:
            result["details"] = self.details
        return result

    def __str__(self) -> str:
        location = []
        if self.object is not None:
            location.append(str(self.object))
        if self.line is not None:
            location.append(f"line={self.line}")
        if self.field is not None:
            location.append(f"field={self.field}")
        suffix = f" ({', '.join(location)})" if location else ""
        return f"{self.code}: {self.message}{suffix}"


def is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require(condition: bool, code: str, message: str, **kwargs: Any) -> None:
    if not condition:
        raise DiagnosticError(code=code, message=message, **kwargs)
