"""Stable diagnostic errors used by the CLI and tests."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any


@dataclass
class DiagnosticError(Exception):
    code: str
    message: str
    file: str = ""
    entry: str = ""
    field: str = ""
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "file": self.file,
            "entry": self.entry,
            "field": self.field,
        }
        if self.details:
            error["details"] = self.details
        return {"error": error}


def spec_error(file: str, entry: str, field: str, message: str, **details: Any) -> DiagnosticError:
    return DiagnosticError("E_SPEC", message, file=file, entry=entry, field=field, details=details)


def field_error(file: str, entry: str, field: str, message: str = "required receipt field is missing or invalid") -> DiagnosticError:
    return DiagnosticError("E_FIELD", message, file=file, entry=entry, field=field)
