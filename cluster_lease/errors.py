"""Structured business errors."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any


@dataclass
class LeaseError(Exception):
    code: str
    message: str
    field: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        if self.field is not None:
            result["field"] = self.field
        if self.details:
            result["details"] = self.details
        return result


def spec_error(field: str, message: str, *, code: str = "E_SPEC") -> LeaseError:
    return LeaseError(code=code, field=field, message=message)
