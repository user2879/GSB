from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any


ERROR_CODES = {
    "E_SPEC",
    "E_FORMAT",
    "E_LEDGER",
    "E_SNAPSHOT",
    "E_PREVIEW",
    "E_REPLAY",
    "E_IO",
    "E_RUNTIME",
}


@dataclass
class LabError(Exception):
    code: str
    message: str
    field: str | None = None
    batch_id: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in ERROR_CODES:
            raise ValueError(f"unknown error code: {self.code}")
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "field": self.field,
            "batch_id": self.batch_id,
            "details": self.details,
        }

    def __str__(self) -> str:
        parts = [self.code, self.message]
        if self.field:
            parts.append(f"field={self.field}")
        if self.batch_id:
            parts.append(f"batch={self.batch_id}")
        return " | ".join(parts)
