from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_CONFIG = "E_CONFIG"
E_DATA = "E_DATA"
E_OOM = "E_OOM"
E_PRESSURE = "E_PRESSURE"
E_LATENCY = "E_LATENCY"
E_IO = "E_IO"


@dataclass
class DiagnosticError(Exception):
    code: str
    message: str
    field_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        location = f" at {self.field_path}" if self.field_path else ""
        return f"{self.code}{location}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "field_path": self.field_path,
            "details": self.details,
        }
