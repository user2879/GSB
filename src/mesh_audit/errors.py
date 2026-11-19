from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


class ErrorCode:
    E_SCHEMA = "E_SCHEMA"
    E_INDEX = "E_INDEX"
    E_IO = "E_IO"
    E_BUDGET = "E_BUDGET"
    E_STATE = "E_STATE"
    E_SNAPSHOT_INVALID = "E_SNAPSHOT_INVALID"
    E_INTERNAL = "E_INTERNAL"


@dataclass(slots=True)
class BusinessError(Exception):
    """A user-facing error carrying a stable business code."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
