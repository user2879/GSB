"""Business errors with stable codes and JSON-pointer style locations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_MANIFEST = "E_MANIFEST"
E_REF = "E_REF"
E_SNAPSHOT = "E_SNAPSHOT"
E_REPLAY = "E_REPLAY"
E_INCR = "E_INCR"
E_GRAPH = "E_GRAPH"
E_IO = "E_IO"


@dataclass
class BusinessError(Exception):
    code: str
    message: str
    location: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.location:
            payload["location"] = self.location
        if self.details:
            payload["details"] = self.details
        return payload
