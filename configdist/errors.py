from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any


@dataclass
class AppError(Exception):
    """A stable, machine-readable application error.

    ``field`` and ``location`` always identify the first rejected domain field.
    ``details`` carries diagnostic values such as the first mismatching shard.
    """

    code: str
    message: str
    field: str | None = None
    location: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field is not None:
            payload["field"] = self.field
        if self.location is not None:
            payload["location"] = self.location
        if self.details:
            payload["details"] = self.details
        return payload
