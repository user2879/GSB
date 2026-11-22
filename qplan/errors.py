from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class QPlanError(Exception):
    """A stable, machine-readable error with a location when applicable."""

    code: str
    message: str
    location: str | None = None
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        prefix = f"{self.code}"
        if self.location:
            prefix += f" at {self.location}"
        return f"{prefix}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.location is not None:
            value["location"] = self.location
        if self.details is not None:
            value["details"] = self.details
        return value


def spec_error(location: str, message: str) -> QPlanError:
    return QPlanError("E_SPEC", message, location)

