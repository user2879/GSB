from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MigrationError(Exception):
    """A machine-readable failure with a stable code and location."""

    code: str
    message: str
    step_id: str | None = None
    field_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "step_id": self.step_id,
            "field_path": self.field_path,
            "details": self.details,
        }


def fail(
    code: str,
    message: str,
    *,
    step_id: str | None = None,
    field_path: str | None = None,
    **details: Any,
) -> None:
    raise MigrationError(code, message, step_id=step_id, field_path=field_path, details=details)

