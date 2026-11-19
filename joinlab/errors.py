from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class JoinLabError(Exception):
    """A business error that is safe to serialise to CLI/UI/log consumers."""

    code: str
    message: str
    field_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        if self.field_path is not None:
            payload["field_path"] = self.field_path
        if self.details:
            payload["details"] = self.details
        return payload


def fail(
    code: str,
    message: str,
    field_path: str | None = None,
    **details: Any,
) -> None:
    raise JoinLabError(code=code, message=message, field_path=field_path, details=details)

