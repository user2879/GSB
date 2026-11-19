from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineError(Exception):
    """A business-coded error with a JSON field path."""

    code: str
    message: str
    path: str = "$"
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }
        if self.details:
            payload["details"] = self.details
        return payload


def fail(code: str, path: str, message: str, **details: Any) -> "NoReturn":  # type: ignore[name-defined]
    raise PipelineError(code=code, path=path, message=message, details=details)

