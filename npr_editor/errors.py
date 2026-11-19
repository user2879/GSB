"""Business error codes shared by every entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ERROR_CODES = frozenset(
    {
        "E_SCHEMA_VERSION",
        "E_FORMAT_ENDIAN",
        "E_FORMAT_INDEX",
        "E_STROKE_SAMPLE",
        "E_UNDO_EMPTY",
        "E_SNAPSHOT_STALE",
        "E_RENDER_MISMATCH",
        # Additional structural errors are kept separate so callers never have
        # to guess which malformed binary field caused a failure.
        "E_FORMAT_VERSION",
        "E_FORMAT_DATA",
    }
)


@dataclass(init=False)
class NprError(Exception):
    """A stable, serializable application error."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        **context: Any,
    ) -> None:
        merged = dict(details or {})
        merged.update(context)
        self.code = code
        self.message = message
        self.details = merged
        if self.code not in ERROR_CODES:
            raise ValueError(f"unknown NPR error code: {self.code}")
        Exception.__init__(self, f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


def require(condition: bool, code: str, message: str, **details: Any) -> None:
    if not condition:
        raise NprError(code, message, details)
