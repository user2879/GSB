from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BUSINESS_CODES = (
    "E_SCRIPT",
    "E_LOCKTIMEOUT",
    "E_DEADLOCK",
    "E_ISOLATION",
    "E_SNAPSHOT",
    "E_REPLAY",
)


@dataclass
class BusinessError(Exception):
    """A user-facing, machine-readable domain error."""

    code: str
    message: str
    path: str = "$"
    session_id: str | None = None
    op_no: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in BUSINESS_CODES:
            raise ValueError(f"unknown business code: {self.code}")
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "session_id": self.session_id,
            "op_no": self.op_no,
            "details": self.details,
        }

    def __str__(self) -> str:
        location = []
        if self.session_id is not None:
            location.append(f"session={self.session_id}")
        if self.op_no is not None:
            location.append(f"op={self.op_no}")
        if self.path:
            location.append(f"path={self.path}")
        suffix = f" ({', '.join(location)})" if location else ""
        return f"{self.code}: {self.message}{suffix}"
