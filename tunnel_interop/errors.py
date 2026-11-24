"""Structured, stable error codes used by the public component and CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


E_SPEC = "E_SPEC"
E_FORMAT = "E_FORMAT"
E_FIELD = "E_FIELD"
E_REF = "E_REF"
E_TIMEOUT = "E_TIMEOUT"
E_SNAPSHOT = "E_SNAPSHOT"
E_REPLAY = "E_REPLAY"
E_INTERNAL = "E_INTERNAL"


@dataclass
class TunnelError(Exception):
    """An error that can cross process and CLI boundaries without ambiguity."""

    code: str
    path: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(f"{self.code} at {self.path}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "path": self.path,
                "message": self.message,
                "details": dict(self.details),
            },
        }


def fail(
    code: str,
    path: str,
    message: str,
    **details: Any,
) -> "None":
    raise TunnelError(code=code, path=path, message=message, details=details)

