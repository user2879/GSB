"""Stable business error protocol used by the CLI and the GUI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

E_SPEC = "E_SPEC"
E_REF = "E_REF"
E_CYCLE = "E_CYCLE"
E_WIDTH = "E_WIDTH"
E_BUCKET = "E_BUCKET"
E_DIVERGE = "E_DIVERGE"
E_LEDGER = "E_LEDGER"
E_REPLAY = "E_REPLAY"
E_GUI = "E_GUI"
E_INTERNAL = "E_INTERNAL"


@dataclass
class BusinessError(Exception):
    """A deterministic, machine-readable rejection.

    ``path`` is the first offending field whenever the error originates from
    validation.  ``details`` is a JSON-compatible mapping and must never
    contain unordered sets.
    """

    code: str
    path: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, f"{self.code} at {self.path}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "details": dict(self.details),
        }


def fail(
    code: str,
    path: str,
    message: str,
    **details: Any,
) -> None:
    raise BusinessError(code=code, path=path, message=message, details=details)
