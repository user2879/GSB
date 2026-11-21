"""Structured errors used across the workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkbenchError(Exception):
    """A stable, machine-readable workbench failure."""

    code: str
    message: str
    revision: int | None = None
    field_path: str | None = None
    layer: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "message": self.message,
            "revision": self.revision,
            "field_path": self.field_path,
            "layer": self.layer,
            "details": self.details,
        }


class SimulatedCrash(RuntimeError):
    """Raised after a deliberately interrupted run has been persisted."""


E_TRUST = "E_TRUST"
E_HANDSHAKE = "E_HANDSHAKE"
E_CHAIN = "E_CHAIN"
E_REVOKE = "E_REVOKE"
E_SNAPSHOT = "E_SNAPSHOT"
E_REPLAY = "E_REPLAY"
E_ROLLBACK = "E_ROLLBACK"
E_PORT = "E_PORT"
E_UI = "E_UI"

