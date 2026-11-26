"""Typed, machine-readable errors used by the routing workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_SPEC = "E_SPEC"
E_PORT = "E_PORT"
E_TIMEOUT = "E_TIMEOUT"
E_LEASE = "E_LEASE"
E_REPLAY = "E_REPLAY"
E_WORKER = "E_WORKER"
E_REPLY = "E_REPLY"


@dataclass
class RouteError(Exception):
    """An error with a stable code, field path, and explanation.

    ``path`` identifies the first invalid protocol/spec field and ``details``
    contains only deterministic JSON-compatible values so errors can be logged
    and compared byte-for-byte.
    """

    code: str
    path: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, f"{self.code} at {self.path}: {self.reason}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "path": self.path,
            "reason": self.reason,
            "details": self.details,
        }


def spec_error(path: str, reason: str, **details: Any) -> RouteError:
    return RouteError(E_SPEC, path, reason, details)


def replay_error(seq: int | None, field: str, reason: str, **details: Any) -> RouteError:
    return RouteError(
        E_REPLAY,
        f"events[seq={seq}].{field}" if seq is not None else f"report.{field}",
        reason,
        {"event_seq": seq, "field": field, **details},
    )

