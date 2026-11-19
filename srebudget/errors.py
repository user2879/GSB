"""Structured, stable error objects used by the workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SreError(Exception):
    """An error with a stable business code and machine-readable details."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, f"{self.code}: {self.message}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


def chain_error(message: str, field_path: str = "$", **details: Any) -> SreError:
    return SreError(
        "E_CHAIN",
        message,
        {"field_path": field_path, **details},
    )


def trace_error(
    message: str, line_number: int, field_path: str = "$", **details: Any
) -> SreError:
    return SreError(
        "E_TRACE",
        message,
        {"line": line_number, "field_path": field_path, **details},
    )


def snapshot_error(message: str, **details: Any) -> SreError:
    return SreError("E_SNAPSHOT", message, details)


def replay_error(
    message: str, *, kind: str = "divergence", **details: Any
) -> SreError:
    return SreError("E_REPLAY", message, {"kind": kind, **details})
