"""Typed errors used across the coordination protocol and engine."""

from __future__ import annotations

from typing import Any


class CoordError(Exception):
    """A stable, machine-readable coordination error."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": self.code,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


def protocol_error(message: str, **details: Any) -> CoordError:
    return CoordError("E_PROTO", message, **details)


def dependency_error(message: str, **details: Any) -> CoordError:
    return CoordError("E_DEPEND", message, **details)


def snapshot_error(message: str, **details: Any) -> CoordError:
    return CoordError("E_SNAPSHOT", message, **details)


def replay_error(message: str, **details: Any) -> CoordError:
    return CoordError("E_REPLAY", message, **details)


def port_error(message: str, **details: Any) -> CoordError:
    return CoordError("E_PORT", message, **details)

