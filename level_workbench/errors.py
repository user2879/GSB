"""Stable, serializable error values used by every public entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkbenchError(Exception):
    code: str
    message: str
    path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.path is not None:
            payload["path"] = self.path
        if self.details:
            payload["details"] = self.details
        return payload


class SpecError(WorkbenchError):
    """A specification was rejected before generation started."""


class InjectionError(WorkbenchError):
    """An injection configuration is malformed or cannot be applied."""


class GenerationError(WorkbenchError):
    """The generator could not produce a candidate satisfying its constraints."""


class VerificationError(WorkbenchError):
    """The level artifact is malformed before constraint verification starts."""


def error_dict(exc: WorkbenchError) -> dict[str, Any]:
    return exc.to_dict()
