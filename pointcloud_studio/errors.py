"""Business error types shared by the CLI, engine, renderer and persistence."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    E_POINT_CLOUD_FORMAT = "E_POINT_CLOUD_FORMAT"
    E_NEIGHBOR_RADIUS = "E_NEIGHBOR_RADIUS"
    E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
    E_MESH_MISMATCH = "E_MESH_MISMATCH"
    E_CACHE_HASH = "E_CACHE_HASH"
    E_GL_CONTEXT = "E_GL_CONTEXT"


class StudioError(RuntimeError):
    """A reproducible business error with machine-readable context."""

    def __init__(
        self,
        code: ErrorCode | str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.code = ErrorCode(code)
        self.message = message
        self.context = context or {}
        super().__init__(f"{self.code}: {message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code.value,
            "message": self.message,
            "context": self.context,
        }


def error_result(code: ErrorCode | str, message: str, **context: Any) -> dict[str, Any]:
    return StudioError(code, message, context=context).to_dict()

