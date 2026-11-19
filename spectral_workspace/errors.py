"""Business error types used by the application and CLI."""

from __future__ import annotations

from typing import Any


E_SPECTRUM_RANGE = "E_SPECTRUM_RANGE"
E_WHITE_POINT = "E_WHITE_POINT"
E_OBSERVER_INVALID = "E_OBSERVER_INVALID"
E_STAGE_STALE = "E_STAGE_STALE"
E_SNAPSHOT_VERSION = "E_SNAPSHOT_VERSION"
E_INJECT_CONFIG = "E_INJECT_CONFIG"
E_ADAPTATION_INVALID = "E_ADAPTATION_INVALID"
E_COLOR_SPACE_INVALID = "E_COLOR_SPACE_INVALID"
E_OPERATION_INVALID = "E_OPERATION_INVALID"
E_STATE_CORRUPT = "E_STATE_CORRUPT"
E_INPUT_JSON = "E_INPUT_JSON"
E_SCENE_INVALID = "E_SCENE_INVALID"
E_MATERIAL_NOT_FOUND = "E_MATERIAL_NOT_FOUND"
E_LIGHT_NOT_FOUND = "E_LIGHT_NOT_FOUND"


class DomainError(Exception):
    """A deterministic, serialisable business error.

    ``pointer`` follows RFC 6901 style JSON pointers so callers can locate the
    exact raw input element that failed validation.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        pointer: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.pointer = pointer
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.pointer is not None:
            error["pointer"] = self.pointer
        if self.details:
            error["details"] = self.details
        return {"ok": False, "error": error}

    def __str__(self) -> str:
        suffix = f" at {self.pointer}" if self.pointer else ""
        return f"{self.code}: {self.message}{suffix}"


def pointer_escape(value: str) -> str:
    """Escape one JSON Pointer path segment."""

    return str(value).replace("~", "~0").replace("/", "~1")


def join_pointer(base: str, *segments: str | int) -> str:
    """Join already unescaped path segments onto a JSON Pointer."""

    out = base.rstrip("/")
    for segment in segments:
        out += "/" + pointer_escape(str(segment))
    return out
