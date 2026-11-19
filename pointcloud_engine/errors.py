"""Business error codes and the structured engine exception.

Every validation or runtime failure is represented by an ``EngineError``
carrying a stable ``E_xxx`` code. The CLI serialises these as JSON; no output
image is produced when an error is raised before successful completion.
"""

from __future__ import annotations

import json
from typing import Any


ERROR_CODES = {
    "E_SCENE": "scene JSON is missing, unreadable or malformed",
    "E_SCHEMA": "a scene field has the wrong type or structure",
    "E_POINTCLOUD": "point cloud file is missing, unreadable or malformed",
    "E_FINITE": "a point coordinate, grayscale or normal is not finite",
    "E_GRAY": "grayscale value is outside the closed interval [0, 1]",
    "E_NORMAL": "a normal vector is the zero vector",
    "E_SCALE": "world scale must be a finite positive number",
    "E_CAMERA": "camera configuration is degenerate (parallel up/look-at vectors)",
    "E_FOV": "vertical field of view must be inside the open interval (0, 180) degrees",
    "E_DIM": "a resolution or tile specification is not a positive integer",
    "E_TILE": "tile grid does not completely cover the supersampled pixel grid",
    "E_RADIUS": "point radius must be a finite non-negative number (zero injected)",
    "E_DECAY": "depth decay parameters must be finite and well formed",
    "E_MEMORY": "in-flight sample buffer limit cannot accommodate even a single tile",
    "E_INJECT": "unknown or malformed fault injection directive",
    "E_DIGEST": "persisted block digest or declared pixel count does not match",
    "E_TILE_RUNTIME": "a render tile failed at runtime",
    "E_IO": "atomic state or image write failed",
}


class EngineError(Exception):
    """Exception carrying a stable business error code and context payload."""

    def __init__(self, code: str, message: str | None = None, **context: Any) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown error code {code!r}")
        self.code = code
        self.context = context
        detail = message or ERROR_CODES[code]
        super().__init__(detail)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": False, "error": {"code": self.code, "message": str(self)}}
        if self.context:
            payload["error"]["context"] = self.context
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def ensure(condition: bool, code: str, message: str | None = None, **context: Any) -> None:
    """Raise an :class:`EngineError` when ``condition`` is falsy."""
    if not condition:
        raise EngineError(code, message, **context)


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and _math_isfinite(value)


def _math_isfinite(value: Any) -> bool:
    import math

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False
