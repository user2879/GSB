"""Business error model used by every public entry point."""

from __future__ import annotations

from typing import Any

E_SCHEMA_VERSION = "E_SCHEMA_VERSION"
E_FORMAT_ENDIAN = "E_FORMAT_ENDIAN"
E_FORMAT_INDEX = "E_FORMAT_INDEX"
E_FORMAT_WIDTH = "E_FORMAT_WIDTH"
E_STROKE_SAMPLE = "E_STROKE_SAMPLE"
E_OUTLINE_DEGENERATE = "E_OUTLINE_DEGENERATE"
E_CLOSURE_MISMATCH = "E_CLOSURE_MISMATCH"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_GL_CONTEXT = "E_GL_CONTEXT"
E_EDIT_TARGET = "E_EDIT_TARGET"
E_RECOVERY = "E_RECOVERY"
E_WORKSPACE = "E_WORKSPACE"

BUSINESS_CODES = frozenset(
    {
        E_SCHEMA_VERSION,
        E_FORMAT_ENDIAN,
        E_FORMAT_INDEX,
        E_FORMAT_WIDTH,
        E_STROKE_SAMPLE,
        E_OUTLINE_DEGENERATE,
        E_CLOSURE_MISMATCH,
        E_SNAPSHOT_STALE,
        E_GL_CONTEXT,
        E_EDIT_TARGET,
        E_RECOVERY,
        E_WORKSPACE,
    }
)


class NPRError(Exception):
    """A stable machine-readable business failure."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        if code not in BUSINESS_CODES:
            raise ValueError(f"unknown NPR error code: {code}")
        self.code = code
        self.message = message
        self.details = details
        super().__init__(f"{code}: {message}")

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.code, "message": self.message, "details": self.details}


def fail(code: str, message: str, **details: Any) -> "None":
    raise NPRError(code, message, **details)
