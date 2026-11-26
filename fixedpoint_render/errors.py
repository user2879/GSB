"""Business error codes (E_*).

Validation failures are collected as :class:`RenderError` entries.  Every entry
carries a stable ``E_`` code, a locator (file + field path) and a human readable
message.  The loader never enters the rendering stage when at least one entry is
present.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RenderError(Exception):
    code: str
    location: str
    field: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.code} @ {self.location}#{self.field}: {self.message}"

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "location": self.location,
            "field": self.field,
            "message": self.message,
        }


class ErrorBundle(Exception):
    """Raised when one or more validation errors block rendering."""

    def __init__(self, errors: list[RenderError]):
        self.errors = list(errors)
        super().__init__("; ".join(str(e) for e in self.errors))

    def codes(self) -> list[str]:
        return [e.code for e in self.errors]

    def count_by_code(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for err in self.errors:
            out[err.code] = out.get(err.code, 0) + 1
        return out


# Allowed codes (kept as constants so typos fail loudly).
E_CONFIG_PARSE = "E_CONFIG_PARSE"
E_CONFIG_FIELD = "E_CONFIG_FIELD"
E_FILE_MISSING = "E_FILE_MISSING"
E_FILE_IO = "E_FILE_IO"
E_PNG_HEADER = "E_PNG_HEADER"
E_PNG_RESOLUTION = "E_PNG_RESOLUTION"
E_PNG_MODE = "E_PNG_MODE"
E_VOCAB_TERM = "E_VOCAB_TERM"
E_PROMPT_PARSE = "E_PROMPT_PARSE"
E_MOTION_PARSE = "E_MOTION_PARSE"
E_MOTION_TILES = "E_MOTION_TILES"
E_MOTION_FIELD = "E_MOTION_FIELD"
E_DIM_MISMATCH = "E_DIM_MISMATCH"
E_QMODE_ENUM = "E_QMODE_ENUM"
E_QBITS_RANGE = "E_QBITS_RANGE"
E_BUDGET_POSITIVE = "E_BUDGET_POSITIVE"
E_WEIGHT_VALUE = "E_WEIGHT_VALUE"
E_COMMAND_PARSE = "E_COMMAND_PARSE"
E_COMMAND_TARGET = "E_COMMAND_TARGET"
E_TILE_BOUNDARY = "E_TILE_BOUNDARY"
E_SNAPSHOT_VERSION = "E_SNAPSHOT_VERSION"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"
E_STATE_INCONSISTENT = "E_STATE_INCONSISTENT"
