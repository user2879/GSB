"""Business error codes and one structured exception type."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    OUTLINE_INVALID = "E_OUTLINE_INVALID"
    FILL_RULE = "E_FILL_RULE"
    TASK_KEY = "E_TASK_KEY"
    ATLAS_FULL = "E_ATLAS_FULL"
    BUDGET_DENIED = "E_BUDGET_DENIED"
    SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
    BITMAP_HASH = "E_BITMAP_HASH"


@dataclass(frozen=True)
class GlyphForgeError(RuntimeError):
    """A deterministic, serializable business error.

    ``details`` deliberately contains only JSON-compatible values.  This keeps
    diagnostics and the demo ledger byte-reproducible.
    """

    code: ErrorCode
    message: str
    glyph_id: str | None = None
    segment_index: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)

    def __str__(self) -> str:
        suffix = []
        if self.glyph_id is not None:
            suffix.append(f"glyph={self.glyph_id}")
        if self.segment_index is not None:
            suffix.append(f"segment={self.segment_index}")
        if suffix:
            return f"{self.code.value}: {self.message} ({', '.join(suffix)})"
        return f"{self.code.value}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code.value,
            "message": self.message,
        }
        if self.glyph_id is not None:
            result["glyph_id"] = self.glyph_id
        if self.segment_index is not None:
            result["segment_index"] = self.segment_index
        if self.details:
            result["details"] = dict(self.details)
        return result


def invalid_outline(
    message: str,
    *,
    glyph_id: str | None = None,
    segment_index: int | None = None,
    **details: Any,
) -> GlyphForgeError:
    return GlyphForgeError(
        ErrorCode.OUTLINE_INVALID,
        message,
        glyph_id=glyph_id,
        segment_index=segment_index,
        details=details,
    )


def fill_rule_error(message: str, **details: Any) -> GlyphForgeError:
    return GlyphForgeError(ErrorCode.FILL_RULE, message, details=details)


def task_key_error(message: str, **details: Any) -> GlyphForgeError:
    return GlyphForgeError(ErrorCode.TASK_KEY, message, details=details)


def atlas_full(message: str, **details: Any) -> GlyphForgeError:
    return GlyphForgeError(ErrorCode.ATLAS_FULL, message, details=details)


def budget_denied(message: str, **details: Any) -> GlyphForgeError:
    return GlyphForgeError(ErrorCode.BUDGET_DENIED, message, details=details)


def snapshot_stale(message: str, **details: Any) -> GlyphForgeError:
    return GlyphForgeError(ErrorCode.SNAPSHOT_STALE, message, details=details)


def bitmap_hash_error(message: str, **details: Any) -> GlyphForgeError:
    return GlyphForgeError(ErrorCode.BITMAP_HASH, message, details=details)
