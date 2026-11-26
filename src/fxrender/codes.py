"""Stable business codes used across validation, scheduling and snapshots.

Codes are part of the on-disk contract and must never be silently renumbered.
"""

from __future__ import annotations

from enum import Enum


class BizCode(str, Enum):
    OK = "OK"
    E_PROMPT_OOV = "E_PROMPT_OOV"
    E_DIM_MISMATCH = "E_DIM_MISMATCH"
    E_PNG_HEADER = "E_PNG_HEADER"
    E_PNG_MODE = "E_PNG_MODE"
    E_PNG_SIZE = "E_PNG_SIZE"
    E_MOTION_SHAPE = "E_MOTION_SHAPE"
    E_MOTION_VALUE = "E_MOTION_VALUE"
    E_POSITIVE_INT = "E_POSITIVE_INT"
    E_FP_ENUM = "E_FP_ENUM"
    E_BAD_COMMAND = "E_BAD_COMMAND"
    E_BUDGET_PENDING = "E_BUDGET_PENDING"
    E_TILE_BORROW = "E_TILE_BORROW"
    E_STALE_SUMMARY = "E_STALE_SUMMARY"
    E_SNAPSHOT_VERSION = "E_SNAPSHOT_VERSION"
    E_MISSING_FIELD = "E_MISSING_FIELD"
    E_FILE_NOT_FOUND = "E_FILE_NOT_FOUND"
    E_PARSE = "E_PARSE"


class CheckItem:
    """One auditable validation conclusion.

    Invalid items are skipped by the loader; they are never auto-corrected.
    """

    __slots__ = ("source", "index", "code", "detail", "accepted")

    def __init__(self, source: str, index: int | None, code: BizCode,
                 detail: str, accepted: bool) -> None:
        self.source = source
        self.index = index
        self.code = code
        self.detail = detail
        self.accepted = accepted

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "index": self.index,
            "code": self.code.value,
            "detail": self.detail,
            "accepted": self.accepted,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (f"CheckItem(source={self.source!r}, index={self.index}, "
                f"code={self.code.value}, accepted={self.accepted})")
