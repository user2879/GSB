"""Domain error codes shared by the editor, cache, and persistence layers."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    SCHEMA_VERSION = "E_SCHEMA_VERSION"
    KNOT_MONOTONIC = "E_KNOT_MONOTONIC"
    GEOM_STALE = "E_GEOM_STALE"
    CLOSURE_MISMATCH = "E_CLOSURE_MISMATCH"
    UNDO_EMPTY = "E_UNDO_EMPTY"
    SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
    CACHE_HASH = "E_CACHE_HASH"


class DomainError(Exception):
    """A business failure with a stable machine-readable error code."""

    def __init__(self, code: ErrorCode, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = ErrorCode(code)
        self.message = message
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }


def schema_error(message: str, **details: Any) -> DomainError:
    return DomainError(ErrorCode.SCHEMA_VERSION, message, **details)


def knot_error(message: str, **details: Any) -> DomainError:
    return DomainError(ErrorCode.KNOT_MONOTONIC, message, **details)
