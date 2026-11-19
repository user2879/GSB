"""Business errors with stable codes and auditable context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


EXIT_CODES: dict[str, int] = {
    "E_FRAME_FORMAT": 10,
    "E_REGION_BOUNDS": 11,
    "E_DEPENDENCY_STALE": 12,
    "E_CLOSURE_MISMATCH": 13,
    "E_SNAPSHOT_VERSION": 14,
    "E_MASK_INVALID": 15,
}


@dataclass
class ForensicError(Exception):
    """A domain error that must never be replaced by a generic exception."""

    code: str
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in EXIT_CODES:
            raise ValueError(f"unknown business error code: {self.code}")
        Exception.__init__(self, f"{self.code}: {self.message}")
        self.context = dict(self.context)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.code]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "context": self.context,
                "exit_code": self.exit_code,
            },
        }


def frame_format(message: str, **context: Any) -> ForensicError:
    return ForensicError("E_FRAME_FORMAT", message, context)


def region_bounds(message: str, **context: Any) -> ForensicError:
    return ForensicError("E_REGION_BOUNDS", message, context)


def dependency_stale(message: str, **context: Any) -> ForensicError:
    return ForensicError("E_DEPENDENCY_STALE", message, context)


def closure_mismatch(message: str, **context: Any) -> ForensicError:
    return ForensicError("E_CLOSURE_MISMATCH", message, context)


def snapshot_version(message: str, **context: Any) -> ForensicError:
    return ForensicError("E_SNAPSHOT_VERSION", message, context)


def mask_invalid(message: str, **context: Any) -> ForensicError:
    return ForensicError("E_MASK_INVALID", message, context)

