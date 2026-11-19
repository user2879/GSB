"""Business error codes and the failure-record container.

Codes are stable strings so reports and manifests remain diff-friendly across
runs. Every failure captured by the pipeline is converted into a
:class:`FailureRecord` carrying a stable sort key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ErrorCode:
    E_INDEX_INVALID = "E_INDEX_INVALID"
    E_TILE_SHAPE_MISMATCH = "E_TILE_SHAPE_MISMATCH"
    E_TILE_READ_FAILED = "E_TILE_READ_FAILED"
    E_NON_FINITE_BAND = "E_NON_FINITE_BAND"
    E_RECOMPUTE_BUDGET_PENDING = "E_RECOMPUTE_BUDGET_PENDING"
    E_SNAPSHOT_INVALID = "E_SNAPSHOT_INVALID"
    E_CONFIG_INVALID = "E_CONFIG_INVALID"
    E_INJECTION_FAULT = "E_INJECTION_FAULT"


@dataclass(frozen=True)
class FailureRecord:
    """One diagnostic entry; ordering matches row-major single-thread runs."""

    code: str
    target: str
    message: str
    sort_key: tuple[int, int]
    injection_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "target": self.target,
            "message": self.message,
            "sort_key": list(self.sort_key),
            "injection_id": self.injection_id,
            "detail": self.detail,
        }


class BusinessError(Exception):
    """Raised for fatal business-level validation failures."""

    def __init__(self, code: str, message: str, *, injection_id: str | None = None,
                 detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.injection_id = injection_id
        self.detail = detail or {}
