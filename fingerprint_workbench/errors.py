from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ERROR_CODES = (
    "E_AUDIO_FORMAT",
    "E_INDEX_VERSION",
    "E_PARTITION_HASH",
    "E_CONFIG_RANGE",
    "E_SNAPSHOT_STALE",
    "E_BUDGET_DENIED",
    "E_QUERY_DRIFT",
    "E_STATE_CONFLICT",
    "E_NOT_FOUND",
    "E_LOCKED",
)


@dataclass
class WorkbenchError(RuntimeError):
    """A business error with a stable machine-readable code."""

    code: str
    message: str
    detail: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.code not in ERROR_CODES:
            raise ValueError(f"unknown business error code: {self.code}")
        super().__init__(f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.detail is not None:
            result["detail"] = self.detail
        return result
