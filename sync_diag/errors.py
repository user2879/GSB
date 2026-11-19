from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ERROR_CODES = (
    "E_CH_FORMAT",
    "E_PULSE_MISSING",
    "E_DRIFT_EXCEED",
    "E_RESAMPLE_PHASE",
    "E_STRATEGY_BUDGET",
    "E_SNAPSHOT_STALE",
    "E_MANIFEST_HASH",
    "E_CONFIG",
    "E_IO",
)


@dataclass
class SyncError(Exception):
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in ERROR_CODES:
            raise ValueError(f"unknown error code: {self.code}")
        super().__init__(f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "context": self.context,
            },
        }


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise SyncError(code, message, context)
