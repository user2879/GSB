"""Stable business errors used by the quantisation calibration pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ERROR_EXIT_CODES = {
    "E_FORMAT_INVALID": 11,
    "E_ZERO_POINT_INVALID": 12,
    "E_QUANT_RANGE_INVALID": 13,
    "E_NON_FINITE_SAMPLE": 14,
    "E_CALIB_BUDGET_PENDING": 15,
    "E_SNAPSHOT_STALE": 16,
    "E_TENSOR_FAILED": 17,
    "E_INTERNAL": 19,
}


@dataclass
class QuantCalError(Exception):
    code: str
    message: str
    tensor: str | None = None
    stage: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.tensor is not None:
            payload["tensor"] = self.tensor
        if self.stage is not None:
            payload["stage"] = self.stage
        if self.details:
            payload["details"] = self.details
        return payload


class SnapshotStale(QuantCalError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__("E_SNAPSHOT_STALE", message, stage="snapshot", details=details)

