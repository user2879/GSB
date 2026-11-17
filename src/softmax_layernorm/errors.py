from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BUSINESS_CODES = {
    "E_CONFIG_INVALID",
    "E_SHAPE_MISMATCH",
    "E_AXIS_OUT_OF_RANGE",
    "E_EPS_INVALID",
    "E_EXP_OVERFLOW",
    "E_ZERO_VARIANCE",
    "E_INTERACT_BUDGET_PENDING",
    "E_CONFIG_MISMATCH",
    "E_INTERNAL",
}

STAGE_ORDER = {"input": 0, "softmax": 1, "layernorm": 2, "command": 3}


@dataclass(frozen=True)
class BusinessFailure:
    code: str
    message: str
    stage: str
    call_id: str | None = None
    call_sequence: int = 0
    lane: int | None = None
    row: int | None = None
    command_index: int | None = None
    command_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def order_key(self) -> tuple[Any, ...]:
        lane = -1 if self.lane is None else self.lane
        command_index = -1 if self.command_index is None else self.command_index
        return (
            self.call_sequence,
            lane,
            STAGE_ORDER.get(self.stage, 99),
            command_index,
            self.code,
            self.message,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "call_id": self.call_id,
            "call_sequence": self.call_sequence,
            "lane": self.lane,
            "row": self.row,
            "command_index": self.command_index,
            "command_id": self.command_id,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BusinessFailure":
        return cls(
            code=str(raw["code"]),
            message=str(raw["message"]),
            stage=str(raw["stage"]),
            call_id=raw.get("call_id"),
            call_sequence=int(raw.get("call_sequence", 0)),
            lane=raw.get("lane"),
            row=raw.get("row"),
            command_index=raw.get("command_index"),
            command_id=raw.get("command_id"),
            details=dict(raw.get("details", {})),
        )


class BusinessError(Exception):
    """An expected validation or snapshot error carrying one or more failures."""

    def __init__(self, failures: list[BusinessFailure] | BusinessFailure, message: str | None = None):
        if isinstance(failures, BusinessFailure):
            failures = [failures]
        self.failures = sorted(failures, key=lambda item: item.order_key())
        self.code = self.failures[0].code if self.failures else "E_INTERNAL"
        super().__init__(message or self.failures[0].message if self.failures else "business error")
