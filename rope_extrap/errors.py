from __future__ import annotations

from dataclasses import dataclass
from typing import Any


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_SAMPLE_INVALID = "E_SAMPLE_INVALID"
E_DIM_MISMATCH = "E_DIM_MISMATCH"
E_BASE_INVALID = "E_BASE_INVALID"
E_SCALE_INVALID = "E_SCALE_INVALID"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_INTERACT_BUDGET_PENDING = "E_INTERACT_BUDGET_PENDING"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_COMPUTE_NONFINITE = "E_COMPUTE_NONFINITE"
E_INTERNAL = "E_INTERNAL"

STAGE_ORDER = {
    "load": 0,
    "position_encoding": 10,
    "attention_score": 20,
    "error_trace": 30,
    "snapshot": 40,
}


class BusinessError(Exception):
    """An expected diagnostic failure that maps to a stable business code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str = "compute",
        sample_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.sample_id = sample_id

    def as_failure(self, *, line: int | None = None, scope: str = "runtime") -> "Failure":
        return Failure(
            code=self.code,
            message=self.message,
            stage=self.stage,
            sample_id=self.sample_id,
            line=line,
            scope=scope,
        )


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    stage: str
    sample_id: str | None = None
    line: int | None = None
    scope: str = "sample"
    detail: str | None = None

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.sample_id is not None,
            self.sample_id or "",
            STAGE_ORDER.get(self.stage, 99),
            self.code,
            self.line if self.line is not None else -1,
            self.message,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "scope": self.scope,
            "sample_id": self.sample_id,
            "line": self.line,
            "detail": self.detail,
        }

