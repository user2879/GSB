"""Business error types and stable error-code constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_CHAIN_INVALID = "E_CHAIN_INVALID"
E_CYCLE_DETECTED = "E_CYCLE_DETECTED"
E_PERM_INVALID = "E_PERM_INVALID"
E_INPLACE_CONFLICT = "E_INPLACE_CONFLICT"
E_RECOMPUTE_BUDGET_PENDING = "E_RECOMPUTE_BUDGET_PENDING"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_ACTIVATION_FAILED = "E_ACTIVATION_FAILED"
E_PERMUTE_FAILED = "E_PERMUTE_FAILED"
E_UPSTREAM_FAILED = "E_UPSTREAM_FAILED"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_INPUT_INVALID = "E_INPUT_INVALID"
E_SNAPSHOT_IO_ERROR = "E_SNAPSHOT_IO_ERROR"
E_RUN_INTERRUPTED = "E_RUN_INTERRUPTED"


@dataclass
class BusinessError(Exception):
    """An error that is safe to expose in the result JSON."""

    code: str
    message: str
    stage: str = "validation"
    node_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def as_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
        }
        if self.node_id is not None:
            record["node_id"] = self.node_id
        if self.details:
            record["details"] = self.details
        return record

