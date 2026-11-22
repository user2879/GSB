"""Business error codes and the exception type used by the lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_SCHEME_UNKNOWN = "E_SCHEME_UNKNOWN"
E_MIX_RANGE = "E_MIX_RANGE"
E_SCALE_RANGE = "E_SCALE_RANGE"
E_FIELD_TRUNCATED = "E_FIELD_TRUNCATED"
E_MODEL_BUDGET_EXHAUSTED = "E_MODEL_BUDGET_EXHAUSTED"
E_TRAIN_BUDGET_EXHAUSTED = "E_TRAIN_BUDGET_EXHAUSTED"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_MODEL_EXCEPTION = "E_MODEL_EXCEPTION"
E_INTERNAL = "E_INTERNAL"


@dataclass
class LabError(Exception):
    """An expected failure with a stable machine-readable business code."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


def fail(code: str, message: str, **details: Any) -> None:
    raise LabError(code, message, details)
