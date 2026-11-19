"""Business error codes and exception types for the workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

E_RAY_DEGENERATE = "E_RAY_DEGENERATE"
E_NORMAL_ZERO = "E_NORMAL_ZERO"
E_OBJECT_DUP_ID = "E_OBJECT_DUP_ID"
E_PRIORITY_RANGE = "E_PRIORITY_RANGE"
E_SNAP_SPACING = "E_SNAP_SPACING"
E_PICK_MISMATCH = "E_PICK_MISMATCH"
E_INJECT_CONFIG = "E_INJECT_CONFIG"

ERROR_CODES = (
    E_RAY_DEGENERATE,
    E_NORMAL_ZERO,
    E_OBJECT_DUP_ID,
    E_PRIORITY_RANGE,
    E_SNAP_SPACING,
    E_PICK_MISMATCH,
    E_INJECT_CONFIG,
)

ERROR_MESSAGES = {
    E_RAY_DEGENERATE: "Ray direction is a zero vector.",
    E_NORMAL_ZERO: "A required plane or polygon normal has zero length.",
    E_OBJECT_DUP_ID: "Two or more pickable objects share the same ID.",
    E_PRIORITY_RANGE: "A pick priority is outside the declared priority range.",
    E_SNAP_SPACING: "Snap spacing must be a single positive, uniform value.",
    E_PICK_MISMATCH: "Screen-space picking disagrees with the equivalent ray pick.",
    E_INJECT_CONFIG: "The injection configuration is invalid or cannot be applied.",
}

# This order is part of the deterministic validation contract.
ANOMALY_ORDER = {
    E_RAY_DEGENERATE: 0,
    E_NORMAL_ZERO: 1,
    E_OBJECT_DUP_ID: 2,
    E_PRIORITY_RANGE: 3,
    E_SNAP_SPACING: 4,
    E_PICK_MISMATCH: 5,
    E_INJECT_CONFIG: 6,
}


@dataclass(slots=True)
class WorkbenchError(Exception):
    """A business failure with a stable code, message, and input context."""

    code: str
    message: str
    trigger: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in ERROR_CODES:
            raise ValueError(f"Unknown workbench error code: {self.code}")
        Exception.__init__(self, f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "trigger": self.trigger,
        }


def make_error(code: str, message: str | None = None, **trigger: Any) -> WorkbenchError:
    return WorkbenchError(code, message or ERROR_MESSAGES[code], dict(trigger))


def error_sort_key(error: WorkbenchError) -> tuple[int, str, str]:
    object_id = str(error.trigger.get("object_id", ""))
    return (ANOMALY_ORDER[error.code], object_id, error.message)
