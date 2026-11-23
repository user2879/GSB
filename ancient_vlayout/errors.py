"""Stable error codes and a small exception type used by the engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_RUNTIME_INVALID = "E_RUNTIME_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_IMAGE_INVALID = "E_IMAGE_INVALID"
E_TRUTH_INVALID = "E_TRUTH_INVALID"
E_TEMPLATE_INVALID = "E_TEMPLATE_INVALID"
E_TEMPLATE_MISSING = "E_TEMPLATE_MISSING"
E_SEGMENT_TOO_SHORT = "E_SEGMENT_TOO_SHORT"
E_SPLIT_INFEASIBLE = "E_SPLIT_INFEASIBLE"
E_MA_BUDGET_PENDING = "E_MA_BUDGET_PENDING"
E_SNAPSHOT_RECOVERED = "E_SNAPSHOT_RECOVERED"
E_SNAPSHOT_INVALID = "E_SNAPSHOT_INVALID"
E_INJECTION_HIT = "E_INJECTION_HIT"
E_DEPENDENCY_VERSION_MISMATCH = "E_DEPENDENCY_VERSION_MISMATCH"


@dataclass
class AppError(Exception):
    """Exception carrying a machine-readable error code and details."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(f"{self.code}: {self.message}")

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}
