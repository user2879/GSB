"""Business error codes used by the solver and report writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_OBS_INVALID = "E_OBS_INVALID"
E_CAMERA_INVALID = "E_CAMERA_INVALID"
E_CFG_INVALID = "E_CFG_INVALID"
E_TRIANG_DEGENERATE = "E_TRIANG_DEGENERATE"
E_DISPARITY_INSUFFICIENT = "E_DISPARITY_INSUFFICIENT"
E_INLIER_SHORTAGE = "E_INLIER_SHORTAGE"
E_POINT_QUOTA_EXHAUSTED = "E_POINT_QUOTA_EXHAUSTED"
E_RESIDUAL_BUDGET_EXHAUSTED = "E_RESIDUAL_BUDGET_EXHAUSTED"
E_IO = "E_IO"
E_SNAPSHOT = "E_SNAPSHOT"

ALL_ERROR_CODES = (
    E_OBS_INVALID,
    E_CAMERA_INVALID,
    E_CFG_INVALID,
    E_TRIANG_DEGENERATE,
    E_DISPARITY_INSUFFICIENT,
    E_INLIER_SHORTAGE,
    E_POINT_QUOTA_EXHAUSTED,
    E_RESIDUAL_BUDGET_EXHAUSTED,
    E_IO,
    E_SNAPSHOT,
)


@dataclass
class Failure:
    """A serializable business failure.

    ``view_index`` and ``observation_index`` are optional so the same record can
    describe global input failures, per-observation failures, and point-level
    failures.
    """

    code: str
    keypoint_id: str | None
    stage: str
    message: str
    view_index: int | None = None
    observation_index: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "code": self.code,
            "keypoint_id": self.keypoint_id,
            "stage": self.stage,
            "message": self.message,
        }
        if self.view_index is not None:
            out["view_index"] = self.view_index
        if self.observation_index is not None:
            out["observation_index"] = self.observation_index
        if self.details:
            out["details"] = self.details
        return out


class SolverFailure(Exception):
    """Exception carrying a :class:`Failure` and optional partial payload."""

    def __init__(self, failure: Failure, partial: dict[str, Any] | None = None):
        super().__init__(failure.message)
        self.failure = failure
        self.partial = partial or {}


class ConfigError(ValueError):
    """Raised when a solver configuration is structurally or semantically bad."""


class BudgetExceeded(SolverFailure):
    """Raised when the per-keypoint residual evaluation quota is exhausted."""
