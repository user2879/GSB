"""Stable per-view quotas and global reconstruction-point budgeting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import Failure
from .models import ReconstructionConfig


@dataclass
class BudgetAllocation:
    accepted_view_ids: list[int] = field(default_factory=list)
    pending_queue: list[dict[str, Any]] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    requested_points: int = 0
    consumed_points: int = 0
    remaining_points: int = 0


def allocate_feature_points(
    feature_stage: dict[str, Any],
    config: ReconstructionConfig,
) -> BudgetAllocation:
    allocation = BudgetAllocation()
    remaining = config.global_point_budget
    for view_id_string in sorted(feature_stage.get("views", {}), key=int):
        view_id = int(view_id_string)
        suite = feature_stage["views"][view_id_string]
        required = len(suite.get("points", []))
        allocation.requested_points += required
        if required > config.view_point_quota:
            allocation.failures.append(
                Failure(
                    "E_VIEW_QUOTA_EXCEEDED",
                    view_id=view_id,
                    artifact=f"v:{view_id:04d}",
                    detail=f"required={required},quota={config.view_point_quota}",
                )
            )
            continue
        if required > remaining:
            allocation.pending_queue.append(
                {
                    "view_id": view_id,
                    "sort_key": f"v:{view_id:04d}",
                    "required_points": required,
                    "remaining_before": remaining,
                    "reason": "global_point_budget_insufficient",
                }
            )
            allocation.failures.append(
                Failure(
                    "E_POINT_BUDGET_PENDING",
                    view_id=view_id,
                    artifact=f"v:{view_id:04d}",
                    detail=f"required={required},remaining={remaining}",
                )
            )
            continue
        allocation.accepted_view_ids.append(view_id)
        allocation.consumed_points += required
        remaining -= required
    allocation.remaining_points = remaining
    return allocation

