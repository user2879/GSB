"""Deterministic greedy budget admission in stable sample-ID order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import FusionConfig
from .errors import E_BUDGET_PENDING
from .features import METRIC_COUNT
from .inputs import SampleInput


@dataclass(frozen=True)
class SampleCost:
    macs: int
    points: int

    def as_dict(self) -> dict[str, int]:
        return {"macs": self.macs, "points": self.points}


@dataclass(frozen=True)
class BudgetDecision:
    admitted: list[str]
    pending: list[dict[str, Any]]
    consumed_macs: int
    consumed_points: int
    remaining_macs: int
    remaining_points: int


def estimate_sample_cost(sample: SampleInput, config: FusionConfig) -> SampleCost:
    block_count = (sample.width // config.block_size) * (sample.height // config.block_size)
    assertion_count = max(1, len(sample.assertions))
    visual_macs = block_count * (METRIC_COUNT * config.visual_dim + 2)
    text_macs = assertion_count * (3 * config.text_dim + 2)
    fusion_macs = (config.visual_dim + config.text_dim) * 5 + 64
    one_profile_macs = visual_macs + text_macs + fusion_macs
    macs = one_profile_macs * 2
    points = (block_count * (config.visual_dim + 1) + config.text_dim * (assertion_count + 1)) * 2
    return SampleCost(macs=macs, points=points)


def admit_samples(samples: list[SampleInput], config: FusionConfig) -> BudgetDecision:
    remaining_macs = config.global_macs
    remaining_points = config.batch_points
    admitted: list[str] = []
    pending: list[dict[str, Any]] = []
    consumed_macs = 0
    consumed_points = 0
    for sample in sorted(samples, key=lambda value: value.sample_id):
        cost = estimate_sample_cost(sample, config)
        reasons: list[str] = []
        if cost.macs > remaining_macs:
            reasons.append("global_macs")
        if cost.points > remaining_points:
            reasons.append("batch_points")
        if reasons:
            pending.append(
                {
                    "sample_id": sample.sample_id,
                    "code": E_BUDGET_PENDING,
                    "message": "sample retained for a later budget window",
                    "details": reasons,
                    "required": cost.as_dict(),
                    "remaining": {"macs": remaining_macs, "points": remaining_points},
                }
            )
            continue
        admitted.append(sample.sample_id)
        consumed_macs += cost.macs
        consumed_points += cost.points
        remaining_macs -= cost.macs
        remaining_points -= cost.points
    return BudgetDecision(
        admitted=admitted,
        pending=pending,
        consumed_macs=consumed_macs,
        consumed_points=consumed_points,
        remaining_macs=remaining_macs,
        remaining_points=remaining_points,
    )
