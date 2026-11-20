"""Resource budgets and the fixed-rule degradation path.

Three budgets are enforced, in this fixed priority order:

1. ``max_workers``  - parallel degree; clamp to the limit, degrade to
   serial (1 worker) when the limit is 1.
2. ``max_buffer_bytes`` - frame-block buffer; shrink blocks (halving,
   minimum 1 frame) until the estimated buffer fits.
3. ``max_frame_cost`` - single-frame compute budget; frames whose cost
   unit exceeds the limit cannot be scheduled at any block size, so the
   offending task is rejected with ``E_BUDGET_DENIED``.

Degradation only changes scheduling shape.  Frame math lives in pure
block functions, so assembled poses and content hashes are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import E_BUDGET_DENIED

# float64 channels per frame: 4*J quaternion floats + 3 root floats,
# plus two (F, J) float64 diagnostic arrays and a small fixed overhead.
BYTES_PER_FRAME_FACTOR = 8.0
BYTES_PER_FRAME_FIXED = 256


@dataclass(frozen=True)
class Budget:
    max_workers: int
    max_buffer_bytes: int
    max_frame_cost: int  # abstract compute units per frame


@dataclass
class BudgetDecision:
    workers: int
    block_size: int
    estimated_buffer_bytes: int
    degradation: list[dict[str, Any]] = field(default_factory=list)
    denied: list[dict[str, Any]] = field(default_factory=list)


def frame_cost_units(joint_count: int) -> int:
    """Deterministic single-frame compute cost unit."""

    return joint_count * 16 + 64


def estimate_buffer_bytes(frame_count: int, joint_count: int) -> int:
    per_frame = BYTES_PER_FRAME_FACTOR * joint_count + BYTES_PER_FRAME_FIXED
    return int(frame_count * per_frame)


class BudgetPlanner:
    def __init__(self, budget: Budget):
        if budget.max_workers < 1 or budget.max_buffer_bytes < 1:
            raise ValueError("budget limits must be positive")
        self.budget = budget

    def plan(
        self,
        *,
        requested_workers: int,
        requested_block_size: int,
        total_frames: int,
        joint_count: int,
        stage: str,
    ) -> BudgetDecision:
        record: list[dict[str, Any]] = []

        workers_before = max(1, requested_workers)
        workers = min(workers_before, self.budget.max_workers)
        if workers != workers_before:
            record.append(
                {
                    "stage": stage,
                    "rule": "clamp_workers",
                    "before_parallelism": workers_before,
                    "after_parallelism": workers,
                    "reason": "max_workers budget",
                }
            )
        if workers == 1 and workers_before > 1:
            record.append(
                {
                    "stage": stage,
                    "rule": "serial_fallback",
                    "before_parallelism": workers_before,
                    "after_parallelism": 1,
                    "reason": "max_workers budget forced serial mode",
                }
            )

        block_size = max(1, requested_block_size)
        effective_block = min(block_size, total_frames)
        buffer_bytes = estimate_buffer_bytes(effective_block, joint_count)
        while buffer_bytes > self.budget.max_buffer_bytes and effective_block > 1:
            before = effective_block
            effective_block = max(1, effective_block // 2)
            buffer_bytes = estimate_buffer_bytes(effective_block, joint_count)
            record.append(
                {
                    "stage": stage,
                    "rule": "shrink_block",
                    "before_block_size": before,
                    "after_block_size": effective_block,
                    "reason": "max_buffer_bytes budget",
                    "estimated_buffer_bytes": buffer_bytes,
                }
            )

        cost = frame_cost_units(joint_count)
        denied: list[dict[str, Any]] = []
        if cost > self.budget.max_frame_cost:
            entry = {
                "stage": stage,
                "rule": "reject_frame",
                "frame_cost": cost,
                "frame_budget": self.budget.max_frame_cost,
                "reason": "max_frame_cost exceeded; no block size can fit",
            }
            denied.append(entry)
            raise E_BUDGET_DENIED(
                "frame compute budget exceeded",
                entry,
            )

        if buffer_bytes > self.budget.max_buffer_bytes:
            entry = {
                "stage": stage,
                "rule": "reject_block",
                "block_size": effective_block,
                "estimated_buffer_bytes": buffer_bytes,
                "buffer_budget": self.budget.max_buffer_bytes,
                "reason": "single-frame block still exceeds buffer budget",
            }
            denied.append(entry)
            raise E_BUDGET_DENIED("block buffer budget exceeded", entry)

        return BudgetDecision(
            workers=workers,
            block_size=effective_block,
            estimated_buffer_bytes=buffer_bytes,
            degradation=record,
            denied=denied,
        )

    def submit_task(
        self,
        *,
        stage: str,
        frame_count: int,
        joint_count: int,
    ) -> dict[str, Any]:
        """Adversarial entry point used to demonstrate the denial chain."""

        cost = frame_cost_units(joint_count) * frame_count
        buffer_bytes = estimate_buffer_bytes(frame_count, joint_count)
        if frame_cost_units(joint_count) > self.budget.max_frame_cost:
            entry = {
                "stage": stage,
                "rule": "reject_frame",
                "frame_cost": frame_cost_units(joint_count),
                "frame_budget": self.budget.max_frame_cost,
            }
            raise E_BUDGET_DENIED("frame compute budget exceeded", entry)
        if buffer_bytes > self.budget.max_buffer_bytes:
            entry = {
                "stage": stage,
                "rule": "reject_block",
                "frame_count": frame_count,
                "estimated_buffer_bytes": buffer_bytes,
                "buffer_budget": self.budget.max_buffer_bytes,
            }
            raise E_BUDGET_DENIED("block buffer budget exceeded", entry)
        return {
            "stage": stage,
            "accepted": True,
            "frame_count": frame_count,
            "cost_units": cost,
            "estimated_buffer_bytes": buffer_bytes,
        }
