"""Per-frame denoise step budget arbitration (business rule 3).

The allocator is pure fixed-point integer math with a fixed tie-break order
(frame index ascending).  Thread scheduling cannot influence the outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import RenderConfig
from .fixedpoint import q_div_round, q_from_int, q_scale
from .fusion import FusionResult
from .encoding import EncodedCondition

# Scoring constants (integer percent / fixed point weights).
COVERAGE_WEIGHT = 60          # percent of score from condition coverage
MOTION_WEIGHT = 40            # percent of score from motion strength
SCORE_DEN = 100


@dataclass
class FrameAllocation:
    frame: int
    demand: int                # requested steps (== per-frame budget b)
    score_q: int               # fixed point priority score
    raw_share_q: int           # proportional share before truncation
    allocated: int             # integer steps granted
    truncated: int             # demand - allocated (>= 0)
    degraded: bool
    fill_steps: int            # missing steps filled deterministically later
    rounding_basis: str
    redistributed: int = 0

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "demand": self.demand,
            "score_q": self.score_q,
            "raw_share_q": self.raw_share_q,
            "allocated": self.allocated,
            "truncated": self.truncated,
            "degraded": self.degraded,
            "fill_steps": self.fill_steps,
            "rounding_basis": self.rounding_basis,
            "redistributed": self.redistributed,
        }


@dataclass
class BudgetPlan:
    per_frame_budget: int
    global_budget: int
    total_demand: int
    total_allocated: int
    truncated_total: int
    frame_budgets: dict[int, int]   # per-frame overrides from edits
    entries: list[FrameAllocation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "per_frame_budget": self.per_frame_budget,
            "global_budget": self.global_budget,
            "total_demand": self.total_demand,
            "total_allocated": self.total_allocated,
            "truncated_total": self.truncated_total,
            "frame_budgets": self.frame_budgets,
            "entries": [e.to_dict() for e in self.entries],
        }


def _motion_metric(motion: EncodedCondition, frac_bits: int) -> int:
    total_q = sum(abs(v) for v in motion.vector)
    return q_div_round(total_q, q_scale(frac_bits), "trunc")


def score_frame(motion: EncodedCondition, fusion: FusionResult,
                config: RenderConfig) -> int:
    """Fixed point priority score from coverage and motion strength."""
    coverage = motion.coverage  # 0..100 integer percent
    # Normalize motion metric to a 0..100 scale against the per-frame budget
    # ceiling so scores stay bounded and comparable across configs.
    metric = _motion_metric(motion, config.frac_bits)
    cap = max(1, config.per_frame_budget * len(config.vocab))
    motion_pct = min(100, q_div_round(metric * 100, cap, "trunc"))
    score = (coverage * COVERAGE_WEIGHT + motion_pct * MOTION_WEIGHT) // SCORE_DEN
    return q_from_int(score, config.frac_bits)


def allocate(frame_count: int,
             encoded_motion: dict[int, EncodedCondition],
             fusions: dict[int, FusionResult],
             config: RenderConfig,
             frame_budgets: dict[int, int] | None = None) -> BudgetPlan:
    frame_budgets = dict(frame_budgets or {})
    scale = q_scale(config.frac_bits)
    demands: dict[int, int] = {}
    scores: dict[int, int] = {}
    for frame in range(frame_count):
        demand = min(frame_budgets.get(frame, config.per_frame_budget),
                     config.per_frame_budget)
        if demand <= 0:
            demand = config.per_frame_budget
        demands[frame] = demand
        scores[frame] = score_frame(encoded_motion[frame], fusions[frame], config)

    total_demand = sum(demands.values())
    global_budget = min(config.global_step_budget, total_demand)
    score_sum = sum(scores.values())

    entries: list[FrameAllocation] = []
    if score_sum == 0:
        # Deterministic uniform fallback.
        for frame in range(frame_count):
            share_q = q_div_round(q_from_int(global_budget, config.frac_bits),
                                  frame_count, "trunc")
            entries.append(FrameAllocation(
                frame=frame, demand=demands[frame], score_q=0,
                raw_share_q=share_q, allocated=0, truncated=0,
                degraded=False, fill_steps=0,
                rounding_basis="uniform_fallback_zero_score"))
    else:
        for frame in range(frame_count):
            # raw proportional share in Q: global_budget * score / score_sum
            numerator = scores[frame] * global_budget
            raw_share_q = q_div_round(numerator, score_sum, "trunc")
            entries.append(FrameAllocation(
                frame=frame, demand=demands[frame], score_q=scores[frame],
                raw_share_q=raw_share_q, allocated=0, truncated=0,
                degraded=False, fill_steps=0,
                rounding_basis="trunc(global*score/score_sum)"))

    # First pass: truncate shares to integer steps, capped by per-frame demand.
    spent = 0
    for entry in entries:
        truncated_share = q_div_round(entry.raw_share_q, scale, "trunc")
        grant = min(truncated_share, entry.demand)
        entry.allocated = grant
        entry.rounding_basis += f"; trunc_q->int cap_demand={entry.demand}"
        spent += grant

    # Leftover redistribution: highest score first, ties broken by frame index.
    leftover = global_budget - spent
    order = sorted(range(frame_count),
                   key=lambda f: (-entries[f].score_q, f))
    cursor = 0
    while leftover > 0 and cursor < len(order) * 2:
        frame = order[cursor % len(order)]
        entry = entries[frame]
        if entry.allocated < entry.demand:
            entry.allocated += 1
            entry.redistributed += 1
            leftover -= 1
        cursor += 1
    # If every frame is saturated, leftover is simply unused (cannot exceed cap).

    for entry in entries:
        entry.truncated = entry.demand - entry.allocated
        entry.degraded = entry.truncated > 0
        entry.fill_steps = entry.truncated
        if entry.degraded:
            entry.rounding_basis += "; degraded: missing steps filled in kernel"

    plan = BudgetPlan(
        per_frame_budget=config.per_frame_budget,
        global_budget=global_budget,
        total_demand=total_demand,
        total_allocated=sum(e.allocated for e in entries),
        truncated_total=sum(e.truncated for e in entries),
        frame_budgets=frame_budgets,
        entries=entries,
    )
    # Hard invariant: never exceed the global or per-frame budgets.
    assert plan.total_allocated <= global_budget
    for entry in entries:
        assert entry.allocated <= entry.demand
    return plan
