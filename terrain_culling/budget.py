"""Error-budget bookkeeping and the deterministic degradation ladder."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import F32_REL_EPS, FRUSTUM_FAR, ABS_FLOOR


@dataclass
class DegradationStep:
    step: int
    hiz_level_offset: int
    expansion_multiplier: float
    preload_rings: int
    label: str

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "hiz_level_offset": self.hiz_level_offset,
            "expansion_multiplier": self.expansion_multiplier,
            "preload_rings": self.preload_rings,
            "label": self.label,
        }


# Fixed-priority ladder; the same triggers always produce the same sequence.
LADDER = (
    DegradationStep(0, 0, 1.0, 2, "baseline"),
    DegradationStep(1, 1, 1.0, 2, "coarser_hiz"),
    DegradationStep(2, 1, 2.0, 2, "larger_conservative_expansion"),
    DegradationStep(3, 2, 2.0, 0, "minimal_resident_set"),
)


def base_expansion_margin() -> float:
    """Conservative world-space AABB margin derived from float32 matrix error."""
    return float(max(F32_REL_EPS * FRUSTUM_FAR, ABS_FLOOR))


class DegradationPolicy:
    def __init__(self, cap: int = 3, budget_utilization_trigger: float = 0.9,
                 over_budget_trigger: int = 4):
        self.cap = int(cap)
        self.budget_utilization_trigger = float(budget_utilization_trigger)
        self.over_budget_trigger = int(over_budget_trigger)
        self.step = 0
        self.sequence_log: list[dict] = []

    def current(self) -> DegradationStep:
        return LADDER[self.step]

    def update(self, frame_index: int, max_budget_utilization: float,
               over_budget_count: int) -> DegradationStep:
        before = self.step
        triggered: list[str] = []
        if max_budget_utilization >= self.budget_utilization_trigger:
            triggered.append("memory_budget_pressure")
        if over_budget_count >= self.over_budget_trigger:
            triggered.append("decision_error_over_budget")
        if triggered and self.step < self.cap:
            self.step += 1
        if self.step != before:
            record = {"frame": frame_index, "from_step": before,
                      "to_step": self.step, "triggers": triggered,
                      "new_state": self.current().to_dict()}
            self.sequence_log.append(record)
        return self.current()

    def reset(self) -> None:
        self.step = 0


class ErrorBookkeeper:
    """Summarizes per-frame decision errors against the declared budget."""

    BIN_EDGES = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 1.01)

    def __init__(self, decision_budget: float):
        self.decision_budget = float(decision_budget)

    def summarize(self, decisions: list) -> dict:
        rel_errors: list[float] = []
        over_budget = 0
        for decision in decisions:
            if not np.isfinite(decision.nearest_depth) or decision.nearest_depth <= ABS_FLOOR:
                # At/inside the near plane the relative model is undefined; the
                # block is unconditionally kept and excluded from ratio stats.
                continue
            rel = decision.total_error / decision.nearest_depth
            rel_errors.append(rel)
            if rel > self.decision_budget:
                over_budget += 1
        if not rel_errors:
            histogram = [0] * (len(self.BIN_EDGES) - 1)
            return {"samples": 0, "max_relative_error": 0.0,
                    "mean_relative_error": 0.0, "budget": self.decision_budget,
                    "budget_remaining": self.decision_budget,
                    "over_budget_count": 0, "histogram": histogram}
        arr = np.asarray(rel_errors, dtype=np.float64)
        counts = [0] * (len(self.BIN_EDGES) - 1)
        for value in rel_errors:
            for i in range(len(counts)):
                if self.BIN_EDGES[i] <= value < self.BIN_EDGES[i + 1]:
                    counts[i] += 1
                    break
        return {
            "samples": int(arr.size),
            "max_relative_error": float(arr.max()),
            "mean_relative_error": float(arr.mean()),
            "budget": self.decision_budget,
            "budget_remaining": float(max(self.decision_budget - arr.max(), 0.0)),
            "over_budget_count": int(over_budget),
            "histogram": counts,
            "histogram_edges": list(self.BIN_EDGES),
        }
