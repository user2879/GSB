"""Frame-time budget enforcement and deterministic degradation.

Two clock modes are supported:

* ``deterministic``: frame costs come from a reproducible cost model driven by
  point counts, patch counts and the frame's interaction, so reports hash
  byte-identically across machines;
* ``wallclock``: the live PyQt viewport measures real elapsed milliseconds.

Degradation ladder (applied in a fixed order until the simulated cost fits):

1. reduce point draw density (100% -> 50% -> 25%);
2. defer patch refresh (display the previous mesh snapshot);
3. skip non-essential frame updates (jump frames).

Degradation never mutates submitted edit state or reconstruction results.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import config
from .errors import BudgetExceededError

DENSITY_LEVELS = (1.0, 0.5, 0.25)
BASE_COST_MS = 6.0
COST_PER_10K_POINTS = 1.6
COST_PER_PATCH = 0.02
COST_PER_INTERACTION = 4.0
JUMP_FRAME_COST_MS = 2.0


@dataclass
class FrameRecord:
    frame: int
    interaction: str
    raw_cost_ms: float
    chosen_cost_ms: float
    budget_ms: float
    density: float
    patch_refresh_deferred: bool
    frame_skipped: bool
    degradation_reasons: tuple
    edit_version: int

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "interaction": self.interaction,
            "raw_cost_ms": round(self.raw_cost_ms, 6),
            "chosen_cost_ms": round(self.chosen_cost_ms, 6),
            "budget_ms": self.budget_ms,
            "density": self.density,
            "patch_refresh_deferred": self.patch_refresh_deferred,
            "frame_skipped": self.frame_skipped,
            "degradation_reasons": list(self.degradation_reasons),
            "edit_version": self.edit_version,
        }


class DeterministicClock:
    """Reproducible per-frame cost model (no wall-clock reads)."""

    def raw_cost(self, live_points: int, active_patches: int, interaction: str) -> float:
        cost = (
            BASE_COST_MS
            + COST_PER_10K_POINTS * live_points / 10_000
            + COST_PER_PATCH * active_patches
        )
        if interaction in ("box_select", "drag", "param_change"):
            cost += COST_PER_INTERACTION
        # Deterministic pseudo variation from frame-independent inputs.
        cost *= 1.0 + 0.08 * ((live_points + active_patches) % 7) / 7.0
        return round(cost, 6)


class WallClock:
    def __init__(self):
        self._start = None

    def raw_cost(self, live_points: int, active_patches: int, interaction: str) -> float:
        now = time.perf_counter()
        if self._start is None:
            self._start = now
        return round((now - self._start) * 1000.0, 6)


class FrameBudgeter:
    def __init__(
        self,
        budget_ms: float = config.FRAME_BUDGET_MS,
        *,
        deterministic: bool = True,
    ):
        self.budget_ms = float(budget_ms)
        self.clock = DeterministicClock() if deterministic else WallClock()
        self.records: list[FrameRecord] = []
        self._frame = 0
        self._deferred_patch_refresh = False

    def plan_frame(self, interaction: str, live_points: int, active_patches: int, edit_version: int) -> FrameRecord:
        raw = self.clock.raw_cost(live_points, active_patches, interaction)
        cost = raw
        density_idx = 0
        deferred = False
        skipped = False
        reasons: list[str] = []

        while cost > self.budget_ms and density_idx < len(DENSITY_LEVELS) - 1:
            density_idx += 1
            cost = self._scaled_cost(raw, DENSITY_LEVELS[density_idx])
            reasons.append("reduce_point_density")

        if cost > self.budget_ms and not self._deferred_patch_refresh:
            deferred = True
            self._deferred_patch_refresh = True
            cost *= 0.7
            reasons.append("defer_patch_refresh")
        elif self._deferred_patch_refresh and cost <= self.budget_ms:
            # The deferred mesh has caught up; latch may re-arm on a later frame.
            self._deferred_patch_refresh = False

        if cost > self.budget_ms:
            skipped = True
            cost = JUMP_FRAME_COST_MS
            reasons.append("skip_frame")

        if cost > self.budget_ms:
            raise BudgetExceededError(
                f"frame {self._frame} cannot meet budget {self.budget_ms} ms",
                context={"raw_cost_ms": raw, "best_cost_ms": cost},
            )

        record = FrameRecord(
            frame=self._frame,
            interaction=interaction,
            raw_cost_ms=raw,
            chosen_cost_ms=round(cost, 6),
            budget_ms=self.budget_ms,
            density=DENSITY_LEVELS[density_idx],
            patch_refresh_deferred=deferred,
            frame_skipped=skipped,
            degradation_reasons=tuple(reasons),
            edit_version=edit_version,
        )
        self.records.append(record)
        self._frame += 1
        return record

    @staticmethod
    def _scaled_cost(raw: float, density: float) -> float:
        # Geometry portion scales with draw density; fixed overhead does not.
        return BASE_COST_MS + (raw - BASE_COST_MS) * density

    def report(self) -> dict:
        if not self.records:
            return {"frame_count": 0}
        costs = [r.chosen_cost_ms for r in self.records]
        raw = [r.raw_cost_ms for r in self.records]
        degraded = [r for r in self.records if r.degradation_reasons]
        return {
            "budget_ms": self.budget_ms,
            "frame_count": len(self.records),
            "max_chosen_cost_ms": round(max(costs), 6),
            "mean_chosen_cost_ms": round(sum(costs) / len(costs), 6),
            "over_budget_raw_frames": sum(1 for v in raw if v > self.budget_ms),
            "degraded_frame_count": len(degraded),
            "density_levels_used": sorted({r.density for r in self.records}),
            "deferred_patch_refresh_frames": sum(1 for r in self.records if r.patch_refresh_deferred),
            "skipped_frames": sum(1 for r in self.records if r.frame_skipped),
            "frames": [r.to_dict() for r in self.records],
        }
