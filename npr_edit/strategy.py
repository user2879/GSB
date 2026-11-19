"""Reproducible per-tier rasterisation strategy selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .geometry import FlatElement
from .model import Scene
from .util import sha256_json

TIER_BUDGETS: dict[int, dict[str, float]] = {
    200: {"frame_ms": 12.0, "memory_mb": 64.0},
    2000: {"frame_ms": 40.0, "memory_mb": 128.0},
    20000: {"frame_ms": 220.0, "memory_mb": 512.0},
}
CANDIDATE_NAMES = ("per_element", "layer_merge", "tile_batch")


@dataclass(frozen=True, slots=True)
class StrategyStats:
    element_count: int
    layer_count: int
    point_count: int
    avg_bbox_area_px: float
    tile_count: int
    viewport: tuple[int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "element_count": self.element_count,
            "layer_count": self.layer_count,
            "point_count": self.point_count,
            "avg_bbox_area_px": round(self.avg_bbox_area_px, 6),
            "tile_count": self.tile_count,
            "viewport": list(self.viewport),
        }


@dataclass(frozen=True, slots=True)
class StrategyCandidate:
    name: str
    estimated_ms: float
    memory_mb: float
    estimated_error_px: float
    feasible: bool
    score: float
    budget_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "estimated_ms": round(self.estimated_ms, 6),
            "memory_mb": round(self.memory_mb, 6),
            "estimated_error_px": self.estimated_error_px,
            "feasible": self.feasible,
            "score": round(self.score, 6),
            "budget_reason": self.budget_reason,
        }


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    tier: int
    chosen: str
    degraded: bool
    reason: str
    frame_budget_ms: float
    memory_budget_mb: float
    stats: StrategyStats
    candidates: tuple[StrategyCandidate, ...]
    fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "chosen": self.chosen,
            "degraded": self.degraded,
            "reason": self.reason,
            "frame_budget_ms": self.frame_budget_ms,
            "memory_budget_mb": self.memory_budget_mb,
            "stats": self.stats.as_dict(),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "fingerprint": self.fingerprint,
        }


def scene_render_stats(scene: Scene, flats: Mapping[str, FlatElement], viewport: tuple[int, int], tile_size: int = 64) -> StrategyStats:
    if not flats:
        return StrategyStats(len(scene.elements), len(scene.layers), 0, 0.0, 0, viewport)
    scale = min(viewport[0] / scene.canvas_width, viewport[1] / scene.canvas_height)
    source_tile = max(1, int(round(tile_size / max(scale, 1e-9))))
    tile_ids: set[tuple[int, int]] = set()
    point_count = 0
    area_sum = 0.0
    for flat in flats.values():
        point_count += len(flat.points)
        area_sum += max(0.0, (flat.bbox.width * scale) * (flat.bbox.height * scale))
        x0 = int(flat.bbox.min_x // source_tile)
        y0 = int(flat.bbox.min_y // source_tile)
        x1 = int(flat.bbox.max_x // source_tile)
        y1 = int(flat.bbox.max_y // source_tile)
        for tile_x in range(x0, x1 + 1):
            for tile_y in range(y0, y1 + 1):
                tile_ids.add((tile_x, tile_y))
    return StrategyStats(
        element_count=len(scene.elements),
        layer_count=len(scene.layers),
        point_count=point_count,
        avg_bbox_area_px=area_sum / max(1, len(flats)),
        tile_count=len(tile_ids),
        viewport=viewport,
    )


def _candidates(stats: StrategyStats, frame_ms: float, memory_mb: float, layers: int, width: int, height: int) -> tuple[StrategyCandidate, ...]:
    n = max(1, stats.element_count)
    points = max(1, stats.point_count)
    area = max(64.0, stats.avg_bbox_area_px)
    levels = max(1, layers)
    tile_area = 64 * 64

    per_time = 0.7 + 0.022 * n + 0.00012 * points
    per_memory = 0.1 + (n * area * 4.0 * 1.25) / 1_000_000.0
    layer_time = 3.0 + 0.009 * n + 0.00008 * points + levels * 0.1
    layer_memory = (levels * width * height * 4.0 + n * 128.0) / 1_000_000.0
    tile_time = 7.0 + 0.006 * n + 0.00006 * points + stats.tile_count * 0.7
    tile_memory = (stats.tile_count * tile_area * 4.0 + n * 96.0) / 1_000_000.0

    raw = (
        ("per_element", per_time, per_memory, 0.0),
        ("layer_merge", layer_time, layer_memory, 0.02),
        ("tile_batch", tile_time, tile_memory, 0.08),
    )
    result: list[StrategyCandidate] = []
    for name, time_ms, memory, error in raw:
        reasons: list[str] = []
        if time_ms > frame_ms:
            reasons.append("frame_budget")
        if memory > memory_mb:
            reasons.append("memory_budget")
        feasible = not reasons
        score = time_ms + error * 5.0
        result.append(
            StrategyCandidate(
                name=name,
                estimated_ms=time_ms,
                memory_mb=memory,
                estimated_error_px=error,
                feasible=feasible,
                score=score,
                budget_reason="within_budget" if feasible else "+".join(reasons),
            )
        )
    return tuple(result)


class StrategySelector:
    """Caches decisions by tier + scene fingerprint.

    A budget change for one tier cannot invalidate another tier because the tier
    is part of the cache key.
    """

    def __init__(self) -> None:
        self._cache: dict[str, StrategyDecision] = {}
        self.selection_calls = 0
        self.cache_hits = 0

    @property
    def cache_entries(self) -> dict[str, StrategyDecision]:
        return dict(self._cache)

    def select(
        self,
        tier: int,
        stats: StrategyStats,
        budget: Mapping[str, float] | None = None,
    ) -> StrategyDecision:
        selected_budget = dict(TIER_BUDGETS.get(tier, {"frame_ms": 50.0, "memory_mb": 256.0}))
        if budget:
            selected_budget.update({key: float(value) for key, value in budget.items()})
        frame_ms = float(selected_budget["frame_ms"])
        memory_mb = float(selected_budget["memory_mb"])
        fingerprint = sha256_json(
            {
                "tier": tier,
                "stats": stats.as_dict(),
                "frame_ms": round(frame_ms, 6),
                "memory_mb": round(memory_mb, 6),
            }
        )
        if fingerprint in self._cache:
            self.cache_hits += 1
            return self._cache[fingerprint]
        self.selection_calls += 1
        candidates = _candidates(stats, frame_ms, memory_mb, stats.layer_count, stats.viewport[0], stats.viewport[1])
        feasible = [candidate for candidate in candidates if candidate.feasible]
        degraded = not feasible
        pool = feasible or list(candidates)
        chosen = min(pool, key=lambda item: (item.score, CANDIDATE_NAMES.index(item.name)))
        if degraded:
            ratio = min(
                max(candidate.estimated_ms / max(frame_ms, 1e-9), candidate.memory_mb / max(memory_mb, 1e-9))
                for candidate in candidates
            )
            reason = f"no candidate is inside both budgets; selected lowest score (degradation ratio {ratio:.3f})"
        else:
            reason = (
                f"selected {chosen.name}: score {chosen.score:.3f} is the lowest feasible score "
                f"under {frame_ms:.1f} ms and {memory_mb:.1f} MB"
            )
        decision = StrategyDecision(
            tier=tier,
            chosen=chosen.name,
            degraded=degraded,
            reason=reason,
            frame_budget_ms=frame_ms,
            memory_budget_mb=memory_mb,
            stats=stats,
            candidates=candidates,
            fingerprint=fingerprint,
        )
        self._cache[fingerprint] = decision
        return decision
