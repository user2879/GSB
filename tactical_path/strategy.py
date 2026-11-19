"""Deterministic, budget-aware solver strategy selection."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .canonical import sha256_json
from .config import AppConfig
from .errors import ErrorCode, TacticalError
from .models import (
    Budgets,
    MapData,
    PlanResult,
    Query,
    StrategyCandidate,
    StrategyDecision,
)
from .solvers.flowfield import flow_eligible


ESTIMATOR_VERSION = "strategy-estimator-v1"


@dataclass
class _MapProfile:
    cells: int
    open_ratio: float
    region_area: int


class StrategyPlanner:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._cache: dict[str, tuple[str, StrategyDecision]] = {}

    def _profile(self, map_data: MapData) -> _MapProfile:
        return _MapProfile(
            cells=map_data.cells,
            open_ratio=float(map_data.passable.mean()),
            region_area=map_data.region_size * map_data.region_size,
        )

    def _candidate_dense(
        self, query: Query, profile: _MapProfile, budget: Budgets
    ) -> StrategyCandidate:
        dx = abs(query.start % int(math.sqrt(profile.cells)) - query.target % int(math.sqrt(profile.cells)))
        dy = abs(query.start // int(math.sqrt(profile.cells)) - query.target // int(math.sqrt(profile.cells)))
        distance = dx + dy
        branching = 2.0 + 6.0 * (1.0 - profile.open_ratio)
        expansions = int(distance * branching + 64)
        estimated_ms = expansions * self.config.estimator["ns_per_expansion"] / 1_000_000.0
        estimated_bytes = expansions * 96 + 4096
        feasible = (
            estimated_ms <= budget.query_time_ms
            and estimated_bytes <= int(budget.query_memory_mib * 1024 * 1024)
        )
        reason = (
            f"exact; {expansions} est. expansions, {estimated_ms:.3f} ms, "
            f"{estimated_bytes} bytes"
        )
        if not feasible:
            reason += "; exceeds declared query time or memory budget"
        return StrategyCandidate(
            strategy="dense_astar",
            estimated_ms=round(estimated_ms, 6),
            estimated_bytes=estimated_bytes,
            feasible=feasible,
            error_upper_bound=0.0,
            reason=reason,
        )

    def _candidate_flow(
        self,
        query: Query,
        map_data: MapData,
        profile: _MapProfile,
        budget: Budgets,
        group_size: int,
    ) -> StrategyCandidate:
        if not flow_eligible(map_data, query):
            return StrategyCandidate(
                strategy="region_flow",
                estimated_ms=0.0,
                estimated_bytes=0,
                feasible=False,
                error_upper_bound=0.0,
                reason="not applicable: source and target are in different tactical regions",
            )
        width = map_data.width
        distance = abs(query.start % width - query.target % width) + abs(
            query.start // width - query.target // width
        )
        precompute_ms = (
            profile.region_area
            * self.config.estimator["flow_ns_per_cell"]
            / 1_000_000.0
        )
        amortized_ms = precompute_ms / max(1, group_size)
        descent_ms = (distance + 2) * 400 / 1_000_000.0
        estimated_ms = amortized_ms + descent_ms
        estimated_bytes = profile.region_area * 13 + 4096
        feasible = (
            estimated_ms <= budget.query_time_ms
            and estimated_bytes <= int(budget.query_memory_mib * 1024 * 1024)
        )
        reason = (
            f"exact regional field; shared by {group_size} queries; "
            f"amortized {estimated_ms:.3f} ms, {estimated_bytes} bytes"
        )
        if not feasible:
            reason += "; exceeds declared query time or memory budget"
        return StrategyCandidate(
            strategy="region_flow",
            estimated_ms=round(estimated_ms, 6),
            estimated_bytes=estimated_bytes,
            feasible=feasible,
            error_upper_bound=0.0,
            reason=reason,
        )

    def _candidate_jump(
        self,
        query: Query,
        map_data: MapData,
        profile: _MapProfile,
        budget: Budgets,
    ) -> StrategyCandidate:
        if profile.cells < self.config.cjp_min_cells:
            return StrategyCandidate(
                strategy="chunk_jump",
                estimated_ms=0.0,
                estimated_bytes=0,
                feasible=False,
                error_upper_bound=0.0,
                reason=(
                    f"not applicable below cjp_min_cells={self.config.cjp_min_cells}"
                ),
            )
        width = map_data.width
        distance = abs(query.start % width - query.target % width) + abs(
            query.start // width - query.target // width
        )
        crossings = int(math.ceil(distance / map_data.region_size)) + 2
        portals_per_region = 4
        precompute_ms = (
            crossings
            * portals_per_region
            * profile.region_area
            * self.config.estimator["jump_ns_per_cell"]
            / 1_000_000.0
        )
        high_level_ms = crossings * portals_per_region * 0.02 + 0.05
        estimated_ms = high_level_ms
        estimated_bytes = crossings * portals_per_region * profile.region_area * 9
        feasible = (
            estimated_ms <= budget.query_time_ms
            and estimated_bytes <= int(budget.query_memory_mib * 1024 * 1024)
            and precompute_ms <= budget.precompute_time_ms
        )
        reason = (
            f"exact portal jump; {crossings} regions, query {estimated_ms:.3f} ms, "
            f"precompute {precompute_ms:.3f} ms, {estimated_bytes} bytes"
        )
        if not feasible:
            reason += "; exceeds query or precompute budget"
        return StrategyCandidate(
            strategy="chunk_jump",
            estimated_ms=round(estimated_ms, 6),
            estimated_bytes=estimated_bytes,
            feasible=feasible,
            error_upper_bound=0.0,
            reason=reason,
        )

    def _group_counts(
        self, map_data: MapData, queries: list[Query]
    ) -> dict[tuple[int, int, str], int]:
        counts: dict[tuple[int, int, str], int] = {}
        for query in queries:
            if not flow_eligible(map_data, query):
                continue
            key = (map_data.region_of_cell(query.start), query.target, query.mobility)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _signature(
        self,
        query: Query,
        map_data: MapData,
        budget: Budgets,
        group_size: int,
    ) -> str:
        payload = {
            "estimator": ESTIMATOR_VERSION,
            "tier": map_data.tier,
            "query": [
                query.id,
                query.start,
                query.target,
                query.mobility,
                query.unit_id,
            ],
            "map": [
                map_data.width,
                map_data.height,
                map_data.region_size,
                round(float(map_data.passable.mean()), 12),
            ],
            "budget": budget.to_dict(),
            "group_size": group_size,
            "cjp_min_cells": self.config.cjp_min_cells,
            "estimator_values": self.config.estimator,
        }
        return sha256_json(payload)

    def _decide(
        self,
        query: Query,
        map_data: MapData,
        budget: Budgets,
        group_size: int,
        signature: str,
    ) -> StrategyDecision:
        profile = self._profile(map_data)
        candidates = [
            self._candidate_flow(query, map_data, profile, budget, group_size),
            self._candidate_jump(query, map_data, profile, budget),
            self._candidate_dense(query, profile, budget),
        ]
        feasible = [candidate for candidate in candidates if candidate.feasible]
        if feasible:
            chosen = min(
                feasible,
                key=lambda item: (item.estimated_ms, item.estimated_bytes, item.strategy),
            )
        else:
            applicable = [item for item in candidates if item.estimated_ms > 0]
            if not applicable:
                raise TacticalError(
                    ErrorCode.BUDGET_DENIED,
                    "no solver strategy is applicable under the declared budgets",
                    context={"query_id": query.id, "tier": map_data.tier},
                )
            chosen = min(
                applicable,
                key=lambda item: (item.estimated_ms, item.estimated_bytes, item.strategy),
            )
        rationale = (
            f"selected {chosen.strategy}: {chosen.reason}; "
            f"time budget={budget.query_time_ms:.3f} ms, "
            f"memory budget={budget.query_memory_mib:.3f} MiB; "
            f"{len(feasible)} feasible candidate(s) evaluated deterministically"
        )
        return StrategyDecision(
            query_id=query.id,
            strategy=chosen.strategy,
            estimated_ms=chosen.estimated_ms,
            estimated_bytes=chosen.estimated_bytes,
            error_upper_bound=chosen.error_upper_bound,
            rationale=rationale,
            candidates=tuple(candidates),
            signature=signature,
        )

    def plan(
        self,
        map_data: MapData,
        queries: list[Query],
        budget: Budgets,
    ) -> PlanResult:
        budget.validate()
        groups = self._group_counts(map_data, queries)
        decisions: dict[str, StrategyDecision] = {}
        recomputed: list[str] = []
        reused: list[str] = []
        active_ids = {query.id for query in queries}
        for query_id in list(self._cache):
            if query_id not in active_ids:
                del self._cache[query_id]
        for query in sorted(queries, key=lambda item: item.id):
            group_size = groups.get(
                (map_data.region_of_cell(query.start), query.target, query.mobility),
                1,
            )
            signature = self._signature(query, map_data, budget, group_size)
            cached = self._cache.get(query.id)
            if cached is not None and cached[0] == signature:
                decision = cached[1]
                reused.append(query.id)
            else:
                decision = self._decide(
                    query, map_data, budget, group_size, signature
                )
                self._cache[query.id] = (signature, decision)
                recomputed.append(query.id)
            decisions[query.id] = decision
        return PlanResult(
            decisions=decisions,
            recomputed=tuple(sorted(recomputed)),
            reused=tuple(sorted(reused)),
        )

