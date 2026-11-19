"""Incremental versus full-rebuild consistency checks."""

from __future__ import annotations

from typing import Any

from .config import AppConfig
from .engine import TierEngine, path_node_id, region_node_id
from .errors import ErrorCode, TacticalError
from .models import Route, StrategyDecision


def _first_difference(left: tuple[int, ...], right: tuple[int, ...]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _dependency_edge(
    engine: TierEngine,
    query_id: str,
    divergence_cell: int | None,
    affected_regions: list[int],
) -> str | None:
    node_id = path_node_id(engine.tier, query_id)
    node = engine.graph.nodes.get(node_id)
    if node is None:
        return None
    preferred: list[str] = []
    if divergence_cell is not None:
        region_id = engine.map.region_of_cell(divergence_cell)
        preferred.append(region_node_id(engine.tier, region_id))
    for region_id in affected_regions:
        preferred.append(region_node_id(engine.tier, region_id))
    for dependency in preferred:
        if dependency in node.deps:
            return f"{node_id} -> {dependency}"
    if node.deps:
        return f"{node_id} -> {node.deps[0]}"
    return None


def compare_incremental_with_full(
    engine: TierEngine,
    incremental_routes: dict[str, Route],
    decisions: dict[str, StrategyDecision],
    config: AppConfig,
    *,
    affected_regions: list[int] | None = None,
) -> dict[str, Any]:
    full_engine = TierEngine(engine.bundle, config)
    full_result = full_engine.solve_batch(
        engine.bundle.queries,
        decisions,
        config.budgets[engine.tier],
        force=True,
    )
    full_routes: dict[str, Route] = full_result["routes"]
    max_cost_deviation = 0
    first_divergence: dict[str, Any] | None = None
    mismatches: list[dict[str, Any]] = []
    compared = 0
    for query in sorted(engine.bundle.queries, key=lambda item: item.id):
        incremental = incremental_routes.get(query.id)
        full = full_routes.get(query.id)
        if incremental is None or full is None:
            raise TacticalError(
                ErrorCode.CLOSURE_MISMATCH,
                "incremental or full rebuild is missing a query result",
                context={
                    "query_id": query.id,
                    "incremental": incremental is not None,
                    "full": full is not None,
                },
            )
        compared += 1
        deviation = abs(int(incremental.cost) - int(full.cost))
        max_cost_deviation = max(max_cost_deviation, deviation)
        scale = max(1, abs(int(full.cost)))
        cost_ok = float(deviation) <= config.cost_tolerance * scale
        sequence_difference = _first_difference(incremental.path, full.path)
        sequence_equal = sequence_difference is None
        consistent = cost_ok and sequence_equal
        if not consistent:
            divergence_cell = None
            if sequence_difference is not None:
                if sequence_difference < len(incremental.path):
                    divergence_cell = int(incremental.path[sequence_difference])
                elif sequence_difference < len(full.path):
                    divergence_cell = int(full.path[sequence_difference])
            item = {
                "query_id": query.id,
                "incremental_cost": incremental.cost,
                "full_cost": full.cost,
                "cost_ok": cost_ok,
                "sequence_equal": sequence_equal,
                "sequence_difference_index": sequence_difference,
                "divergence_cell": divergence_cell,
                "divergence_xy": None
                if divergence_cell is None
                else [divergence_cell % engine.map.width, divergence_cell // engine.map.width],
                "dependency_edge": _dependency_edge(
                    engine,
                    query.id,
                    divergence_cell,
                    affected_regions or [],
                ),
            }
            mismatches.append(item)
            if first_divergence is None:
                first_divergence = item
    return {
        "tier": engine.tier,
        "queries_compared": compared,
        "all_consistent": not mismatches,
        "cost_tolerance": config.cost_tolerance,
        "sequence_tolerance": config.sequence_tolerance,
        "sequence_rule": "exact path sequence; equal cost alone is insufficient",
        "max_cost_deviation": max_cost_deviation,
        "first_divergence": first_divergence,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "incremental_route_hashes": {
            query.id: incremental_routes[query.id].route_hash(engine.map.width)
            for query in sorted(engine.bundle.queries, key=lambda item: item.id)
        },
        "full_rebuild_route_hashes": {
            query.id: full_routes[query.id].route_hash(engine.map.width)
            for query in sorted(engine.bundle.queries, key=lambda item: item.id)
        },
    }

