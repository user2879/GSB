"""Root-cause candidate association along the dependency graph."""
from __future__ import annotations

from .models import AggregateAlert, Correlation, RootCandidate
from .rules import RuleSpec
from .topology import DependencyGraph

_DIRECTION_RANK = {"self": 0, "upstream": 1, "downstream": 2}


def _candidate_sort_key(graph: DependencyGraph, cand: RootCandidate):
    return (cand.layer, _DIRECTION_RANK[cand.direction],
            graph.topo_rank[cand.service_id], cand.service_id)


def correlate_one(aggregate: AggregateAlert, graph: DependencyGraph,
                  rules: RuleSpec) -> Correlation:
    origin = aggregate.service_id
    matched = rules.matching_rules(origin)
    if matched:
        include_up = any(r.include_upstream for r in matched)
        include_down = any(r.include_downstream for r in matched)
        max_layer = min(r.max_layer for r in matched)
    else:
        # Documented default when no rule matches: upstream only, unbounded.
        include_up, include_down, max_layer = True, False, 2 ** 31 - 1

    candidates: list[RootCandidate] = [
        RootCandidate(origin, "self", 0, (origin,))]

    if include_up:
        for svc, path in graph.shortest_paths_from(origin).items():
            if svc == origin or len(path) - 1 > max_layer:
                continue
            candidates.append(RootCandidate(
                svc, "upstream", len(path) - 1, path))
    if include_down:
        for svc, path in graph.shortest_paths_to(origin).items():
            if svc == origin or len(path) - 1 > max_layer:
                continue
            candidates.append(RootCandidate(
                svc, "downstream", len(path) - 1, path))

    # De-duplicate by service id, keep the first occurrence then sort
    # deterministically; output is independent of graph node insertion order.
    unique: dict[str, RootCandidate] = {}
    for cand in candidates:
        unique.setdefault(cand.service_id, cand)
    ordered = sorted(unique.values(),
                     key=lambda c: _candidate_sort_key(graph, c))
    return Correlation(
        aggregate_id=aggregate.aggregate_id,
        rule_version=rules.rule_version,
        candidates=tuple(ordered),
    )


def correlate_all(aggregates: list[AggregateAlert], graph: DependencyGraph,
                  rules: RuleSpec) -> list[Correlation]:
    return [correlate_one(agg, graph, rules) for agg in aggregates]
