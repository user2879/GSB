from __future__ import annotations

from collections import deque
from typing import Iterable, Mapping

from .models import (
    AggregateAlert,
    LedgerEntry,
    RootCandidate,
    RootRule,
    RootRules,
    ServiceGraph,
)
from .util import digest_json


def build_adjacency(graph: ServiceGraph) -> dict[str, tuple[str, ...]]:
    upstream: dict[str, list[str]] = {service: [] for service in graph.services}
    downstream: dict[str, list[str]] = {service: [] for service in graph.services}
    for source, target in graph.dependencies:
        upstream[source].append(target)
        downstream[target].append(source)
    for service in graph.services:
        upstream[service].sort(key=lambda item: (graph.topology_rank[item], item))
        downstream[service].sort(key=lambda item: (graph.topology_rank[item], item))
    return {
        "upstream": {key: tuple(value) for key, value in upstream.items()},
        "downstream": {key: tuple(value) for key, value in downstream.items()},
    }


def _rule_map(rules: RootRules) -> dict[str, RootRule]:
    result: dict[str, RootRule] = {}
    for rule in rules.rules:
        for service in rule.services:
            result[service] = rule
    return result


def _default_rule(rules: RootRules) -> RootRule:
    directions = tuple(rules.defaults["directions"])
    return RootRule(
        rule_id="default-rule",
        services=tuple(),
        directions=directions,
        max_depth=int(rules.defaults["max_depth"]),
        include_self=bool(rules.defaults["include_self"]),
    )


def derive_candidates(
    service_id: str,
    graph: ServiceGraph,
    rules: RootRules,
    adjacency: Mapping[str, Mapping[str, tuple[str, ...]]] | None = None,
) -> tuple[tuple[RootCandidate, ...], str]:
    graph_adj = adjacency if adjacency is not None else build_adjacency(graph)
    rule = _rule_map(rules).get(service_id, _default_rule(rules))
    best: dict[str, tuple[int, int, RootCandidate]] = {}

    def consider(candidate: RootCandidate) -> None:
        direction_priority = {"upstream": 0, "self": 0, "downstream": 1}[candidate.direction]
        score = (candidate.depth, direction_priority, candidate)
        previous = best.get(candidate.service_id)
        if previous is None or score[:2] < previous[:2]:
            best[candidate.service_id] = (candidate.depth, direction_priority, candidate)

    if rule.include_self:
        consider(
            RootCandidate(
                service_id=service_id,
                direction="self",
                depth=0,
                path=(service_id,),
                evidence=f"self; rule={rule.rule_id}",
            )
        )
    for direction in rule.directions:
        queue: deque[tuple[str, tuple[str, ...]]] = deque([(service_id, (service_id,))])
        visited: dict[str, int] = {service_id: 0}
        while queue:
            current, path = queue.popleft()
            depth = len(path) - 1
            if depth >= rule.max_depth:
                continue
            for neighbor in graph_adj[direction].get(current, tuple()):
                next_depth = depth + 1
                if neighbor in visited and visited[neighbor] <= next_depth:
                    continue
                visited[neighbor] = next_depth
                next_path = path + (neighbor,)
                consider(
                    RootCandidate(
                        service_id=neighbor,
                        direction=direction,
                        depth=next_depth,
                        path=next_path,
                        evidence=(
                            f"{direction}; rule={rule.rule_id}; "
                            f"topology_level={graph.topology_level[neighbor]}"
                        ),
                    )
                )
                queue.append((neighbor, next_path))

    ordered = sorted(
        best.values(),
        key=lambda item: (
            item[2].depth,
            graph.topology_level[item[2].service_id],
            graph.topology_rank[item[2].service_id],
            item[2].service_id,
        ),
    )
    return tuple(item[2] for item in ordered), rule.rule_id


def build_ledger_entry(
    aggregate: AggregateAlert,
    graph: ServiceGraph,
    rules: RootRules,
    declared_refs: Iterable[str] = tuple(),
    adjacency: Mapping[str, Mapping[str, tuple[str, ...]]] | None = None,
) -> LedgerEntry:
    candidates, rule_id = derive_candidates(
        aggregate.service_id, graph, rules, adjacency
    )
    refs = tuple(sorted(set(declared_refs)))
    payload = {
        "alert_id": aggregate.alert_id,
        "aggregate": aggregate.to_public_dict(),
        "candidates": [item.to_public_dict() for item in candidates],
        "rule_id": rule_id,
        "rule_version": rules.version,
        "graph_version": graph.version,
        "declared_version": aggregate.declared_version,
        "format_version": aggregate.declared_format_version,
        "declared_root_refs": list(refs),
    }
    return LedgerEntry(
        alert_id=aggregate.alert_id,
        aggregate=aggregate,
        candidates=candidates,
        rule_id=rule_id,
        rule_version=rules.version,
        graph_version=graph.version,
        declared_version=aggregate.declared_version,
        format_version=aggregate.declared_format_version,
        declared_root_refs=refs,
        content_digest=digest_json(payload),
    )


def build_ledger(
    aggregates: Iterable[AggregateAlert],
    graph: ServiceGraph,
    rules: RootRules,
    alert_sources: Mapping[str, Iterable[str]],
) -> tuple[LedgerEntry, ...]:
    adjacency = build_adjacency(graph)
    entries: list[LedgerEntry] = []
    for aggregate in sorted(
        aggregates,
        key=lambda item: (
            item.first_timestamp_ms,
            item.service_id,
            item.fingerprint,
            item.alert_id,
        ),
    ):
        entries.append(
            build_ledger_entry(
                aggregate,
                graph,
                rules,
                alert_sources.get(aggregate.alert_id, tuple()),
                adjacency,
            )
        )
    return tuple(entries)


def correlate_serial(
    aggregates: Iterable[AggregateAlert],
    graph: ServiceGraph,
    rules: RootRules,
    alerts_by_aggregate: Mapping[str, Iterable[str]] | None = None,
) -> tuple[LedgerEntry, ...]:
    return build_ledger(aggregates, graph, rules, alerts_by_aggregate or {})
