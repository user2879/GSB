from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from typing import Any, Iterable

from .errors import LabError
from .model import JoinPredicate, JoinStep, Plan, Query, Table, Workload


ALGORITHMS = ("HASH", "MERGE", "NESTED_LOOP")
ALGORITHM_ORDER = {name: index for index, name in enumerate(ALGORITHMS)}


@dataclass(frozen=True)
class JoinGraph:
    query_digest: str
    digest: str
    summary_json: str
    adjacency: dict[str, tuple[str, ...]]
    edges: tuple[JoinPredicate, ...]


@dataclass(frozen=True)
class CandidateEnumeration:
    graph: JoinGraph
    plans: tuple[Plan, ...]
    pruned_orders: tuple[dict[str, Any], ...]
    omitted_candidates: tuple[dict[str, Any], ...]
    total_before_limit: int
    applied_limit: int
    baseline: Plan

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(plan.candidate_id for plan in self.plans)


def compile_join_graph(query: Query, tables: dict[str, Table]) -> JoinGraph:
    nodes = []
    for name in sorted(query.tables):
        table = tables[name]
        nodes.append(
            {
                "name": name,
                "columns": [
                    {"name": column.name, "type": column.type} for column in table.columns
                ],
            }
        )
    summary = {
        "nodes": nodes,
        "edges": sorted((edge.canonical() for edge in query.joins), key=_canonical_key),
        "filters": sorted((item.canonical() for item in query.filters), key=_canonical_key),
        "projection": [item.canonical() for item in query.select],
        "group_by": sorted([ref.as_list() for ref in query.group_by]),
    }
    summary_json = _canonical_json(summary)
    summary_digest = hashlib.sha256(summary_json.encode("utf-8")).hexdigest()
    adjacency: dict[str, set[str]] = {name: set() for name in query.tables}
    for edge in query.joins:
        left, right = edge.left.table, edge.right.table
        adjacency[left].add(right)
        adjacency[right].add(left)
    frozen_adjacency = {
        name: tuple(sorted(neighbours)) for name, neighbours in sorted(adjacency.items())
    }
    return JoinGraph(
        query_digest=summary_digest,
        digest=summary_digest,
        summary_json=summary_json,
        adjacency=frozen_adjacency,
        edges=tuple(sorted(query.joins, key=lambda item: item.key)),
    )


def enumerate_candidates(
    workload: Workload,
    graph: JoinGraph,
    *,
    limit: int | None = None,
) -> CandidateEnumeration:
    table_names = tuple(sorted(workload.query.tables))
    table_map = workload.table_map()
    limit = workload.tuning.max_candidates if limit is None else limit

    valid_orders: list[tuple[str, ...]] = []
    pruned_orders: list[dict[str, Any]] = []

    def visit(prefix: tuple[str, ...]) -> None:
        if len(prefix) == len(table_names):
            valid_orders.append(prefix)
            return
        joined = set(prefix)
        for next_table in table_names:
            if next_table in joined:
                continue
            if prefix and next_table not in graph.adjacency[prefix[-1]] and not any(
                next_table in graph.adjacency[member] for member in joined
            ):
                pruned_orders.append(
                    {
                        "prefix": list(prefix),
                        "next_table": next_table,
                        "reason": "CARTESIAN_WITHOUT_EQUALITY",
                    }
                )
                continue
            visit(prefix + (next_table,))

    for first in table_names:
        visit((first,))

    all_specs: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for order in valid_orders:
        algorithm_count = max(0, len(order) - 1)
        for algorithms in itertools.product(ALGORITHMS, repeat=algorithm_count):
            all_specs.append((order, tuple(algorithms)))

    kept_specs = all_specs[:limit]
    omitted_specs = all_specs[limit:]
    plans = tuple(
        _make_plan(workload, graph, order, algorithms, origin="enumerated")
        for order, algorithms in kept_specs
    )
    omitted = tuple(
        {
            "candidate_id": _make_plan(
                workload, graph, order, algorithms, origin="enumerated"
            ).candidate_id,
            "table_order": list(order),
            "algorithms": list(algorithms),
            "reason": "CANDIDATE_LIMIT",
        }
        for order, algorithms in omitted_specs
    )
    baseline = _make_baseline(workload, graph)
    return CandidateEnumeration(
        graph=graph,
        plans=plans,
        pruned_orders=tuple(pruned_orders),
        omitted_candidates=omitted,
        total_before_limit=len(all_specs),
        applied_limit=limit,
        baseline=baseline,
    )


def _make_plan(
    workload: Workload,
    graph: JoinGraph,
    order: tuple[str, ...],
    algorithms: tuple[str, ...],
    *,
    origin: str,
    baseline_reason: str | None = None,
) -> Plan:
    steps: list[JoinStep] = []
    joined: set[str] = {order[0]} if order else set()
    for index, right_table in enumerate(order[1:]):
        predicates = tuple(
            edge
            for edge in graph.edges
            if _edge_connects_new(edge, right_table, joined)
        )
        if not predicates:
            raise LabError(
                "E_GRAPH",
                "候选顺序缺少等值谓词，拒绝执行笛卡尔积",
                table=right_table,
                field="plan.steps.predicates",
            )
        steps.append(JoinStep(right_table, algorithms[index], predicates))
        joined.add(right_table)
    canonical = {
        "query_digest": graph.query_digest,
        "table_order": list(order),
        "steps": [step.canonical() for step in steps],
    }
    signature_json = _canonical_json(canonical)
    candidate_id = "P-" + hashlib.sha256(signature_json.encode("utf-8")).hexdigest()[:20]
    signature = _signature(order, steps)
    return Plan(
        query_digest=graph.query_digest,
        table_order=order,
        steps=tuple(steps),
        candidate_id=candidate_id,
        signature=signature,
        origin=origin,
        baseline_reason=baseline_reason,
    )


def _make_baseline(workload: Workload, graph: JoinGraph) -> Plan:
    source_order = workload.query.source_order
    if _order_is_connected(source_order, graph):
        return _make_plan(
            workload,
            graph,
            source_order,
            tuple("NESTED_LOOP" for _ in source_order[1:]),
            origin="baseline",
            baseline_reason="SQL_FROM_ORDER_WITH_NESTED_LOOP",
        )
    # The query graph is connected, so at least one connected order exists.  Use
    # the lexicographically first one but keep the baseline clearly identified.
    fallback = _first_connected_order(workload.query.tables, graph)
    return _make_plan(
        workload,
        graph,
        fallback,
        tuple("NESTED_LOOP" for _ in fallback[1:]),
        origin="baseline",
        baseline_reason="SOURCE_ORDER_DISCONNECTED_FALLBACK",
    )


def _order_is_connected(order: tuple[str, ...], graph: JoinGraph) -> bool:
    joined: set[str] = set()
    for index, table in enumerate(order):
        if index and not any(table in graph.adjacency[member] for member in joined):
            return False
        joined.add(table)
    return True


def _first_connected_order(tables: Iterable[str], graph: JoinGraph) -> tuple[str, ...]:
    ordered = tuple(sorted(tables))

    def search(prefix: tuple[str, ...]) -> tuple[str, ...] | None:
        if len(prefix) == len(ordered):
            return prefix
        joined = set(prefix)
        for table in ordered:
            if table in joined:
                continue
            if prefix and not any(table in graph.adjacency[member] for member in joined):
                continue
            result = search(prefix + (table,))
            if result is not None:
                return result
        return None

    result = search(())
    if result is None:
        raise LabError("E_GRAPH", "连接图不存在无笛卡尔积的顺序", field="query.joins")
    return result


def _edge_connects_new(edge: JoinPredicate, new_table: str, joined: set[str]) -> bool:
    left, right = edge.left.table, edge.right.table
    return (left == new_table and right in joined) or (right == new_table and left in joined)


def _signature(order: tuple[str, ...], steps: tuple[JoinStep, ...] | list[JoinStep]) -> str:
    parts = [order[0]]
    for step in steps:
        parts.append(f"{step.algorithm}->{step.right_table}")
    return "[" + " ".join(parts) + "]"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_key(value: Any) -> str:
    return _canonical_json(value)

