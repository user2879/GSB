from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from .canonical import digest_json
from .graph import JoinGraph, build_join_graph, graph_for_workload
from .schema import (
    Aggregate,
    FilterPredicate,
    JoinPredicate,
    Projection,
    QueryDecl,
    Workload,
)

ALGORITHMS = ("hash", "nested_loop", "sort_merge")


@dataclass(frozen=True)
class PlanOperator:
    kind: str
    table: str | None = None
    algorithm: str | None = None
    left_tables: tuple[str, ...] | None = None
    predicates: tuple[JoinPredicate, ...] = ()
    filters: tuple[FilterPredicate, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        if self.kind in ("scan", "join_right"):
            result: dict[str, Any] = {"kind": self.kind, "table": self.table}
            if self.algorithm is not None:
                result["algorithm"] = self.algorithm
            if self.left_tables is not None:
                result["left_tables"] = list(self.left_tables)
            if self.predicates:
                result["predicates"] = [predicate.as_dict() for predicate in self.predicates]
            if self.filters:
                result["filters"] = [predicate.as_dict() for predicate in self.filters]
            return result
        raise ValueError(f"unsupported operator kind {self.kind}")


@dataclass(frozen=True)
class Plan:
    plan_id: str
    operator_digest: str
    query_id: str
    tables: tuple[str, ...]
    operators: tuple[PlanOperator, ...]
    projections: tuple[Projection, ...]
    group_by: tuple[Any, ...]
    aggregates: tuple[Aggregate, ...]

    def identity_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "tables": list(self.tables),
            "operators": [operator.as_dict() for operator in self.operators],
            "projections": [projection.as_dict() for projection in self.projections],
            "group_by": [ref.qualified for ref in self.group_by],
            "aggregates": [aggregate.as_dict() for aggregate in self.aggregates],
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "operator_digest": self.operator_digest,
            **self.identity_dict(),
        }

    def execution_summary(self, *, plan_version: int | None = None, cost_vector: list[int] | None = None) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "plan_id": self.plan_id,
            "operator_digest": self.operator_digest,
            "query_id": self.query_id,
            "tables": list(self.tables),
            "operators": [operator.as_dict() for operator in self.operators],
            "projections": [projection.as_dict() for projection in self.projections],
            "group_by": [ref.qualified for ref in self.group_by],
            "aggregates": [aggregate.as_dict() for aggregate in self.aggregates],
        }
        if plan_version is not None:
            summary["plan_version"] = plan_version
        if cost_vector is not None:
            summary["cost_vector"] = list(cost_vector)
        return summary


@dataclass(frozen=True)
class CandidateSet:
    query_id: str
    graph: JoinGraph
    plans: tuple[Plan, ...]

    @property
    def digest(self) -> str:
        return digest_json([plan.as_dict() for plan in self.plans])

    def canonical_bytes(self) -> bytes:
        from .canonical import canonical_bytes

        return canonical_bytes([plan.as_dict() for plan in self.plans])


def enumerate_candidates(workload: Workload) -> CandidateSet:
    query = workload.query
    graph = graph_for_workload(workload)
    tables = query.tables
    sequences = _connected_sequences(tables, graph)
    plans: list[Plan] = []
    for sequence in sequences:
        algorithm_count = len(sequence) - 1
        for algorithms in itertools.product(ALGORITHMS, repeat=algorithm_count):
            plans.append(_make_plan(query, sequence, algorithms, graph))
    plans.sort(key=lambda plan: (plan.plan_id, plan.operator_digest))
    if not plans:
        raise ValueError("query produced no connected join plan")
    return CandidateSet(query_id=query.id, graph=graph, plans=tuple(plans))


def _connected_sequences(tables: tuple[str, ...], graph: JoinGraph) -> tuple[tuple[str, ...], ...]:
    sequences: list[tuple[str, ...]] = []

    def visit(prefix: tuple[str, ...]) -> None:
        if len(prefix) == len(tables):
            sequences.append(prefix)
            return
        remaining = [table for table in tables if table not in prefix]
        for table in remaining:
            if graph.equality_link(prefix, table):
                visit(prefix + (table,))

    for first in tables:
        visit((first,))
    sequences.sort()
    return tuple(sequences)


def _make_plan(
    query: QueryDecl,
    sequence: tuple[str, ...],
    algorithms: tuple[str, ...],
    graph: JoinGraph,
) -> Plan:
    filters_by_table = {
        table: tuple(
            sorted(
                (item for item in query.filters if item.column.block_group == table),
                key=lambda item: (
                    item.column.column,
                    item.operator,
                    item.value if isinstance(item.value, tuple) else (item.value,),
                ),
            )
        )
        for table in sequence
    }

    operators: list[PlanOperator] = []
    prefix: tuple[str, ...] = ()
    for step, table in enumerate(sequence):
        operators.append(
            PlanOperator(
                kind="scan",
                table=table,
                filters=filters_by_table[table],
            )
        )
        if step > 0:
            predicates = tuple(
                sorted(
                    graph.predicates_for_step(prefix, table),
                    key=lambda item: (
                        item.left.qualified,
                        item.right.qualified,
                        item.operator,
                    ),
                )
            )
            operators.append(
                PlanOperator(
                    kind="join_right",
                    table=table,
                    algorithm=algorithms[step - 1],
                    left_tables=prefix,
                    predicates=predicates,
                )
            )
        prefix = prefix + (table,)

    identity = {
        "query_fingerprint": query.fingerprint,
        "tables": list(sequence),
        "operators": [operator.as_dict() for operator in operators],
        "projections": [projection.as_dict() for projection in query.projections],
        "group_by": [ref.qualified for ref in query.group_by],
        "aggregates": [aggregate.as_dict() for aggregate in query.aggregates],
    }
    operator_digest = digest_json([operator.as_dict() for operator in operators])
    plan_id = "plan-" + digest_json(identity)
    return Plan(
        plan_id=plan_id,
        operator_digest=operator_digest,
        query_id=query.id,
        tables=sequence,
        operators=tuple(operators),
        projections=query.projections,
        group_by=query.group_by,
        aggregates=query.aggregates,
    )


def build_plan_for_sequence(
    workload: Workload,
    sequence: tuple[str, ...],
    algorithms: tuple[str, ...],
) -> Plan:
    graph = build_join_graph(workload.query)
    if any(not graph.equality_link(sequence[:index], sequence[index]) for index in range(1, len(sequence))):
        raise ValueError("sequence contains a Cartesian step")
    for index in range(1, len(sequence)):
        if not graph.equality_link(sequence[:index], sequence[index]):
            raise ValueError("sequence contains a Cartesian step")
    if len(algorithms) != max(0, len(sequence) - 1):
        raise ValueError("algorithm count must equal join count")
    return _make_plan(workload.query, sequence, algorithms, graph)
