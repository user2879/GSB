from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import JoinPredicate, QueryDecl, Workload


@dataclass(frozen=True)
class JoinEdge:
    left_table: str
    right_table: str
    predicate: JoinPredicate

    def as_dict(self) -> dict[str, Any]:
        return {
            "left_table": self.left_table,
            "right_table": self.right_table,
            "predicate": self.predicate.as_dict(),
        }


@dataclass(frozen=True)
class JoinGraph:
    tables: tuple[str, ...]
    equality_edges: tuple[JoinEdge, ...]
    residual_edges: tuple[JoinEdge, ...]

    def equality_link(self, prefix: tuple[str, ...] | list[str] | set[str], table: str) -> tuple[JoinEdge, ...]:
        prefix_set = set(prefix)
        return tuple(
            edge
            for edge in self.equality_edges
            if (edge.left_table == table and edge.right_table in prefix_set)
            or (edge.right_table == table and edge.left_table in prefix_set)
        )

    def predicates_for_step(self, prefix: tuple[str, ...] | list[str], table: str) -> tuple[JoinPredicate, ...]:
        prefix_set = set(prefix)
        return tuple(
            edge.predicate
            for edge in self.equality_edges + self.residual_edges
            if (edge.left_table == table and edge.right_table in prefix_set)
            or (edge.right_table == table and edge.left_table in prefix_set)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "tables": list(self.tables),
            "equality_edges": [edge.as_dict() for edge in self.equality_edges],
            "residual_edges": [edge.as_dict() for edge in self.residual_edges],
        }

    @property
    def digest(self) -> str:
        from .canonical import digest_json

        return digest_json(self.as_dict())


def build_join_graph(query: QueryDecl) -> JoinGraph:
    equality: list[JoinEdge] = []
    residual: list[JoinEdge] = []
    for predicate in query.joins:
        edge = JoinEdge(predicate.left.block_group, predicate.right.block_group, predicate)
        if predicate.is_equality:
            equality.append(edge)
        else:
            residual.append(edge)
    equality.sort(key=lambda edge: (edge.left_table, edge.right_table, edge.predicate.operator, edge.predicate.left.qualified, edge.predicate.right.qualified))
    residual.sort(key=lambda edge: (edge.left_table, edge.right_table, edge.predicate.operator, edge.predicate.left.qualified, edge.predicate.right.qualified))
    return JoinGraph(tables=query.tables, equality_edges=tuple(equality), residual_edges=tuple(residual))


def graph_for_workload(workload: Workload) -> JoinGraph:
    return build_join_graph(workload.query)

