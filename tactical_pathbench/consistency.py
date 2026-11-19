"""增量结果 vs 全量重建的双口径一致性核对。

判定必须同时满足：
1. 路径代价差在容差内（整数代价下容差默认 0）；
2. 完整路线序列逐格相同。
禁止仅凭代价相等即判定一致。出现分歧时定位首个分歧查询、分歧格与
相关依赖边。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

from .astar import Route
from .errors import ClosureMismatchError
from .grid import Cell

DEFAULT_COST_TOLERANCE = 0


@dataclass
class QueryDivergence:
    query_id: str
    first_divergent_index: Optional[int]
    incremental_cell: Optional[List[int]]
    rebuilt_cell: Optional[List[int]]
    incremental_cost: int
    rebuilt_cost: int
    cost_delta: int
    dependency_edges: List[str]
    reason: str

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConsistencyReport:
    consistent: bool
    compared: int
    cost_tolerance: int
    divergences: List[QueryDivergence] = field(default_factory=list)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "consistent": self.consistent,
            "compared": self.compared,
            "cost_tolerance": self.cost_tolerance,
            "divergences": [d.to_payload() for d in self.divergences],
        }

    def raise_if_inconsistent(self) -> None:
        if not self.consistent:
            first = self.divergences[0]
            raise ClosureMismatchError(
                "增量与全量重建结果不一致",
                query_id=first.query_id,
                reason=first.reason,
                divergent_index=first.first_divergent_index,
                dependency_edges=first.dependency_edges,
            )


def compare_routes(
    incremental: Route,
    rebuilt: Route,
    query_id: str,
    dependency_edges: Sequence[str],
    cost_tolerance: int = DEFAULT_COST_TOLERANCE,
) -> Optional[QueryDivergence]:
    cost_delta = abs(incremental.cost - rebuilt.cost)
    first_index: Optional[int] = None
    inc_cell: Optional[Cell] = None
    reb_cell: Optional[Cell] = None
    reason_parts: List[str] = []

    if cost_delta > cost_tolerance:
        reason_parts.append("cost_mismatch")

    inc_cells = list(incremental.cells)
    reb_cells = list(rebuilt.cells)
    if len(inc_cells) != len(reb_cells):
        reason_parts.append("length_mismatch")
    for index in range(max(len(inc_cells), len(reb_cells))):
        inc = inc_cells[index] if index < len(inc_cells) else None
        reb = reb_cells[index] if index < len(reb_cells) else None
        if inc != reb:
            first_index = index
            inc_cell = inc
            reb_cell = reb
            reason_parts.append("cell_sequence_mismatch")
            break

    if not reason_parts:
        return None
    return QueryDivergence(
        query_id=query_id,
        first_divergent_index=first_index,
        incremental_cell=list(inc_cell) if inc_cell is not None else None,
        rebuilt_cell=list(reb_cell) if reb_cell is not None else None,
        incremental_cost=incremental.cost,
        rebuilt_cost=rebuilt.cost,
        cost_delta=cost_delta,
        dependency_edges=list(dependency_edges),
        reason="+".join(reason_parts),
    )


def compare_sets(
    incremental_routes: Dict[str, Route],
    rebuilt_routes: Dict[str, Route],
    edges_by_query: Dict[str, Sequence[str]],
    cost_tolerance: int = DEFAULT_COST_TOLERANCE,
) -> ConsistencyReport:
    query_ids = sorted(set(incremental_routes) & set(rebuilt_routes))
    divergences: List[QueryDivergence] = []
    for query_id in query_ids:
        divergence = compare_routes(
            incremental_routes[query_id],
            rebuilt_routes[query_id],
            query_id,
            edges_by_query.get(query_id, []),
            cost_tolerance,
        )
        if divergence is not None:
            divergences.append(divergence)
    return ConsistencyReport(
        consistent=not divergences,
        compared=len(query_ids),
        cost_tolerance=cost_tolerance,
        divergences=divergences,
    )
