"""Deterministic bounded Dijkstra traversal with propagated error bounds.

Determinism contract
--------------------
* The heap key is an explicit, fully ordered tuple; no dict / set / hash
  iteration influences expansion.  Adjacency is pre-sorted in the loader.
* Equal-distance candidate paths are resolved by an explicit stable ranking:
  ``(distance fp, exact remainder bound micro, conservative ulp bound,
  lexicographic node path)``.  Error bounds therefore participate only after
  distance equality, and never change the shortest-distance result.
* The same graph + query yields byte-identical visit sequences regardless of
  the original insertion order of nodes and edges.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from .fixedpoint import (
    EdgeErrorTerm,
    Quantizer,
    build_path_terms,
    dominant_term,
    saturating_add,
)
from .models import Graph, Query

# Classification labels.
WITHIN = "within"                          # deterministically inside the cap
CONFIRMED_OVER = "confirmed_over"          # estimate itself exceeds the cap
ERROR_CROSS = "indeterminate_error_cross"  # d <= cap < d + error bound
UNREACHABLE = "unreachable"                # no in-cap path reached the target


@dataclass(frozen=True)
class PruneRecord:
    source: int
    target: int
    weight_micro: int
    dist_before_fp: int
    candidate_fp: int
    cap_fp: int
    reason: str


@dataclass
class TargetResult:
    target: int
    reached: bool
    path_nodes: tuple[int, ...] = ()
    dist_fp: int | None = None
    dist_micro: int | None = None
    exact_bound_micro: int | None = None
    conservative_bound_ulp: int | None = None
    conservative_bound_micro: int | None = None
    terms: list[EdgeErrorTerm] = field(default_factory=list)
    dominant_edge: tuple[int, int] | None = None  # (source, target)
    dominant_remainder_micro: int | None = None
    cap_micro: int = 0
    classification_exact: str = UNREACHABLE
    classification_conservative: str = UNREACHABLE
    classification: str = UNREACHABLE
    error_affected: bool = False

    @property
    def over_limit(self) -> bool:
        return self.classification in (CONFIRMED_OVER, ERROR_CROSS)


@dataclass
class TraversalResult:
    query_id: str
    source: int
    cap_fp: int
    visited_sequence: list[int] = field(default_factory=list)
    pruned: list[PruneRecord] = field(default_factory=list)
    targets: dict[int, TargetResult] = field(default_factory=dict)
    saturated_events: int = 0
    budget_visited: int = 0
    budget_exhausted: bool = False
    error_mode: str = "exact"


def _classify(
    dist_micro: int,
    cap_micro: int,
    exact_bound_micro: int,
    conservative_micro: int,
) -> tuple[str, str, bool]:
    """Return ``(exact label, conservative label, error-affected flag)``."""
    if dist_micro > cap_micro:
        return CONFIRMED_OVER, CONFIRMED_OVER, False

    exact_label = (
        WITHIN if dist_micro + exact_bound_micro <= cap_micro else ERROR_CROSS
    )
    conservative_label = (
        WITHIN if dist_micro + conservative_micro <= cap_micro else ERROR_CROSS
    )
    affected = ERROR_CROSS in (exact_label, conservative_label)
    return exact_label, conservative_label, affected


def run_traversal(
    graph: Graph,
    query: Query,
    quantizer: Quantizer,
    budget,
    error_mode: str = "exact",
) -> TraversalResult:
    """Run one bounded, budgeted traversal for ``query``.

    ``budget`` only needs ``acquire() -> bool``; ``False`` means the session
    traversal budget is exhausted and the query must stop immediately.
    """
    if error_mode not in ("exact", "conservative"):
        raise ValueError("error_mode must be 'exact' or 'conservative'")

    cap_fp = quantizer.quantize(query.cap_micro)
    result = TraversalResult(
        query_id=query.query_id,
        source=query.source,
        cap_fp=cap_fp,
        error_mode=error_mode,
    )
    remaining_targets = set(query.targets)

    # best[node] -> ranking key of the currently best candidate
    best: dict[int, tuple] = {}
    heap: list[tuple] = []
    counter = 0
    start_path = (query.source,)
    start_key = (0, 0, 0, start_path, counter, start_path, ())
    best[query.source] = (0, 0, 0, start_path)
    heapq.heappush(heap, start_key)

    def ranking(dist_fp: int, exact_micro: int, cons_ulp: int, path_nodes: tuple) -> tuple:
        return (dist_fp, exact_micro, cons_ulp, path_nodes)

    while heap:
        dist_fp, exact_micro, cons_ulp, path_nodes, seq, _, path_edges = heapq.heappop(heap)
        node = path_nodes[-1]
        if best.get(node) != ranking(dist_fp, exact_micro, cons_ulp, path_nodes):
            continue  # stale heap entry, a strictly better candidate exists
        if not budget.acquire():
            result.budget_exhausted = True
            break

        result.visited_sequence.append(node)
        result.budget_visited += 1
        if node in remaining_targets:
            remaining_targets.remove(node)
            terms = build_path_terms(list(path_edges), quantizer)
            dom = dominant_term(terms)
            dist_micro = quantizer.fp_to_micro(dist_fp)
            cons_micro = cons_ulp * quantizer.micro_per_ulp
            exact_label, cons_label, affected = _classify(
                dist_micro, query.cap_micro, exact_micro, cons_micro
            )
            result.targets[node] = TargetResult(
                target=node,
                reached=True,
                path_nodes=tuple(path_nodes),
                dist_fp=dist_fp,
                dist_micro=dist_micro,
                exact_bound_micro=exact_micro,
                conservative_bound_ulp=cons_ulp,
                conservative_bound_micro=cons_micro,
                terms=terms,
                dominant_edge=(dom.source, dom.target) if dom else None,
                dominant_remainder_micro=dom.edge_remainder_micro if dom else None,
                cap_micro=query.cap_micro,
                classification_exact=exact_label,
                classification_conservative=cons_label,
                classification=exact_label if error_mode == "exact" else cons_label,
                error_affected=affected,
            )

        if not remaining_targets:
            break  # all requested targets settled; no further expansion needed

        for edge in graph.outgoing(node):
            weight_fp = quantizer.quantize(edge.weight_micro)
            candidate_fp, saturated = saturating_add(dist_fp, weight_fp)
            if saturated:
                result.saturated_events += 1
                result.pruned.append(
                    PruneRecord(
                        edge.source, edge.target, edge.weight_micro,
                        dist_fp, candidate_fp, cap_fp, "saturated",
                    )
                )
                continue
            if candidate_fp > cap_fp:
                result.pruned.append(
                    PruneRecord(
                        edge.source, edge.target, edge.weight_micro,
                        dist_fp, candidate_fp, cap_fp, "cap_exceeded",
                    )
                )
                continue

            mpu = quantizer.micro_per_ulp
            edge_remainder = edge.weight_micro - weight_fp * mpu
            candidate_exact = exact_micro + edge_remainder
            candidate_cons = cons_ulp + 1
            candidate_path_nodes = path_nodes + (edge.target,)
            candidate_edges = path_edges + ((edge.source, edge.target, edge.weight_micro),)
            key = ranking(
                candidate_fp, candidate_exact, candidate_cons, candidate_path_nodes
            )
            if edge.target not in best or key < best[edge.target]:
                best[edge.target] = key
                counter += 1
                heapq.heappush(
                    heap,
                    (
                        candidate_fp,
                        candidate_exact,
                        candidate_cons,
                        candidate_path_nodes,
                        counter,
                        candidate_path_nodes,
                        candidate_edges,
                    ),
                )

    # Every target never settled inside the cap is an explicit result row.
    for target in query.targets:
        if target not in result.targets:
            result.targets[target] = TargetResult(
                target=target, reached=False, cap_micro=query.cap_micro
            )
    return result
