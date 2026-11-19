"""Deterministic incremental invalidation over the logical shard DAG."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Iterable

from .models import RelationDecl, Spec


@dataclass(frozen=True)
class TracePath:
    seed: str
    path: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "path": list(self.path)}


@dataclass(frozen=True)
class AffectedShard:
    logical_id: str
    seq: int
    depth: int
    direct: bool
    cause: tuple[TracePath, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_id": self.logical_id,
            "seq": self.seq,
            "depth": self.depth,
            "direct": self.direct,
            "trace": [item.as_dict() for item in self.cause],
        }


@dataclass(frozen=True)
class PropagationResult:
    seeds: tuple[str, ...]
    affected: tuple[str, ...]
    entries: tuple[AffectedShard, ...]
    affected_blocks: tuple[str, ...]
    graph_version: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "graph_version": self.graph_version,
            "seeds": list(self.seeds),
            "ordered_affected": list(self.affected),
            "affected_blocks": list(self.affected_blocks),
            "trace": [entry.as_dict() for entry in self.entries],
        }


def _shortest_paths(
    seed: str,
    adjacency: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    """Lexicographically smallest shortest path for every reachable node."""

    distances: dict[str, int] = {seed: 0}
    paths: dict[str, tuple[str, ...]] = {seed: (seed,)}
    heap: list[tuple[int, tuple[str, ...], str]] = [(0, (seed,), seed)]
    while heap:
        distance, path, node = heapq.heappop(heap)
        if distance != distances.get(node) or path != paths.get(node):
            continue
        for target in adjacency[node]:
            candidate_distance = distance + 1
            candidate_path = path + (target,)
            known_distance = distances.get(target)
            if known_distance is None or candidate_distance < known_distance:
                distances[target] = candidate_distance
                paths[target] = candidate_path
                heapq.heappush(heap, (candidate_distance, candidate_path, target))
            elif candidate_distance == known_distance and candidate_path < paths[target]:
                paths[target] = candidate_path
                heapq.heappush(heap, (candidate_distance, candidate_path, target))
    return paths


def propagate_invalidation(
    spec: Spec,
    seed_logical_ids: Iterable[str],
    *,
    relations: Iterable[RelationDecl] | None = None,
) -> PropagationResult:
    seeds = tuple(sorted(set(seed_logical_ids)))
    if not seeds:
        return PropagationResult((), (), (), (), spec.mirror_graph_version)
    unknown = [seed for seed in seeds if seed not in spec.logical_shards]
    if unknown:
        raise KeyError(f"unknown logical shard(s): {', '.join(unknown)}")

    adjacency = spec.adjacency
    if relations is not None:
        from .spec import stable_topological_order

        order, adjacency, _predecessors = stable_topological_order(spec.logical_shards, relations)
        topo_order = order
    else:
        topo_order = spec.topo_order

    paths_by_seed = {seed: _shortest_paths(seed, adjacency) for seed in seeds}
    affected_set: set[str] = set()
    causes: dict[str, list[TracePath]] = {}
    for seed in seeds:
        for target, path in paths_by_seed[seed].items():
            affected_set.add(target)
            causes.setdefault(target, []).append(TracePath(seed=seed, path=path))
    for values in causes.values():
        values.sort(key=lambda item: (item.seed, len(item.path), item.path))

    entries: list[AffectedShard] = []
    for logical_id in topo_order:
        if logical_id not in affected_set:
            continue
        cause = tuple(causes[logical_id])
        minimum_distance = min(len(item.path) - 1 for item in cause)
        direct = logical_id in seeds or minimum_distance == 1
        logical = spec.logical_shards[logical_id]
        entries.append(
            AffectedShard(
                logical_id=logical_id,
                seq=logical.seq,
                depth=minimum_distance,
                direct=direct,
                cause=cause,
            )
        )
    ordered_affected = tuple(entry.logical_id for entry in entries)

    affected_blocks = tuple(
        block.block_id
        for block in spec.blocks
        if any(logical_id in affected_set for logical_id in spec.logicals_for_block(block))
    )
    return PropagationResult(
        seeds=seeds,
        affected=ordered_affected,
        entries=tuple(entries),
        affected_blocks=affected_blocks,
        graph_version=spec.mirror_graph_version,
    )

