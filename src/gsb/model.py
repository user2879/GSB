"""Immutable, validated domain objects."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Iterable


@dataclass(frozen=True)
class NodeSpec:
    id: str
    host: str
    port: int
    seed: int
    concurrency: int
    lease_limit: int
    quota: int
    used: int
    fixed_scale: int

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "host": self.host,
            "port": self.port,
            "seed": self.seed,
            "concurrency": self.concurrency,
            "lease_limit": self.lease_limit,
            "quota": self.quota,
            "used": self.used,
            "fixed_scale": self.fixed_scale,
        }


@dataclass(frozen=True)
class ShardSpec:
    id: str
    node_id: str
    dependencies: tuple[str, ...]
    demand: int
    sample_count: int
    group_count: int
    batch_size: int

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "node_id": self.node_id,
            "dependencies": list(self.dependencies),
            "demand": self.demand,
            "sample_count": self.sample_count,
            "group_count": self.group_count,
            "batch_size": self.batch_size,
        }


@dataclass(frozen=True)
class Spec:
    format_version: int
    capacity_version: str
    fixed_scale: int
    global_budget: int
    max_workers: int
    lease_ttl_ms: int
    handshake_timeout_ms: int
    io_timeout_ms: int
    poll_interval_ms: int
    nodes: tuple[NodeSpec, ...]
    shards: tuple[ShardSpec, ...]

    def node_map(self) -> dict[str, NodeSpec]:
        return {node.id: node for node in self.nodes}

    def shard_map(self) -> dict[str, ShardSpec]:
        return {shard.id: shard for shard in self.shards}

    def dependents(self) -> dict[str, tuple[str, ...]]:
        result: dict[str, list[str]] = {shard.id: [] for shard in self.shards}
        for shard in self.shards:
            for dependency in shard.dependencies:
                result[dependency].append(shard.id)
        return {key: tuple(sorted(value)) for key, value in result.items()}

    def topological_order(self, shard_ids: Iterable[str] | None = None) -> tuple[str, ...]:
        """Kahn ordering with a min-heap of shard IDs.

        The ordering is stable across JSON member order, process scheduling and
        concurrent completion order.
        """

        selected = set(self.shard_map() if shard_ids is None else shard_ids)
        unknown = selected - set(self.shard_map())
        if unknown:
            raise ValueError(f"unknown shards in topological selection: {sorted(unknown)}")
        indegree = {shard_id: 0 for shard_id in selected}
        dependents = self.dependents()
        for shard in self.shards:
            if shard.id not in selected:
                continue
            for dependency in shard.dependencies:
                if dependency in selected:
                    indegree[shard.id] += 1
        ready = [shard_id for shard_id, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        order: list[str] = []
        while ready:
            shard_id = heapq.heappop(ready)
            order.append(shard_id)
            for child in dependents[shard_id]:
                if child not in selected:
                    continue
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
        if len(order) != len(selected):
            raise ValueError("dependency graph is not acyclic")
        return tuple(order)

    def topo_rank(self) -> dict[str, int]:
        return {shard_id: index for index, shard_id in enumerate(self.topological_order())}

    def as_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "capacity_version": self.capacity_version,
            "fixed_scale": self.fixed_scale,
            "global_budget": self.global_budget,
            "max_workers": self.max_workers,
            "lease_ttl_ms": self.lease_ttl_ms,
            "handshake_timeout_ms": self.handshake_timeout_ms,
            "io_timeout_ms": self.io_timeout_ms,
            "poll_interval_ms": self.poll_interval_ms,
            "nodes": [node.as_dict() for node in self.nodes],
            "shards": [shard.as_dict() for shard in self.shards],
        }

