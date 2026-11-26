from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    concurrency_quota: int
    lease_limit: int
    seed: int
    capacity_version: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ShardSpec:
    shard_id: str
    node_id: str
    dependencies: tuple[str, ...]
    sample_rows: int
    resource_units: int
    priority: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dependencies"] = list(self.dependencies)
        return data


@dataclass(frozen=True, slots=True)
class DelayBatch:
    batch_id: str
    node_id: str
    primary_revision: int
    replayed_revision: int
    delay_samples_ms: tuple[int, ...]

    @property
    def delay_ms(self) -> int:
        return max(self.delay_samples_ms)

    @property
    def jitter_ms(self) -> int:
        return max(self.delay_samples_ms) - min(self.delay_samples_ms)

    @property
    def revision_lag(self) -> int:
        return self.primary_revision - self.replayed_revision

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "node_id": self.node_id,
            "primary_revision": self.primary_revision,
            "replayed_revision": self.replayed_revision,
            "delay_samples_ms": list(self.delay_samples_ms),
        }


@dataclass(frozen=True, slots=True)
class Spec:
    format_version: int
    session_id: str
    scale_power: int
    global_resource_budget: int
    lease_ttl_ms: int
    max_revision_lag: int
    dispatch_delay_limit_ms: int
    revision_chain: tuple[int, ...]
    nodes: tuple[NodeSpec, ...]
    shards: tuple[ShardSpec, ...]
    delay_batches: tuple[DelayBatch, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "session_id": self.session_id,
            "scale_power": self.scale_power,
            "global_resource_budget": self.global_resource_budget,
            "lease_ttl_ms": self.lease_ttl_ms,
            "max_revision_lag": self.max_revision_lag,
            "dispatch_delay_limit_ms": self.dispatch_delay_limit_ms,
            "revision_chain": list(self.revision_chain),
            "nodes": [node.to_dict() for node in self.nodes],
            "shards": [shard.to_dict() for shard in self.shards],
            "delay_batches": [batch.to_dict() for batch in self.delay_batches],
        }

    def node(self, node_id: str) -> NodeSpec:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def shard(self, shard_id: str) -> ShardSpec:
        for shard in self.shards:
            if shard.shard_id == shard_id:
                return shard
        raise KeyError(shard_id)


@dataclass(slots=True)
class AggregateSummary:
    node_id: str
    shard_id: str
    sample_rows: int
    capacity_version: int
    participating_rows: int
    sample_digest: str
    groups: list[dict[str, Any]] = field(default_factory=list)
    format_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "node_id": self.node_id,
            "shard_id": self.shard_id,
            "sample_rows": self.sample_rows,
            "capacity_version": self.capacity_version,
            "participating_rows": self.participating_rows,
            "sample_digest": self.sample_digest,
            "groups": self.groups,
        }
