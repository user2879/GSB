"""Validated immutable configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import sha256_hex


@dataclass(frozen=True)
class LayerSpec:
    layer_id: str
    input_dim: int
    output_dim: int
    scale: int
    depends_on: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "scale": self.scale,
            "depends_on": list(self.depends_on),
        }


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    host: str
    port: int
    lease_limit: int
    concurrency_limit: int
    seed: int
    parameter_version: int
    sample_dim: int
    lease_ttl_ms: int
    layers: tuple[LayerSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "host": self.host,
            "port": self.port,
            "lease_limit": self.lease_limit,
            "concurrency_limit": self.concurrency_limit,
            "seed": self.seed,
            "parameter_version": self.parameter_version,
            "sample_dim": self.sample_dim,
            "lease_ttl_ms": self.lease_ttl_ms,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @property
    def parameters_digest(self) -> str:
        return sha256_hex(
            {
                "shard_id": self.shard_id,
                "seed": self.seed,
                "sample_dim": self.sample_dim,
                "parameter_version": self.parameter_version,
                "layers": [layer.to_dict() for layer in self.layers],
            }
        )


@dataclass(frozen=True)
class CheckpointSpec:
    checkpoint_id: str
    shard_id: str
    parameter_version: int
    loss_baseline_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "shard_id": self.shard_id,
            "parameter_version": self.parameter_version,
            "loss_baseline_digest": self.loss_baseline_digest,
        }


@dataclass(frozen=True)
class BatchSpec:
    batch_id: str
    shard_id: str
    sample_ids: tuple[int, ...]
    resource_units: int
    order: int
    waterline: int = 0
    checkpoint_id: str | None = None
    depends_on_batches: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "batch_id": self.batch_id,
            "shard_id": self.shard_id,
            "sample_ids": list(self.sample_ids),
            "resource_units": self.resource_units,
            "order": self.order,
            "waterline": self.waterline,
            "depends_on_batches": list(self.depends_on_batches),
        }
        if self.checkpoint_id is not None:
            payload["checkpoint_id"] = self.checkpoint_id
        return payload


@dataclass(frozen=True)
class EventSpec:
    seq: int
    type: str
    shard_id: str | None = None
    checkpoint_id: str | None = None
    lease_id: str | None = None
    lease_version: int | None = None
    ttl_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"seq": self.seq, "type": self.type}
        for key in ("shard_id", "checkpoint_id", "lease_id", "lease_version", "ttl_ms"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class SessionSpec:
    format_version: int
    global_budget: int
    shards: tuple[ShardSpec, ...]
    checkpoints: tuple[CheckpointSpec, ...]
    batches: tuple[BatchSpec, ...]
    events: tuple[EventSpec, ...]
    source: str = "<memory>"

    def shard(self, shard_id: str) -> ShardSpec:
        for shard in self.shards:
            if shard.shard_id == shard_id:
                return shard
        raise KeyError(shard_id)

    def checkpoint(self, checkpoint_id: str) -> CheckpointSpec:
        for checkpoint in self.checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        raise KeyError(checkpoint_id)

    def batch(self, batch_id: str) -> BatchSpec:
        for batch in self.batches:
            if batch.batch_id == batch_id:
                return batch
        raise KeyError(batch_id)

    @property
    def shard_set_version(self) -> str:
        return sha256_hex(
            {
                "global_budget": self.global_budget,
                "shards": [shard.to_dict() for shard in self.shards],
            }
        )

    @property
    def parameter_versions(self) -> dict[str, int]:
        return {shard.shard_id: shard.parameter_version for shard in self.shards}

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "shard_set_version": self.shard_set_version,
            "parameter_versions": self.parameter_versions,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "global_budget": self.global_budget,
            "shards": [shard.to_dict() for shard in self.shards],
            "checkpoints": [cp.to_dict() for cp in self.checkpoints],
            "batches": [batch.to_dict() for batch in self.batches],
            "events": [event.to_dict() for event in self.events],
        }
