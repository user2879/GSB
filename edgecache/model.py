"""Validated declaration data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    slots: int
    lease_limit: int
    seed: int
    quota_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "slots": self.slots,
            "lease_limit": self.lease_limit,
            "seed": self.seed,
            "quota_version": self.quota_version,
        }


@dataclass(frozen=True)
class ContentSpec:
    content_key: str
    total_bytes: int
    shard_count: int
    content_seed: int

    @property
    def shard_bytes(self) -> int:
        return self.total_bytes // self.shard_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_key": self.content_key,
            "total_bytes": self.total_bytes,
            "shard_count": self.shard_count,
            "content_seed": self.content_seed,
        }


@dataclass(frozen=True)
class Config:
    sample_scale: int
    lease_ttl_ms: int
    global_lease_limit: int
    global_byte_budget: int
    socket_timeout_ms: int = 3000
    handshake_timeout_ms: int = 4000
    max_workers: int = 8
    drain_timeout_ms: int = 5000

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_scale": self.sample_scale,
            "lease_ttl_ms": self.lease_ttl_ms,
            "global_lease_limit": self.global_lease_limit,
            "global_byte_budget": self.global_byte_budget,
            "socket_timeout_ms": self.socket_timeout_ms,
            "handshake_timeout_ms": self.handshake_timeout_ms,
            "max_workers": self.max_workers,
            "drain_timeout_ms": self.drain_timeout_ms,
        }


@dataclass(frozen=True)
class Event:
    seq: int
    type: str
    delay_ms: int = 0
    node_id: str | None = None
    content_key: str | None = None
    shards: tuple[int, ...] = field(default_factory=tuple)
    lease_id: str | None = None
    expected_version: int | None = None
    extend_ms: int | None = None
    quota_version: int | None = None
    node_ids: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "seq": self.seq,
            "type": self.type,
            "delay_ms": self.delay_ms,
        }
        if self.node_id is not None:
            result["node_id"] = self.node_id
        if self.content_key is not None:
            result["content_key"] = self.content_key
        if self.shards:
            result["shards"] = sorted(self.shards)
        if self.lease_id is not None:
            result["lease_id"] = self.lease_id
        if self.expected_version is not None:
            result["expected_version"] = self.expected_version
        if self.extend_ms is not None:
            result["extend_ms"] = self.extend_ms
        if self.quota_version is not None:
            result["quota_version"] = self.quota_version
        if self.node_ids:
            result["node_ids"] = sorted(self.node_ids)
        return result


@dataclass(frozen=True)
class Manifest:
    format_version: int
    config: Config
    nodes: tuple[NodeSpec, ...]
    contents: tuple[ContentSpec, ...]
    events: tuple[Event, ...]
    source_path: str | None = None

    @property
    def initial_quota_version(self) -> int:
        return max(node.quota_version for node in self.nodes)

    @property
    def node_map(self) -> dict[str, NodeSpec]:
        return {node.node_id: node for node in self.nodes}

    @property
    def content_map(self) -> dict[str, ContentSpec]:
        return {content.content_key: content for content in self.contents}

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "config": self.config.to_dict(),
            "nodes": [node.to_dict() for node in sorted(self.nodes, key=lambda item: item.node_id)],
            "contents": [content.to_dict() for content in sorted(self.contents, key=lambda item: item.content_key)],
            "events": [event.to_dict() for event in sorted(self.events, key=lambda item: item.seq)],
        }
