"""Input and protocol data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import canonical_bytes, sha256_bytes

EVENT_TYPES = ("apply", "renew", "release")
EVENT_TYPE_ALIASES = {
    "apply": "apply",
    "request": "apply",
    "grant": "apply",
    "申请": "apply",
    "renew": "renew",
    "heartbeat": "renew",
    "续约": "renew",
    "release": "release",
    "主动释放": "release",
    "释放": "release",
}


@dataclass(frozen=True)
class ClusterSpec:
    cluster_id: str
    host: str
    port: int
    lease_duration_ms: int
    renewal_interval_ms: int
    capacity: int

    @classmethod
    def from_json(cls, raw: Any) -> "ClusterSpec":
        return cls(
            cluster_id=raw["cluster_id"],
            host=raw["host"],
            port=raw["port"],
            lease_duration_ms=raw["lease_duration_ms"],
            renewal_interval_ms=raw["renewal_interval_ms"],
            capacity=raw["capacity"],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "cluster_id": self.cluster_id,
            "host": self.host,
            "lease_duration_ms": self.lease_duration_ms,
            "port": self.port,
            "renewal_interval_ms": self.renewal_interval_ms,
        }


@dataclass(frozen=True)
class HolderSpec:
    holder_id: str
    priority: int
    targets: tuple[str, ...]

    @classmethod
    def from_json(cls, raw: Any) -> "HolderSpec":
        return cls(
            holder_id=raw["holder_id"],
            priority=raw["priority"],
            targets=tuple(raw["targets"]),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "holder_id": self.holder_id,
            "priority": self.priority,
            "targets": list(self.targets),
        }


@dataclass(frozen=True)
class LeaseEvent:
    event_seq: int
    timestamp_ms: int
    cluster_id: str
    holder_id: str
    event_type: str
    line_no: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "event_seq": self.event_seq,
            "event_type": self.event_type,
            "holder_id": self.holder_id,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass
class SpecBundle:
    clusters: tuple[ClusterSpec, ...]
    holders: tuple[HolderSpec, ...]
    events: tuple[LeaseEvent, ...]
    clusters_path: str
    holders_path: str
    events_path: str

    @property
    def cluster_map(self) -> dict[str, ClusterSpec]:
        return {cluster.cluster_id: cluster for cluster in self.clusters}

    @property
    def holder_map(self) -> dict[str, HolderSpec]:
        return {holder.holder_id: holder for holder in self.holders}

    def priorities_for(self, cluster_id: str) -> dict[str, int]:
        return {
            holder.holder_id: holder.priority
            for holder in self.holders
            if cluster_id in holder.targets
        }

    def events_for(self, cluster_id: str, max_seq: int | None = None) -> list[LeaseEvent]:
        return [
            event
            for event in self.events
            if event.cluster_id == cluster_id
            and (max_seq is None or event.event_seq <= max_seq)
        ]

    def event_stream_digest(self) -> str:
        payload = b"".join(canonical_bytes(event.to_json()) + b"\n" for event in self.events)
        return sha256_bytes(payload)

    def priority_digest(self, cluster_id: str) -> str:
        items = sorted(self.priorities_for(cluster_id).items())
        return sha256_bytes(canonical_bytes(items))

    def holder_ids(self) -> Iterable[str]:
        for holder in self.holders:
            yield holder.holder_id
