"""Core immutable data models for declarations, requests and metadata."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

FORMAT_C1 = 1
FORMAT_C2 = 2
FORMAT_C3 = 3
FORMAT_NAMES = {FORMAT_C1: "c1", FORMAT_C2: "c2", FORMAT_C3: "c3"}

# Canonical field order used for byte-identical serialization.
CANONICAL_FIELDS = (
    "object_key",
    "shard_index",
    "shard_digest",
    "merge_id",
    "shard_bytes",
    "meta_version",
    "content_check",
    "parent_merge_id",
    "commit_cursor",
    "source_format",
    "merge_seq",
)


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    capacity_slots: int
    lease_limit: int
    seed: int
    meta_version: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ShardPlacement:
    shard_index: int
    node_id: str
    size: int


@dataclass(frozen=True)
class ObjectManifest:
    object_key: str
    total_bytes: int
    shard_size: int
    content_seed: int
    placements: tuple[ShardPlacement, ...]

    @property
    def shard_count(self) -> int:
        return len(self.placements)

    def node_of(self, shard_index: int) -> str:
        for placement in self.placements:
            if placement.shard_index == shard_index:
                return placement.node_id
        raise KeyError(shard_index)


@dataclass(frozen=True)
class Request:
    seq: int
    op: str  # grant | upload | renew | abort | read
    object_key: str
    shard_index: Optional[int] = None
    lease_id: Optional[str] = None
    expected_lease_version: Optional[int] = None
    ttl_ms: Optional[int] = None
    source_file: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class NormalizedRecord:
    """A metadata record normalized across c1/c2/c3 into one canonical shape."""

    object_key: str
    shard_index: int
    shard_digest: str
    merge_id: Optional[str]
    shard_bytes: Optional[int]
    meta_version: Optional[int]
    content_check: Optional[str]
    parent_merge_id: Optional[str]
    commit_cursor: Optional[int]
    source_format: int
    source_line: int
    filled_fields: frozenset[str] = frozenset()
    fill_basis: dict[str, str] = field(default_factory=dict)
    merge_seq: Optional[int] = None

    def canonical_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in CANONICAL_FIELDS:
            value = getattr(self, name)
            if isinstance(value, frozenset):
                value = sorted(value)
            out[name] = value
        return out

    def canonical_json(self) -> bytes:
        import json

        return json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")


@dataclass(frozen=True)
class Lease:
    lease_id: str
    object_key: str
    shard_index: int
    node_id: str
    granted_ms: int
    expires_ms: int
    renew_count: int
    lease_version: int
    state: str = "active"  # active | expired | aborted

    def is_active(self, now_ms: int) -> bool:
        return self.state == "active" and now_ms < self.expires_ms

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MergeCandidate:
    """A shard participating in an adoption decision for one object key."""

    object_key: str
    shard_index: int
    shard_digest: str
    content_check: Optional[str]
    merge_id: str
    parent_merge_id: Optional[str]
    shard_bytes: int
    meta_version: int
    source_format: int
    merge_seq: int
    origin: str  # ledger-upload | metadata-file:<name>
    node_id: Optional[str] = None
    lease_id: Optional[str] = None

    def canonical_tuple(self) -> tuple:
        # Stable winner ordering: index, meta_version desc, format desc, seq.
        return (
            self.shard_index,
            -int(self.meta_version),
            -int(self.source_format),
            int(self.merge_seq),
            self.merge_id,
        )


@dataclass(frozen=True)
class RejectedShard:
    candidate: MergeCandidate
    reason: str
    winner_merge_id: str


@dataclass(frozen=True)
class AdoptionDecision:
    object_key: str
    shard_index: int
    winner: MergeCandidate
    rejected: tuple[RejectedShard, ...]

    def canonical_bytes(self) -> bytes:
        import json

        payload = {
            "object_key": self.object_key,
            "shard_index": self.shard_index,
            "winner_merge_id": self.winner.merge_id,
            "winner_digest": self.winner.shard_digest,
            "rejected": [
                {
                    "merge_id": item.candidate.merge_id,
                    "reason": item.reason,
                    "winner_merge_id": item.winner_merge_id,
                }
                for item in self.rejected
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


__all__ = [
    "FORMAT_C1",
    "FORMAT_C2",
    "FORMAT_C3",
    "FORMAT_NAMES",
    "CANONICAL_FIELDS",
    "NodeSpec",
    "ShardPlacement",
    "ObjectManifest",
    "Request",
    "NormalizedRecord",
    "Lease",
    "MergeCandidate",
    "RejectedShard",
    "AdoptionDecision",
]
