from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CandidateRow:
    request_id: str
    channel: str
    item_id: str
    score: float
    features: tuple[float, ...]
    shard_seq: int
    coverage_start_ms: int
    coverage_end_ms: int
    line_no: int
    source: str
    injection_id: str | None = None

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "channel": self.channel,
            "item_id": self.item_id,
            "score": self.score,
            "features": list(self.features),
            "shard_seq": self.shard_seq,
            "coverage_start_ms": self.coverage_start_ms,
            "coverage_end_ms": self.coverage_end_ms,
        }


@dataclass
class Shard:
    shard_id: str
    channel: str
    shard_seq: int
    coverage_start_ms: int
    coverage_end_ms: int
    rows: list[CandidateRow]
    fingerprint: str
    source: str
    injection_id: str | None = None

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "channel": self.channel,
            "shard_seq": self.shard_seq,
            "coverage_start_ms": self.coverage_start_ms,
            "coverage_end_ms": self.coverage_end_ms,
            "rows": [row.semantic_dict() for row in self.rows],
        }


@dataclass
class ShardDescriptor:
    path: str
    injection_id: str | None = None
    origin: str = "base"
    channel: str | None = None
    shard_seq: int | None = None
    coverage_start_ms: int | None = None
    coverage_end_ms: int | None = None


@dataclass
class MaterializedInput:
    descriptors: list[ShardDescriptor]
    config_raw: dict[str, Any]
    applied_steps: list[int]
    operations: list[dict[str, Any]]
    effective_plan_hash: str


@dataclass
class ValidationFailure:
    code: str
    source: str
    reason: str
    shard_id: str | None = None
    line_no: int | None = None
    injection_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "source": self.source,
            "reason": self.reason,
            "shard_id": self.shard_id,
            "line_no": self.line_no,
            "injection_id": self.injection_id,
        }


@dataclass
class LoadResult:
    shards: list[Shard]
    failures: list[ValidationFailure]
    attributes: dict[str, dict[str, Any]]
    attributes_fingerprint: str
    attributes_failures: list[ValidationFailure] = field(default_factory=list)

    @property
    def complete_shard_set(self) -> list[dict[str, Any]]:
        return [
            {
                "shard_id": shard.shard_id,
                "channel": shard.channel,
                "coverage_start_ms": shard.coverage_start_ms,
                "coverage_end_ms": shard.coverage_end_ms,
                "fingerprint": shard.fingerprint,
                "source": shard.source,
            }
            for shard in self.shards
        ]
