from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .utils import canonical_json, sha256_text


LIMIT_DIMENSION_BYTES = "bytes"
LIMIT_DIMENSION_OPERATIONS = "operations"
LIMIT_DIMENSIONS = (LIMIT_DIMENSION_BYTES, LIMIT_DIMENSION_OPERATIONS)

DECISION_ALLOW = "ALLOW"
DECISION_RATE_LIMITED = "RATE_LIMITED"
DECISION_OVER_QUOTA = "OVER_QUOTA"
DECISIONS = (DECISION_ALLOW, DECISION_RATE_LIMITED, DECISION_OVER_QUOTA)

TOKEN_SCALE = 1000
ERROR_BOUND_ZERO = 0


@dataclass(frozen=True)
class Derivation:
    upstream_id: str
    factor: int
    offset: int

    def to_json(self) -> dict[str, Any]:
        return {
            "upstream_id": self.upstream_id,
            "factor": self.factor,
            "offset": self.offset,
        }


@dataclass(frozen=True)
class VolumeConfig:
    volume_id: str
    capacity_bytes: int
    ops_per_sec: int = 0
    burst_capacity: int = 0
    limit_dimension: str = LIMIT_DIMENSION_BYTES
    derivation: Derivation | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.volume_id,
            "capacity_bytes": self.capacity_bytes,
            "ops_per_sec": self.ops_per_sec,
            "burst_capacity": self.burst_capacity,
            "limit_dimension": self.limit_dimension,
            "derivation": self.derivation.to_json() if self.derivation else None,
        }


@dataclass
class ConfigDocument:
    revision: int
    volumes: tuple[VolumeConfig, ...]
    schema_version: int = 3
    original_version: int = 3
    name: str = "storage-config"
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def volume_map(self) -> dict[str, VolumeConfig]:
        return {volume.volume_id: volume for volume in self.volumes}

    @property
    def fingerprint(self) -> str:
        payload = {
            "revision": self.revision,
            "volumes": [volume.to_json() for volume in self.volumes],
        }
        return sha256_text(canonical_json(payload))

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "format_version": 3,
            "revision": self.revision,
            "name": self.name,
            "volumes": [volume.to_json() for volume in self.volumes],
        }
        if self.original_version != 3:
            result["upgraded_from"] = self.original_version
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def with_volume(self, updated: VolumeConfig, revision: int | None = None) -> "ConfigDocument":
        volumes: list[VolumeConfig] = []
        found = False
        for volume in self.volumes:
            if volume.volume_id == updated.volume_id:
                volumes.append(updated)
                found = True
            else:
                volumes.append(volume)
        if not found:
            raise KeyError(updated.volume_id)
        return ConfigDocument(
            revision=self.revision if revision is None else revision,
            volumes=tuple(volumes),
            schema_version=3,
            original_version=self.original_version,
            name=self.name,
            source_path=self.source_path,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class EffectiveVolume:
    volume_id: str
    capacity_bytes: int
    explicit_capacity_bytes: int
    derived_capacity_bytes: int | None
    capacity_basis: str
    arbitration: str
    derivation_chain: tuple[str, ...]
    ops_per_sec: int
    burst_capacity: int
    limit_dimension: str

    @property
    def rate_limit_enabled(self) -> bool:
        return self.ops_per_sec > 0 or self.burst_capacity > 0

    @property
    def token_capacity_scaled(self) -> int:
        return self.burst_capacity * TOKEN_SCALE

    def to_json(self) -> dict[str, Any]:
        return {
            "volume_id": self.volume_id,
            "capacity_bytes": self.capacity_bytes,
            "explicit_capacity_bytes": self.explicit_capacity_bytes,
            "derived_capacity_bytes": self.derived_capacity_bytes,
            "capacity_basis": self.capacity_basis,
            "arbitration": self.arbitration,
            "derivation_chain": list(self.derivation_chain),
            "ops_per_sec": self.ops_per_sec,
            "burst_capacity": self.burst_capacity,
            "limit_dimension": self.limit_dimension,
            "rate_limit_enabled": self.rate_limit_enabled,
        }


@dataclass(frozen=True)
class AccessEvent:
    seq: int
    timestamp_ms: int
    volume_id: str
    operation_count: int
    write_bytes: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        result = {
            "seq": self.seq,
            "timestamp_ms": self.timestamp_ms,
            "volume_id": self.volume_id,
            "operation_count": self.operation_count,
            "write_bytes": self.write_bytes,
        }
        result.update(self.extra)
        return result


@dataclass
class VolumeUsage:
    volume_id: str
    cumulative_write_bytes: int = 0
    tokens_scaled: int = 0
    last_refill_ms: int | None = None
    allowed_events: int = 0
    rate_limited_events: int = 0
    over_quota_events: int = 0
    allowed_write_bytes: int = 0
    allowed_operations: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "volume_id": self.volume_id,
            "cumulative_write_bytes": self.cumulative_write_bytes,
            "tokens_scaled": self.tokens_scaled,
            "last_refill_ms": self.last_refill_ms,
            "allowed_events": self.allowed_events,
            "rate_limited_events": self.rate_limited_events,
            "over_quota_events": self.over_quota_events,
            "allowed_write_bytes": self.allowed_write_bytes,
            "allowed_operations": self.allowed_operations,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "VolumeUsage":
        return cls(
            volume_id=str(value["volume_id"]),
            cumulative_write_bytes=int(value.get("cumulative_write_bytes", 0)),
            tokens_scaled=int(value.get("tokens_scaled", 0)),
            last_refill_ms=(
                None if value.get("last_refill_ms") is None else int(value["last_refill_ms"])
            ),
            allowed_events=int(value.get("allowed_events", 0)),
            rate_limited_events=int(value.get("rate_limited_events", 0)),
            over_quota_events=int(value.get("over_quota_events", 0)),
            allowed_write_bytes=int(value.get("allowed_write_bytes", 0)),
            allowed_operations=int(value.get("allowed_operations", 0)),
        )


@dataclass(frozen=True)
class Judgment:
    seq: int
    timestamp_ms: int
    volume_id: str
    decision: str
    reason: str
    operation_count: int
    write_bytes: int
    cumulative_write_bytes: int
    tokens_scaled: int
    tokens_capacity_scaled: int
    last_refill_ms: int | None
    single_error_upper: int = ERROR_BOUND_ZERO
    aggregate_error_upper: int = ERROR_BOUND_ZERO

    def to_json(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp_ms": self.timestamp_ms,
            "volume_id": self.volume_id,
            "decision": self.decision,
            "reason": self.reason,
            "operation_count": self.operation_count,
            "write_bytes": self.write_bytes,
            "cumulative_write_bytes": self.cumulative_write_bytes,
            "tokens_scaled": self.tokens_scaled,
            "tokens_capacity_scaled": self.tokens_capacity_scaled,
            "last_refill_ms": self.last_refill_ms,
            "single_error_upper": self.single_error_upper,
            "aggregate_error_upper": self.aggregate_error_upper,
        }

    def comparable(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "operation_count": self.operation_count,
            "write_bytes": self.write_bytes,
            "cumulative_write_bytes": self.cumulative_write_bytes,
            "tokens_scaled": self.tokens_scaled,
            "tokens_capacity_scaled": self.tokens_capacity_scaled,
            "last_refill_ms": self.last_refill_ms,
            "single_error_upper": self.single_error_upper,
            "aggregate_error_upper": self.aggregate_error_upper,
        }

    def difference_fields(self, other: "Judgment") -> tuple[str, ...]:
        left = self.comparable()
        right = other.comparable()
        return tuple(sorted(key for key in left if left[key] != right[key]))


def judgment_map(judgments: Iterable[Judgment]) -> dict[int, Judgment]:
    return {judgment.seq: judgment for judgment in judgments}
