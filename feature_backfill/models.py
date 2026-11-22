"""Core immutable domain models."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .fingerprints import sha256_json, value_token


METADATA_FIELDS = {"row_id", "entity_id", "timestamp_ms", "missing_fields"}
AGGREGATIONS = {"count", "sum", "mean", "latest"}


@dataclass(frozen=True)
class Row:
    row_id: str
    entity_id: str
    timestamp_ms: int
    values: Mapping[str, float | int | None]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "row_id": self.row_id,
            "entity_id": self.entity_id,
            "timestamp_ms": self.timestamp_ms,
        }
        payload.update({key: self.values[key] for key in sorted(self.values)})
        return payload

    def content_fingerprint(self) -> str:
        return sha256_json(
            {
                "row_id": self.row_id,
                "entity_id": self.entity_id,
                "timestamp_ms": self.timestamp_ms,
                "values": {
                    key: value_token(self.values[key]) for key in sorted(self.values)
                },
            }
        )

    def with_values(self, changes: Mapping[str, Any]) -> "Row":
        new_values = dict(self.values)
        for key, value in changes.items():
            new_values[key] = value
        return replace(self, values=new_values)


@dataclass(frozen=True)
class FeatureDef:
    name: str
    window_ms: int
    aggregation: str
    source_field: str | None = None
    source_feature: str | None = None
    depends_on: tuple[str, ...] = ()

    def parameter_fingerprint(self) -> str:
        return sha256_json(
            {
                "name": self.name,
                "window_ms": self.window_ms,
                "aggregation": self.aggregation,
                "source_field": self.source_field,
                "source_feature": self.source_feature,
                "depends_on": list(self.depends_on),
            }
        )

    def with_window(self, window_ms: int) -> "FeatureDef":
        return replace(self, window_ms=window_ms)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "window_ms": self.window_ms,
            "aggregation": self.aggregation,
            "depends_on": list(self.depends_on),
        }
        if self.source_field is not None:
            payload["source_field"] = self.source_field
        if self.source_feature is not None:
            payload["source_feature"] = self.source_feature
        return payload


@dataclass(frozen=True)
class AppConfig:
    batch_rows: int = 64
    entity_shards: int = 4
    batch_point_quota: int = 100_000
    lease_ttl_ms: int = 2_000
    lease_renew_interval_ms: int = 500
    max_workers: int = 4
    snapshot_every_batches: int = 1
    cache_blocks: int = 64
    clock_source: str = "monotonic_ns"
    missing_marker: str = "__MISSING__"
    lease_wait_timeout_ms: int = 10_000
    snapshot_dir: str = "snapshots"

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_rows": self.batch_rows,
            "entity_shards": self.entity_shards,
            "batch_point_quota": self.batch_point_quota,
            "lease_ttl_ms": self.lease_ttl_ms,
            "lease_renew_interval_ms": self.lease_renew_interval_ms,
            "max_workers": self.max_workers,
            "snapshot_every_batches": self.snapshot_every_batches,
            "cache_blocks": self.cache_blocks,
            "clock_source": self.clock_source,
            "missing_marker": self.missing_marker,
            "lease_wait_timeout_ms": self.lease_wait_timeout_ms,
            "snapshot_dir": self.snapshot_dir,
        }

    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class PlanBatch:
    batch_id: int
    operations: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "operations": [dict(op) for op in self.operations],
        }


@dataclass(frozen=True)
class FeaturePoint:
    entity_id: str
    feature: str
    row_id: str
    timestamp_ms: int
    value: float | int | None
    count: int
    source_row_ids: tuple[str, ...]
    upstream_features: tuple[str, ...]
    missing_sources: tuple[str, ...]
    fingerprint: str
    lease_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "feature": self.feature,
            "row_id": self.row_id,
            "timestamp_ms": self.timestamp_ms,
            "value": self.value,
            "count": self.count,
            "source_row_ids": list(self.source_row_ids),
            "upstream_features": list(self.upstream_features),
            "missing_sources": list(self.missing_sources),
            "fingerprint": self.fingerprint,
            "lease_version": self.lease_version,
        }


@dataclass(frozen=True)
class EntitySummary:
    entity_id: str
    feature: str
    points: int
    non_null_points: int
    missing_skips: int
    value_sum: float
    latest_value: float | int | None
    fingerprint: str
    lease_version: str
    upstream_features: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "feature": self.feature,
            "points": self.points,
            "non_null_points": self.non_null_points,
            "missing_skips": self.missing_skips,
            "value_sum": self.value_sum,
            "latest_value": self.latest_value,
            "fingerprint": self.fingerprint,
            "lease_version": self.lease_version,
            "upstream_features": list(self.upstream_features),
        }
