"""Deterministic topological window feature computation."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .cache import LRUBlockCache
from .config_loader import validate_feature_graph
from .fingerprints import sha256_json, value_token
from .models import AppConfig, EntitySummary, FeatureDef, FeaturePoint, Row


@dataclass
class EntityComputation:
    entity_id: str
    lease_version: str
    points: list[FeaturePoint]
    summaries: list[EntitySummary]
    cache_events: list[dict[str, Any]] = field(default_factory=list)


def entity_lease_version(
    entity_id: str,
    rows: Iterable[Row] | None = None,
    features: Mapping[str, FeatureDef] | None = None,
    *,
    shard_count: int = 1,
) -> str:
    """Stable entity lease epoch used in cache keys.

    Content changes are already represented by source/upstream fingerprints in
    each block key. Keeping the lease version stable across appends is what
    allows unchanged windows to be reused instead of invalidating every block.
    """

    import hashlib

    shard = int.from_bytes(hashlib.sha256(entity_id.encode("utf-8")).digest()[:8], "big") % shard_count
    return sha256_json(
        {
            "entity_id": entity_id,
            "shard": shard,
            "lease_epoch": "stable-entity-epoch-v1",
        }
    )[:32]


def aggregate_lease_version(versions: Mapping[str, str]) -> str:
    return sha256_json({key: versions[key] for key in sorted(versions)})[:32]


class FeatureComputer:
    def __init__(
        self,
        *,
        rows_by_entity: Mapping[str, list[Row]],
        features: Mapping[str, FeatureDef],
        config: AppConfig,
        cache: LRUBlockCache,
        lease_versions: Mapping[str, str],
        persistent_points: Mapping[tuple[str, str], FeaturePoint] | None = None,
    ) -> None:
        self.rows_by_entity = {
            entity: sorted(rows, key=lambda row: (row.timestamp_ms, row.row_id))
            for entity, rows in rows_by_entity.items()
        }
        self.features = dict(features)
        self.config = config
        self.cache = cache
        self.lease_versions = dict(lease_versions)
        self.persistent_points = dict(persistent_points or {})
        self._persistent_checked: set[tuple[str, str]] = set()
        self.topological_order = validate_feature_graph(
            self.features,
            raw_fields=set(
                feature.source_field
                for feature in self.features.values()
                if feature.source_field is not None
            ),
        )

    def compute_entity(self, entity_id: str) -> EntityComputation:
        rows = self.rows_by_entity.get(entity_id, [])
        lease_version = self.lease_versions[entity_id]
        points: list[FeaturePoint] = []
        events: list[dict[str, Any]] = []
        for feature_name in self.topological_order:
            feature = self.features[feature_name]
            for target_row in rows:
                point = self._ensure_point(
                    entity_id, feature, target_row, lease_version, events
                )
                points.append(point)
        points.sort(
            key=lambda point: (point.timestamp_ms, point.row_id, point.feature)
        )
        summaries: list[EntitySummary] = []
        for feature_name in sorted(self.features):
            feature_points = [point for point in points if point.feature == feature_name]
            non_null_points = sum(point.value is not None for point in feature_points)
            missing_skips = sum(len(point.missing_sources) for point in feature_points)
            numeric_values = [
                float(point.value) for point in feature_points if point.value is not None
            ]
            latest_value = None
            for point in sorted(feature_points, key=lambda item: (item.timestamp_ms, item.row_id)):
                if point.value is not None:
                    latest_value = point.value
            feature_def = self.features[feature_name]
            fingerprint = sha256_json(
                {
                    "entity_id": entity_id,
                    "feature": feature_name,
                    "lease_version": lease_version,
                    "parameter_fingerprint": feature_def.parameter_fingerprint(),
                    "point_fingerprints": [point.fingerprint for point in feature_points],
                    "upstream_features": list(feature_def.depends_on),
                }
            )
            summaries.append(
                EntitySummary(
                    entity_id=entity_id,
                    feature=feature_name,
                    points=len(feature_points),
                    non_null_points=non_null_points,
                    missing_skips=missing_skips,
                    value_sum=math.fsum(numeric_values),
                    latest_value=latest_value,
                    fingerprint=fingerprint,
                    lease_version=lease_version,
                    upstream_features=tuple(feature_def.depends_on),
                )
            )
        events.sort(
            key=lambda event: (
                event.get("entity_id", ""),
                event.get("feature", ""),
                event.get("row_id", ""),
                event.get("kind", ""),
                event.get("key", ""),
            )
        )
        return EntityComputation(
            entity_id=entity_id,
            lease_version=lease_version,
            points=points,
            summaries=summaries,
            cache_events=events,
        )

    def _ensure_point(
        self,
        entity_id: str,
        feature: FeatureDef,
        target_row: Row,
        lease_version: str,
        events: list[dict[str, Any]],
    ) -> FeaturePoint:
        rows = self.rows_by_entity[entity_id]
        timestamps = [row.timestamp_ms for row in rows]
        target_index = next(
            index for index, row in enumerate(rows) if row.row_id == target_row.row_id
        )
        lower = bisect.bisect_left(timestamps, target_row.timestamp_ms - feature.window_ms)
        upper = target_index + 1
        source_rows = rows[lower:upper]
        source_values: list[float | int | None] = []
        upstream_points: list[FeaturePoint | None] = []
        missing_sources: list[str] = []
        source_tokens: list[dict[str, Any]] = []
        for source_row in source_rows:
            if feature.source_field is not None:
                value = source_row.values.get(feature.source_field)
                source_values.append(value)
                token = value_token(value)
                source_tokens.append(
                    {
                        "row_id": source_row.row_id,
                        "timestamp_ms": source_row.timestamp_ms,
                        "value": token,
                    }
                )
                if value is None:
                    missing_sources.append(
                        f"{source_row.row_id}\x1f{feature.source_field}"
                    )
            else:
                assert feature.source_feature is not None
                upstream_feature = self.features[feature.source_feature]
                upstream = self._ensure_point(
                    entity_id,
                    upstream_feature,
                    source_row,
                    lease_version,
                    events,
                )
                upstream_points.append(upstream)
                value = upstream.value
                source_values.append(value)
                source_tokens.append(
                    {
                        "row_id": source_row.row_id,
                        "timestamp_ms": source_row.timestamp_ms,
                        "upstream_fingerprint": upstream.fingerprint,
                        "value": value_token(value),
                    }
                )
                if value is None:
                    missing_sources.append(
                        f"{source_row.row_id}\x1f{feature.source_feature}"
                    )
        block_payload = {
            "entity_id": entity_id,
            "feature": feature.name,
            "target_row_id": target_row.row_id,
            "target_timestamp_ms": target_row.timestamp_ms,
            "parameter_fingerprint": feature.parameter_fingerprint(),
            "lease_version": lease_version,
            "missing_marker": self.config.missing_marker,
            "source_field": feature.source_field,
            "source_feature": feature.source_feature,
            "source_rows": source_tokens,
        }
        block_key = sha256_json(block_payload)
        sort_key = f"{entity_id}|{feature.name}|{target_row.timestamp_ms:020d}|{target_row.row_id}"
        cached = self.cache.get(block_key)
        if cached is not None:
            events.append(
                {
                    "kind": "reuse",
                    "entity_id": entity_id,
                    "feature": feature.name,
                    "row_id": target_row.row_id,
                    "key": block_key,
                }
            )
            return cached
        logical_key = (feature.name, target_row.row_id)
        if logical_key not in self._persistent_checked:
            self._persistent_checked.add(logical_key)
            persisted = self.persistent_points.get(logical_key)
            if (
                persisted is not None
                and persisted.fingerprint == block_key
                and persisted.lease_version == lease_version
            ):
                self.cache.put(block_key, persisted, sort_key=sort_key)
                events.append(
                    {
                        "kind": "reuse",
                        "reuse_source": "persistent_snapshot",
                        "entity_id": entity_id,
                        "feature": feature.name,
                        "row_id": target_row.row_id,
                        "key": block_key,
                    }
                )
                return persisted
        non_missing = [value for value in source_values if value is not None]
        count = len(non_missing)
        if feature.aggregation == "count":
            value: float | int | None = count
        elif feature.aggregation == "sum":
            value = math.fsum(float(item) for item in non_missing) if non_missing else 0.0
        elif feature.aggregation == "mean":
            value = (
                math.fsum(float(item) for item in non_missing) / count
                if non_missing
                else None
            )
        elif feature.aggregation == "latest":
            value = non_missing[-1] if non_missing else None
        else:  # validated earlier
            raise AssertionError(f"unknown aggregation {feature.aggregation}")
        point = FeaturePoint(
            entity_id=entity_id,
            feature=feature.name,
            row_id=target_row.row_id,
            timestamp_ms=target_row.timestamp_ms,
            value=value,
            count=count,
            source_row_ids=tuple(row.row_id for row in source_rows),
            upstream_features=tuple(feature.depends_on),
            missing_sources=tuple(sorted(set(missing_sources))),
            fingerprint=block_key,
            lease_version=lease_version,
        )
        self.cache.put(block_key, point, sort_key=sort_key)
        events.append(
            {
                "kind": "recompute",
                "reuse_source": None,
                "entity_id": entity_id,
                "feature": feature.name,
                "row_id": target_row.row_id,
                "key": block_key,
            }
        )
        return point


def skip_details(points: Iterable[FeaturePoint], features: Mapping[str, FeatureDef]) -> list[dict[str, Any]]:
    details: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for point in points:
        feature = features[point.feature]
        for missing_source in point.missing_sources:
            source_row_id, source_name = missing_source.split("\x1f", 1)
            key = (point.entity_id, point.feature, point.row_id, source_row_id)
            details[key] = {
                "entity_id": point.entity_id,
                "feature": point.feature,
                "target_row_id": point.row_id,
                "target_timestamp_ms": point.timestamp_ms,
                "source_row_id": source_row_id,
                "source": source_name,
                "source_kind": "field" if feature.source_field else "feature",
                "reason": "MISSING_VALUE_SKIPPED",
            }
    return [details[key] for key in sorted(details)]
