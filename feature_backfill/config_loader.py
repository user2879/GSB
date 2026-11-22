"""Load and validate YAML config, feature JSON and plan JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .clock import CLOCK_SOURCES
from .errors import (
    AppError,
    E_CONFIG_INVALID,
    E_FEATURE_DAG_INVALID,
    E_PLAN_INVALID,
    make_failure,
)
from .fingerprints import canonical_json, sha256_json
from .models import AGGREGATIONS, AppConfig, FeatureDef, PlanBatch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AppError(
            E_CONFIG_INVALID,
            f"{name} must be a positive integer",
            stage="config",
            details={"field": name, "value": value},
        )
    return value


def parse_config(path: str | Path) -> AppConfig:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - converted to business failure
        raise AppError(
            E_CONFIG_INVALID,
            f"cannot parse YAML config: {exc}",
            stage="config",
            details={"path": str(path)},
        ) from exc
    if not isinstance(raw, dict):
        raise AppError(
            E_CONFIG_INVALID,
            "config root must be an object",
            stage="config",
            details={"path": str(path)},
        )

    def pick(*names: str, default: Any = None) -> Any:
        for name in names:
            if name in raw:
                return raw[name]
        return default

    batch_rows = _positive_int(pick("batch_rows", default=64), "batch_rows")
    entity_shards = _positive_int(pick("entity_shards", default=4), "entity_shards")
    quota = _positive_int(
        pick("batch_point_quota", "point_quota_per_batch", default=100_000),
        "batch_point_quota",
    )
    ttl = _positive_int(pick("lease_ttl_ms", default=2_000), "lease_ttl_ms")
    renew = _positive_int(
        pick("lease_renew_interval_ms", default=min(500, ttl)),
        "lease_renew_interval_ms",
    )
    if renew >= ttl:
        raise AppError(
            E_CONFIG_INVALID,
            "lease_renew_interval_ms must be smaller than lease_ttl_ms",
            stage="config",
            details={"lease_renew_interval_ms": renew, "lease_ttl_ms": ttl},
        )
    workers = _positive_int(
        pick("max_workers", "thread_pool_concurrency", "max_concurrency", default=4),
        "max_workers",
    )
    snapshot_every = _positive_int(
        pick("snapshot_every_batches", "snapshot_commit_interval_batches", default=1),
        "snapshot_every_batches",
    )
    cache_blocks = _positive_int(
        pick("cache_blocks", "cache_resident_blocks", default=64), "cache_blocks"
    )
    wait_timeout = _positive_int(
        pick("lease_wait_timeout_ms", default=max(ttl * 5, 1_000)),
        "lease_wait_timeout_ms",
    )
    clock_source = pick("clock_source", default="monotonic_ns")
    if not isinstance(clock_source, str) or clock_source not in CLOCK_SOURCES:
        raise AppError(
            E_CONFIG_INVALID,
            "clock_source is not a supported monotonic source",
            stage="config",
            details={
                "clock_source": clock_source,
                "allowed": sorted(CLOCK_SOURCES),
            },
        )
    missing_marker = pick("missing_marker", default="__MISSING__")
    if not isinstance(missing_marker, str) or not missing_marker:
        raise AppError(
            E_CONFIG_INVALID,
            "missing_marker must be a non-empty string",
            stage="config",
            details={"missing_marker": missing_marker},
        )
    snapshot_dir = pick("snapshot_dir", default="snapshots")
    if not isinstance(snapshot_dir, str) or not snapshot_dir:
        raise AppError(
            E_CONFIG_INVALID,
            "snapshot_dir must be a non-empty string",
            stage="config",
            details={"snapshot_dir": snapshot_dir},
        )
    return AppConfig(
        batch_rows=batch_rows,
        entity_shards=entity_shards,
        batch_point_quota=quota,
        lease_ttl_ms=ttl,
        lease_renew_interval_ms=renew,
        max_workers=workers,
        snapshot_every_batches=snapshot_every,
        cache_blocks=cache_blocks,
        clock_source=clock_source,
        missing_marker=missing_marker,
        lease_wait_timeout_ms=wait_timeout,
        snapshot_dir=snapshot_dir,
    )


def load_feature_file(path: str | Path) -> tuple[dict[str, FeatureDef], list[str], list[dict[str, Any]]]:
    path = Path(path)
    failures: list[dict[str, Any]] = []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {}, [], [
            make_failure(
                E_FEATURE_DAG_INVALID,
                f"cannot parse feature definition JSON: {exc}",
                stage="feature_dag",
                details={"path": str(path)},
            )
        ]
    if isinstance(raw, dict) and isinstance(raw.get("features"), list):
        items = raw["features"]
        declared_fields = raw.get("raw_fields", [])
    elif isinstance(raw, list):
        items = raw
        declared_fields = []
    else:
        return {}, [], [
            make_failure(
                E_FEATURE_DAG_INVALID,
                "feature definition must be a list or an object containing features",
                stage="feature_dag",
                details={"path": str(path)},
            )
        ]
    raw_fields = [str(item) for item in declared_fields if isinstance(item, str)]
    features: dict[str, FeatureDef] = {}
    for index, item in enumerate(items):
        try:
            if not isinstance(item, dict):
                raise ValueError("feature item must be an object")
            name = item.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("feature name must be non-empty")
            if name in features:
                raise ValueError(f"duplicate feature name: {name}")
            window = item.get("window_ms")
            if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
                raise ValueError(f"{name}: window_ms must be a positive integer")
            aggregation = item.get("aggregation")
            if aggregation not in AGGREGATIONS:
                raise ValueError(f"{name}: unknown aggregation {aggregation!r}")
            source_field = item.get("source_field")
            source_feature = item.get("source_feature")
            if source_field is not None and (not isinstance(source_field, str) or not source_field):
                raise ValueError(f"{name}: source_field must be a non-empty string")
            if source_feature is not None and (
                not isinstance(source_feature, str) or not source_feature
            ):
                raise ValueError(f"{name}: source_feature must be a non-empty string")
            if (source_field is None) == (source_feature is None):
                raise ValueError(f"{name}: exactly one of source_field/source_feature is required")
            depends_raw = item.get("depends_on", [])
            if not isinstance(depends_raw, list) or not all(
                isinstance(value, str) and value for value in depends_raw
            ):
                raise ValueError(f"{name}: depends_on must be a list of feature names")
            depends = tuple(sorted(set(depends_raw)))
            if source_feature is not None and source_feature not in depends:
                raise ValueError(f"{name}: source_feature must be declared in depends_on")
            features[name] = FeatureDef(
                name=name,
                window_ms=window,
                aggregation=aggregation,
                source_field=source_field,
                source_feature=source_feature,
                depends_on=depends,
            )
        except ValueError as exc:
            failures.append(
                make_failure(
                    E_FEATURE_DAG_INVALID,
                    str(exc),
                    stage="feature_dag",
                    details={"index": index},
                )
            )
    if failures:
        return features, raw_fields, failures
    try:
        validate_feature_graph(features, raw_fields)
    except AppError as exc:
        failures.append(exc.to_failure())
    return features, raw_fields, failures


def validate_feature_graph(
    features: dict[str, FeatureDef], raw_fields: list[str] | set[str] | None = None
) -> list[str]:
    if not features:
        raise AppError(
            E_FEATURE_DAG_INVALID,
            "at least one feature is required",
            stage="feature_dag",
        )
    raw_set = set(raw_fields or [])
    for name, feature in features.items():
        if feature.source_feature is not None and feature.source_feature not in features:
            raise AppError(
                E_FEATURE_DAG_INVALID,
                f"{name} references missing upstream feature {feature.source_feature}",
                stage="feature_dag",
                feature=name,
            )
        for dependency in feature.depends_on:
            if dependency not in features:
                raise AppError(
                    E_FEATURE_DAG_INVALID,
                    f"{name} depends on missing feature {dependency}",
                    stage="feature_dag",
                    feature=name,
                )
        if raw_set and feature.source_field is not None and feature.source_field not in raw_set:
            raise AppError(
                E_FEATURE_DAG_INVALID,
                f"{name} references unknown raw field {feature.source_field}",
                stage="feature_dag",
                feature=name,
            )
    indegree = {name: 0 for name in features}
    outgoing: dict[str, set[str]] = {name: set() for name in features}
    for name, feature in features.items():
        for dependency in feature.depends_on:
            outgoing[dependency].add(name)
            indegree[name] += 1
    ready = sorted(name for name, degree in indegree.items() if degree == 0)
    ordered: list[str] = []
    index = 0
    while index < len(ready):
        current = ready[index]
        index += 1
        ordered.append(current)
        for child in sorted(outgoing[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        # Heap-like deterministic insertion while preserving lexicographic order.
        ready[index:] = sorted(ready[index:])
    if len(ordered) != len(features):
        cyclic = sorted(name for name, degree in indegree.items() if degree > 0)
        raise AppError(
            E_FEATURE_DAG_INVALID,
            "feature dependency graph contains a cycle",
            stage="feature_dag",
            details={"cyclic_features": cyclic},
        )
    return ordered


def load_plan(path: str | Path) -> tuple[list[PlanBatch], list[dict[str, Any]]]:
    path = Path(path)
    failures: list[dict[str, Any]] = []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [], [
            make_failure(
                E_PLAN_INVALID,
                f"cannot parse plan JSON: {exc}",
                stage="plan",
                details={"path": str(path)},
            )
        ]
    items = raw.get("batches") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return [], [
            make_failure(
                E_PLAN_INVALID,
                "plan must be a list or contain a batches list",
                stage="plan",
            )
        ]
    batches: list[PlanBatch] = []
    seen: set[int] = set()
    for index, item in enumerate(items):
        try:
            if not isinstance(item, dict):
                raise ValueError("batch must be an object")
            batch_id = item.get("batch_id", index + 1)
            if isinstance(batch_id, bool) or not isinstance(batch_id, int) or batch_id <= 0:
                raise ValueError("batch_id must be a positive integer")
            if batch_id in seen:
                raise ValueError(f"duplicate batch_id {batch_id}")
            seen.add(batch_id)
            operations = item.get("operations", [])
            if not isinstance(operations, list) or not all(
                isinstance(op, dict) for op in operations
            ):
                raise ValueError("operations must be a list of objects")
            for op in operations:
                kind = op.get("op")
                if kind == "append":
                    if not isinstance(op.get("rows"), list):
                        raise ValueError("append.rows must be a list")
                elif kind == "correct":
                    if not isinstance(op.get("row_id"), str) or not isinstance(
                        op.get("changes", op.get("set")), dict
                    ):
                        raise ValueError("correct requires row_id and changes object")
                elif kind == "window_adjust":
                    if not isinstance(op.get("feature"), str):
                        raise ValueError("window_adjust requires feature")
                    window = op.get("window_ms")
                    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
                        raise ValueError("window_adjust.window_ms must be a positive integer")
                elif kind == "rollback":
                    target = op.get("to_batch")
                    back = op.get("batches_back")
                    if target is None and not isinstance(back, int):
                        raise ValueError("rollback requires to_batch or batches_back")
                    if target is not None and (
                        isinstance(target, bool) or not isinstance(target, int)
                    ):
                        raise ValueError("rollback.to_batch must be an integer")
                else:
                    raise ValueError(f"unknown plan operation: {kind!r}")
            batches.append(PlanBatch(batch_id=batch_id, operations=tuple(operations)))
        except ValueError as exc:
            failures.append(
                make_failure(
                    E_PLAN_INVALID,
                    str(exc),
                    stage="plan",
                    details={"index": index},
                )
            )
    if not failures:
        ids = [batch.batch_id for batch in batches]
        if ids != sorted(ids):
            failures.append(
                make_failure(
                    E_PLAN_INVALID,
                    "batch ids must be strictly ascending",
                    stage="plan",
                    details={"batch_ids": ids},
                )
            )
    return batches, failures


def plan_fingerprint(batches: list[PlanBatch]) -> str:
    return sha256_json([batch.to_dict() for batch in batches])


def feature_fingerprint(features: dict[str, FeatureDef]) -> str:
    return sha256_json([features[name].to_dict() for name in sorted(features)])


def canonical_config(config: AppConfig) -> str:
    return canonical_json(config.to_dict())
