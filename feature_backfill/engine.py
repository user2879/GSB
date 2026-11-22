"""Quota-, lease- and snapshot-aware deterministic backfill engine."""

from __future__ import annotations

import json
import os
import traceback
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Mapping

from .cache import LRUBlockCache
from .clock import MonotonicClock
from .compute import (
    EntityComputation,
    FeatureComputer,
    aggregate_lease_version,
    entity_lease_version,
    skip_details,
)
from .config_loader import feature_fingerprint, plan_fingerprint, validate_feature_graph
from .data_loader import dataset_fingerprint, rows_by_entity
from .errors import (
    AppError,
    E_BATCH_QUOTA_EXHAUSTED,
    E_INTERNAL,
    E_LEASE_EXPIRED,
    E_LEASE_LOST,
    E_PLAN_INVALID,
    E_SNAPSHOT_STALE,
    make_failure,
    sorted_failures,
)
from .fingerprints import canonical_json, sha256_json
from .injections import Injection, InjectionRuntime
from .leases import LeaseGuard, LeaseManager, shard_for_entity
from .models import AppConfig, EntitySummary, FeatureDef, FeaturePoint, PlanBatch, Row
from .plan import apply_operations, materialize_plan_state
from .snapshot import SnapshotStore


@dataclass
class RunOptions:
    resume: bool = False
    crash_before_commit_batch: int | None = None
    report_path: str | Path | None = None
    enable_injections: bool = True
    write_report: bool = True


@dataclass
class EngineState:
    rows: list[Row]
    features: dict[str, FeatureDef]
    results: dict[tuple[str, str, str], FeaturePoint] = field(default_factory=dict)
    summaries: dict[tuple[str, str], EntitySummary] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    pending: set[str] = field(default_factory=set)
    completed_batches: list[int] = field(default_factory=list)
    consumed_points_by_batch: dict[int, int] = field(default_factory=dict)
    quota_allocations: list[dict[str, Any]] = field(default_factory=list)
    quota_events: list[dict[str, Any]] = field(default_factory=list)
    lease_versions: dict[str, str] = field(default_factory=dict)
    cache_events: list[dict[str, Any]] = field(default_factory=list)
    injection_hits: list[dict[str, Any]] = field(default_factory=list)
    lease_events: list[dict[str, Any]] = field(default_factory=list)
    invalidation_events: list[dict[str, Any]] = field(default_factory=list)
    global_version: int = 0
    resumed_from_generation: int | None = None
    resumed: bool = False
    fresh_after_uncommitted: bool = False


@dataclass
class WorkerOutcome:
    entity_id: str
    computation: EntityComputation | None = None
    failures: list[dict[str, Any]] = field(default_factory=list)
    error_code: str | None = None


class BackfillEngine:
    def __init__(
        self,
        *,
        base_rows: list[Row],
        base_features: dict[str, FeatureDef],
        raw_fields: Iterable[str],
        config: AppConfig,
        plan: list[PlanBatch],
        base_failures: Iterable[dict[str, Any]] = (),
        injections: Iterable[Injection] = (),
        snapshot_root: str | Path | None = None,
    ) -> None:
        self.base_rows = list(base_rows)
        self.base_features = dict(base_features)
        self.raw_fields = tuple(sorted(set(raw_fields)))
        self.config = config
        self.plan = list(plan)
        self.base_failures = sorted_failures(base_failures)
        self.snapshot_root = Path(snapshot_root or config.snapshot_dir)
        self.snapshot_store = SnapshotStore(self.snapshot_root)
        self.injection_runtime = InjectionRuntime(injections)
        self.plan_hash = plan_fingerprint(self.plan)
        self.base_input_hash = dataset_fingerprint(self.base_rows)
        self.base_feature_hash = feature_fingerprint(self.base_features)
        self.config_hash = sha256_json(
            {
                "config": config.to_dict(),
                "raw_fields": list(self.raw_fields),
                "base_features": {
                    name: self.base_features[name].to_dict()
                    for name in sorted(self.base_features)
                },
            }
        )
        self.clock = MonotonicClock(config.clock_source)
        self.lease_manager = LeaseManager(
            clock=self.clock,
            ttl_ms=config.lease_ttl_ms,
            renew_interval_ms=config.lease_renew_interval_ms,
            shard_count=config.entity_shards,
            wait_timeout_ms=config.lease_wait_timeout_ms,
        )
        self.cache = LRUBlockCache(config.cache_blocks)
        self._worker_lock = Lock()
        self._active_workers = 0
        self._max_observed_workers = 0

    # ------------------------------------------------------------------
    # Serialisation and snapshot compatibility
    # ------------------------------------------------------------------
    def _versions_for_state(
        self, rows: Iterable[Row], features: Mapping[str, FeatureDef]
    ) -> dict[str, str]:
        grouped = rows_by_entity(rows)
        return {
            entity: entity_lease_version(
                entity,
                grouped[entity],
                features,
                shard_count=self.config.entity_shards,
            )
            for entity in sorted(grouped)
        }

    def _state_payload(self, state: EngineState) -> dict[str, Any]:
        points = sorted(
            state.results.values(),
            key=lambda point: (
                point.entity_id,
                point.timestamp_ms,
                point.row_id,
                point.feature,
            ),
        )
        summaries = sorted(
            state.summaries.values(),
            key=lambda summary: (summary.entity_id, summary.feature),
        )
        grouped = rows_by_entity(state.rows)
        skips = skip_details(points, state.features)
        lease_versions = self._versions_for_state(state.rows, state.features)
        dataset_fp = dataset_fingerprint(state.rows)
        feature_fp = feature_fingerprint(state.features)
        completed_entities = sorted(
            entity
            for entity in grouped
            if entity not in state.pending
            and all(
                (entity, feature_name, row.row_id) in state.results
                for feature_name in state.features
                for row in grouped[entity]
            )
        )
        return {
            "format_version": 1,
            "global_version": state.global_version,
            "entities": sorted(grouped),
            "completed_entities": completed_entities,
            "rows": [row.to_dict() for row in state.rows],
            "features": [
                state.features[name].to_dict() for name in sorted(state.features)
            ],
            "results": [point.to_dict() for point in points],
            "summaries": [summary.to_dict() for summary in summaries],
            "skips": skips,
            "failures": sorted_failures(state.failures),
            "pending": sorted(state.pending),
            "completed_batches": list(state.completed_batches),
            "consumed_points_by_batch": {
                str(key): state.consumed_points_by_batch[key]
                for key in sorted(state.consumed_points_by_batch)
            },
            "quota_allocations": list(state.quota_allocations),
            "quota_events": list(state.quota_events),
            "lease_versions": lease_versions,
            "lease_version": aggregate_lease_version(lease_versions),
            "cache_events": sorted(
                state.cache_events,
                key=lambda event: (
                    event.get("entity_id", ""),
                    event.get("feature", ""),
                    event.get("row_id", ""),
                    event.get("kind", ""),
                    event.get("key", ""),
                    event.get("access_sequence", 0),
                ),
            ),
            "injection_hits": sorted(
                state.injection_hits, key=lambda item: item["injection_id"]
            ),
            "lease_events": list(state.lease_events),
            "invalidation_events": list(state.invalidation_events),
            "dataset_fingerprint": dataset_fp,
            "feature_fingerprint": feature_fp,
        }

    def _manifest_fields(
        self,
        state: EngineState,
        *,
        dataset_fp: str,
        feature_fp: str,
        lease_version: str,
    ) -> dict[str, Any]:
        return {
            "global_version": state.global_version,
            "config_hash": self.config_hash,
            "plan_hash": self.plan_hash,
            "base_input_hash": self.base_input_hash,
            "base_feature_fingerprint": self.base_feature_hash,
            "completed_batches": list(state.completed_batches),
            "entities": sorted(rows_by_entity(state.rows)),
            "completed_entities": sorted(
                entity
                for entity in rows_by_entity(state.rows)
                if entity not in state.pending
                and all(
                    (entity, feature, row.row_id) in state.results
                    for feature in state.features
                    for row in rows_by_entity(state.rows)[entity]
                )
            ),
            "consumed_points": sum(state.consumed_points_by_batch.values()),
            "consumed_points_by_batch": {
                str(key): state.consumed_points_by_batch[key]
                for key in sorted(state.consumed_points_by_batch)
            },
            "lease_version": lease_version,
            "lease_versions": self._versions_for_state(state.rows, state.features),
            "dataset_fingerprint": dataset_fp,
            "feature_fingerprint": feature_fp,
            "pending": sorted(state.pending),
        }

    def _deserialize_state(self, payload: dict[str, Any]) -> EngineState:
        rows = [
            Row(
                row_id=item["row_id"],
                entity_id=item["entity_id"],
                timestamp_ms=item["timestamp_ms"],
                values={
                    key: item[key]
                    for key in item
                    if key not in {"row_id", "entity_id", "timestamp_ms"}
                },
            )
            for item in payload["rows"]
        ]
        features = {
            item["name"]: FeatureDef(
                name=item["name"],
                window_ms=item["window_ms"],
                aggregation=item["aggregation"],
                source_field=item.get("source_field"),
                source_feature=item.get("source_feature"),
                depends_on=tuple(item.get("depends_on", [])),
            )
            for item in payload["features"]
        }
        results: dict[tuple[str, str, str], FeaturePoint] = {}
        for item in payload.get("results", []):
            point = FeaturePoint(
                entity_id=item["entity_id"],
                feature=item["feature"],
                row_id=item["row_id"],
                timestamp_ms=item["timestamp_ms"],
                value=item["value"],
                count=item["count"],
                source_row_ids=tuple(item.get("source_row_ids", [])),
                upstream_features=tuple(item.get("upstream_features", [])),
                missing_sources=tuple(item.get("missing_sources", [])),
                fingerprint=item["fingerprint"],
                lease_version=item["lease_version"],
            )
            results[(point.entity_id, point.feature, point.row_id)] = point
        summaries: dict[tuple[str, str], EntitySummary] = {}
        for item in payload.get("summaries", []):
            summary = EntitySummary(
                entity_id=item["entity_id"],
                feature=item["feature"],
                points=item["points"],
                non_null_points=item["non_null_points"],
                missing_skips=item["missing_skips"],
                value_sum=item["value_sum"],
                latest_value=item["latest_value"],
                fingerprint=item["fingerprint"],
                lease_version=item["lease_version"],
                upstream_features=tuple(item.get("upstream_features", [])),
            )
            summaries[(summary.entity_id, summary.feature)] = summary
        return EngineState(
            rows=rows,
            features=features,
            results=results,
            summaries=summaries,
            failures=list(payload.get("failures", [])),
            pending=set(payload.get("pending", [])),
            completed_batches=[int(value) for value in payload.get("completed_batches", [])],
            consumed_points_by_batch={
                int(key): int(value)
                for key, value in payload.get("consumed_points_by_batch", {}).items()
            },
            quota_allocations=list(payload.get("quota_allocations", [])),
            quota_events=list(payload.get("quota_events", [])),
            lease_versions={
                key: value for key, value in payload.get("lease_versions", {}).items()
            },
            cache_events=list(payload.get("cache_events", [])),
            injection_hits=list(payload.get("injection_hits", [])),
            lease_events=list(payload.get("lease_events", [])),
            invalidation_events=list(payload.get("invalidation_events", [])),
            global_version=int(payload.get("global_version", 0)),
        )

    def _load_resume_state(self, state: EngineState) -> None:
        loaded = self.snapshot_store.load_current()
        if loaded is None:
            if self.snapshot_store.uncommitted_temporary_generations():
                state.fresh_after_uncommitted = True
            state.resumed = False
            return
        manifest, payload, generation = loaded
        resumed_state = self._deserialize_state(payload)
        materialized_rows, materialized_features, materialization_failures = (
            materialize_plan_state(
                self.base_rows,
                self.base_features,
                self.plan,
                completed_batches=resumed_state.completed_batches,
                config=self.config,
                raw_fields=self.raw_fields,
            )
        )
        if materialization_failures:
            raise AppError(
                E_SNAPSHOT_STALE,
                "plan cannot reproduce the state recorded in the snapshot",
                stage="snapshot",
                details={"failures": materialization_failures[:5]},
            )
        expected_dataset_fp = dataset_fingerprint(materialized_rows)
        expected_feature_fp = feature_fingerprint(materialized_features)
        expected_lease_versions = self._versions_for_state(
            resumed_state.rows, resumed_state.features
        )
        expected_lease_version = aggregate_lease_version(expected_lease_versions)
        resumed_grouped = rows_by_entity(resumed_state.rows)
        completed_entities = sorted(
            entity
            for entity in resumed_grouped
            if entity not in resumed_state.pending
            and all(
                (entity, feature_name, row.row_id) in resumed_state.results
                for feature_name in resumed_state.features
                for row in resumed_grouped[entity]
            )
        )
        self.snapshot_store.assert_compatible(
            manifest,
            config_hash=self.config_hash,
            plan_hash=self.plan_hash,
            base_input_hash=self.base_input_hash,
            expected_lease_version=expected_lease_version,
            expected_global_version=resumed_state.global_version,
            expected_dataset_fingerprint=expected_dataset_fp,
            expected_feature_fingerprint=expected_feature_fp,
            expected_lease_versions=expected_lease_versions,
            expected_completed_batches=resumed_state.completed_batches,
            expected_pending=sorted(resumed_state.pending),
            expected_completed_entities=completed_entities,
        )
        # The plan oracle and the materialised snapshot must agree field-for-field.
        if (
            dataset_fingerprint(resumed_state.rows) != expected_dataset_fp
            or feature_fingerprint(resumed_state.features) != expected_feature_fp
        ):
            raise AppError(
                E_SNAPSHOT_STALE,
                "snapshot data/features do not match the plan checkpoint",
                stage="snapshot",
                details={"generation": generation},
            )
        if resumed_state.lease_versions != expected_lease_versions:
            raise AppError(
                E_SNAPSHOT_STALE,
                "snapshot lease versions are inconsistent with current content",
                stage="snapshot",
                details={"generation": generation},
            )
        for entity in completed_entities:
            for feature_name in resumed_state.features:
                if (entity, feature_name) not in resumed_state.summaries:
                    raise AppError(
                        E_SNAPSHOT_STALE,
                        "snapshot is missing a summary for a completed entity/feature",
                        stage="snapshot",
                        entity_id=entity,
                        feature=feature_name,
                        details={"generation": generation},
                    )
        resumed_state.resumed = True
        resumed_state.resumed_from_generation = generation
        for key, value in resumed_state.__dict__.items():
            setattr(state, key, value)

    def _commit(
        self,
        state: EngineState,
        *,
        incomplete_after_stage: str | None = None,
    ) -> dict[str, Any]:
        payload = self._state_payload(state)
        manifest_fields = self._manifest_fields(
            state,
            dataset_fp=payload["dataset_fingerprint"],
            feature_fp=payload["feature_fingerprint"],
            lease_version=payload["lease_version"],
        )
        state.lease_versions = dict(payload["lease_versions"])
        if incomplete_after_stage is not None:
            self.snapshot_store.stage_incomplete(
                state_payload=payload,
                manifest_fields=manifest_fields,
                incomplete_after_stage=incomplete_after_stage,
            )
            return {"incomplete": True}
        manifest = self.snapshot_store.commit(
            state_payload=payload,
            manifest_fields=manifest_fields,
        )
        state.global_version = int(manifest["global_version"])
        return manifest

    # ------------------------------------------------------------------
    # Scheduling and workers
    # ------------------------------------------------------------------
    def _worker(
        self,
        *,
        entity_id: str,
        rows: list[Row],
        features: Mapping[str, FeatureDef],
        versions: Mapping[str, str],
        persistent_points: Mapping[tuple[str, str], FeaturePoint],
        owner: str,
    ) -> WorkerOutcome:
        with self._worker_lock:
            self._active_workers += 1
            self._max_observed_workers = max(
                self._max_observed_workers, self._active_workers
            )
        try:
            return self._worker_impl(
                entity_id=entity_id,
                rows=rows,
                features=features,
                versions=versions,
                persistent_points=persistent_points,
                owner=owner,
            )
        finally:
            with self._worker_lock:
                self._active_workers -= 1

    def _worker_impl(
        self,
        *,
        entity_id: str,
        rows: list[Row],
        features: Mapping[str, FeatureDef],
        versions: Mapping[str, str],
        persistent_points: Mapping[tuple[str, str], FeaturePoint],
        owner: str,
    ) -> WorkerOutcome:
        try:
            self.injection_runtime.cache_capacity_for(
                entity_id=entity_id, stage="cache"
            )
            ttl_override = self.injection_runtime.lease_ttl_for(
                entity_id=entity_id, stage="lease"
            )
            faulty_rows, validation_failures = self.injection_runtime.faulty_validation(
                entity_id=entity_id,
                stage="window_features",
                rows=rows,
                config=self.config,
                raw_fields=self.raw_fields,
            )
            if validation_failures:
                return WorkerOutcome(
                    entity_id=entity_id,
                    failures=sorted_failures(validation_failures),
                    error_code=validation_failures[0].get("code"),
                )
            if len(faulty_rows) != len(rows):
                raise AssertionError("injection must not change entity row count")
            if faulty_rows is not rows:
                # Only compute the copied fault-free path when no mutation was
                # requested; a mutated copy is revalidated by the validator.
                rows = faulty_rows
            computer = FeatureComputer(
                rows_by_entity={entity_id: rows},
                features=features,
                config=self.config,
                cache=self.cache,
                lease_versions=versions,
                persistent_points=persistent_points,
            )
            with LeaseGuard(
                self.lease_manager,
                entity_id=entity_id,
                owner=owner,
                version=versions[entity_id],
                ttl_ms=ttl_override,
            ) as guard:
                computation = computer.compute_entity(entity_id)
                guard.validate()
            return WorkerOutcome(entity_id=entity_id, computation=computation)
        except AppError as exc:
            failure = exc.to_failure()
            injection_id = exc.details.get("injection_id")
            if injection_id is not None:
                failure["injection_id"] = injection_id
            return WorkerOutcome(
                entity_id=entity_id,
                failures=[failure],
                error_code=exc.code,
            )
        except BaseException as exc:  # noqa: BLE001 - isolated as a business failure
            return WorkerOutcome(
                entity_id=entity_id,
                failures=[
                    make_failure(
                        E_INTERNAL,
                        f"worker failed: {exc}",
                        stage="runtime",
                        entity_id=entity_id,
                        details={"traceback": traceback.format_exc(limit=8)},
                    )
                ],
                error_code=E_INTERNAL,
            )

    def _allocate(
        self,
        *,
        state: EngineState,
        batch_id: int,
        ready_entities: list[str],
        grouped: Mapping[str, list[Row]],
    ) -> tuple[list[str], list[str], int]:
        allocated: list[str] = []
        used_rows = 0
        used_points = 0
        blocked: list[str] = []
        for entity_id in ready_entities:
            if entity_id in blocked:
                continue
            entity_rows = len(grouped[entity_id])
            cost = entity_rows * len(state.features)
            if (
                used_rows + entity_rows > self.config.batch_rows
                or used_points + cost > self.config.batch_point_quota
            ):
                blocked.extend(
                    candidate
                    for candidate in ready_entities[ready_entities.index(entity_id) :]
                    if candidate not in allocated
                )
                break
            allocated.append(entity_id)
            used_rows += entity_rows
            used_points += cost
            state.quota_allocations.append(
                {
                    "batch_id": batch_id,
                    "entity_id": entity_id,
                    "task_sort_key": f"{entity_id}|window_features",
                    "shard": shard_for_entity(entity_id, self.config.entity_shards),
                    "rows": entity_rows,
                    "feature_count": len(state.features),
                    "points": cost,
                    "cumulative_rows": used_rows,
                    "cumulative_points": used_points,
                    "remaining_points": self.config.batch_point_quota - used_points,
                }
            )
        for entity_id in blocked:
            state.pending.add(entity_id)
            state.quota_events.append(
                {
                    "code": E_BATCH_QUOTA_EXHAUSTED,
                    "stage": "quota",
                    "batch_id": batch_id,
                    "entity_id": entity_id,
                    "entity_shard": shard_for_entity(entity_id, self.config.entity_shards),
                    "remaining_points": self.config.batch_point_quota - used_points,
                    "resource": "batch_rows_or_point_quota",
                }
            )
        return allocated, blocked, used_points

    def _downstream_closure(
        self, features: Mapping[str, FeatureDef], changed_feature: str
    ) -> set[str]:
        closure = {changed_feature}
        changed = True
        while changed:
            changed = False
            for name in sorted(features):
                feature = features[name]
                dependencies = set(feature.depends_on)
                if feature.source_feature is not None:
                    dependencies.add(feature.source_feature)
                if dependencies & closure and name not in closure:
                    closure.add(name)
                    changed = True
        return closure

    def _target_rows_for_source(
        self, rows: list[Row], source_timestamp_ms: int, window_ms: int
    ) -> list[Row]:
        timestamps = [row.timestamp_ms for row in rows]
        left = bisect_left(timestamps, source_timestamp_ms)
        right = bisect_right(timestamps, source_timestamp_ms + window_ms)
        return rows[left:right]

    def _affected_correction_rows(
        self,
        *,
        rows: list[Row],
        corrected_row_id: str,
        changed_fields: set[str],
        features: Mapping[str, FeatureDef],
    ) -> dict[str, set[str]]:
        order = validate_feature_graph(features)
        by_id = {row.row_id: row for row in rows}
        affected: dict[str, set[str]] = {name: set() for name in features}
        for name in order:
            feature = features[name]
            if feature.source_field in changed_fields:
                corrected = by_id[corrected_row_id]
                affected[name].update(
                    row.row_id
                    for row in self._target_rows_for_source(
                        rows, corrected.timestamp_ms, feature.window_ms
                    )
                )
            if feature.source_feature is not None:
                upstream_rows = affected.get(feature.source_feature, set())
                for source_row_id in sorted(upstream_rows):
                    source_row = by_id[source_row_id]
                    affected[name].update(
                        row.row_id
                        for row in self._target_rows_for_source(
                            rows, source_row.timestamp_ms, feature.window_ms
                        )
                    )
        return affected

    def _remove_points(
        self,
        state: EngineState,
        targets: Iterable[tuple[str, str, str]],
        *,
        event: dict[str, Any],
    ) -> list[dict[str, Any]]:
        removed: list[dict[str, Any]] = []
        for key in targets:
            point = state.results.pop(key, None)
            if point is None:
                removed.append(
                    {
                        "entity_id": key[0],
                        "feature": key[1],
                        "row_id": key[2],
                        "old_fingerprint": None,
                    }
                )
            else:
                removed.append(
                    {
                        "entity_id": point.entity_id,
                        "feature": point.feature,
                        "row_id": point.row_id,
                        "old_fingerprint": point.fingerprint,
                    }
                )
        event["invalidated_blocks"] = sorted(
            removed,
            key=lambda item: (item["entity_id"], item["feature"], item["row_id"]),
        )
        state.invalidation_events.append(event)
        return removed

    def _invalidate_edit(
        self,
        *,
        state: EngineState,
        batch: PlanBatch,
        old_rows: list[Row],
        old_features: Mapping[str, FeatureDef],
    ) -> set[str]:
        pending_entities: set[str] = set()
        old_by_id = {row.row_id: row for row in old_rows}
        new_by_id = {row.row_id: row for row in state.rows}
        new_grouped = rows_by_entity(state.rows)
        all_entities = sorted(new_grouped)
        for operation in batch.operations:
            kind = operation.get("op")
            if kind == "append":
                for record in operation.get("rows", []):
                    row_id = record.get("row_id")
                    row = new_by_id.get(row_id)
                    if row is None:
                        continue
                    pending_entities.add(row.entity_id)
                    for key in [
                        key for key in state.summaries if key[0] == row.entity_id
                    ]:
                        del state.summaries[key]
                    state.invalidation_events.append(
                        {
                            "batch_id": batch.batch_id,
                            "operation": "append",
                            "entity_id": row.entity_id,
                            "reason": "new_timestamp_and_downstream",
                            "new_targets": [
                                {"feature": feature_name, "row_id": row.row_id}
                                for feature_name in sorted(state.features)
                            ],
                            "invalidated_blocks": [],
                        }
                    )
            elif kind == "correct":
                old_row_id = operation["row_id"]
                old_row = old_by_id.get(old_row_id)
                changes = dict(operation.get("changes", operation.get("set", {})))
                metadata_keys = {"row_id", "entity_id", "timestamp_ms"}
                metadata_changes = metadata_keys & set(changes)
                new_row = new_by_id.get(str(changes.get("row_id", old_row_id)))
                if old_row is None or new_row is None:
                    continue
                if metadata_changes and any(
                    changes.get(key) != getattr(old_row, key)
                    for key in metadata_changes
                ):
                    entities = {old_row.entity_id, new_row.entity_id}
                    for entity_id in sorted(entities):
                        pending_entities.add(entity_id)
                        for key in [
                            key for key in state.summaries if key[0] == entity_id
                        ]:
                            del state.summaries[key]
                        targets = [
                            key
                            for key in state.results
                            if key[0] == entity_id
                        ]
                        self._remove_points(
                            state,
                            targets,
                            event={
                                "batch_id": batch.batch_id,
                                "operation": "correct",
                                "reason": "row_identity_or_timestamp_changed_full_entity",
                                "entity_id": entity_id,
                                "row_id": old_row_id,
                                "changed_fields": sorted(metadata_changes),
                            },
                        )
                    continue
                changed_fields = {
                    key
                    for key, value in changes.items()
                    if key not in metadata_keys
                    and old_row.values.get(key) != new_row.values.get(key)
                }
                if not changed_fields:
                    continue
                pending_entities.add(new_row.entity_id)
                for key in [
                    key for key in state.summaries if key[0] == new_row.entity_id
                ]:
                    del state.summaries[key]
                affected = self._affected_correction_rows(
                    rows=new_grouped[new_row.entity_id],
                    corrected_row_id=new_row.row_id,
                    changed_fields=changed_fields,
                    features=state.features,
                )
                targets = [
                    (new_row.entity_id, feature_name, row_id)
                    for feature_name, row_ids in sorted(affected.items())
                    for row_id in sorted(row_ids)
                ]
                self._remove_points(
                    state,
                    targets,
                    event={
                        "batch_id": batch.batch_id,
                        "operation": "correct",
                        "reason": "row_window_and_downstream",
                        "entity_id": new_row.entity_id,
                        "row_id": new_row.row_id,
                        "changed_fields": sorted(changed_fields),
                        "affected_features": sorted(
                            name for name, row_ids in affected.items() if row_ids
                        ),
                    },
                )
            elif kind == "window_adjust":
                feature_name = operation["feature"]
                new_window = operation["window_ms"]
                old_feature = old_features[feature_name]
                if old_feature.window_ms == new_window:
                    continue
                closure = self._downstream_closure(state.features, feature_name)
                targets: list[tuple[str, str, str]] = []
                for entity_id in all_entities:
                    pending_entities.add(entity_id)
                    for feature in sorted(closure):
                        for row in new_grouped[entity_id]:
                            targets.append((entity_id, feature, row.row_id))
                    for key in [
                        key
                        for key in state.summaries
                        if key[0] == entity_id and key[1] in closure
                    ]:
                        del state.summaries[key]
                self._remove_points(
                    state,
                    targets,
                    event={
                        "batch_id": batch.batch_id,
                        "operation": "window_adjust",
                        "reason": "window_parameter_changed_all_entities",
                        "feature": feature_name,
                        "old_window_ms": old_feature.window_ms,
                        "new_window_ms": new_window,
                        "affected_features": sorted(closure),
                        "entities": all_entities,
                    },
                )
        return pending_entities

    def _prune_orphan_results(self, state: EngineState) -> None:
        grouped = rows_by_entity(state.rows)
        valid_rows = {
            (entity, row.row_id) for entity, rows in grouped.items() for row in rows
        }
        valid_features = set(state.features)
        for key in list(state.results):
            if (key[0], key[2]) not in valid_rows or key[1] not in valid_features:
                del state.results[key]
        for key in list(state.summaries):
            if key[0] not in grouped or key[1] not in valid_features:
                del state.summaries[key]

    def _run_batch_workers(
        self,
        *,
        state: EngineState,
        batch_id: int,
        allocated: list[str],
        grouped: Mapping[str, list[Row]],
        versions: Mapping[str, str],
    ) -> None:
        if not allocated:
            return
        outcomes: list[WorkerOutcome] = []
        with ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix=f"backfill-b{batch_id}",
        ) as executor:
            futures = {
                executor.submit(
                    self._worker,
                    entity_id=entity_id,
                    rows=grouped[entity_id],
                    features=dict(state.features),
                    versions=versions,
                    persistent_points={
                        (point.feature, point.row_id): point
                        for key, point in state.results.items()
                        if key[0] == entity_id
                    },
                    owner=f"worker-{entity_id}",
                ): entity_id
                for entity_id in allocated
            }
            for future in as_completed(futures):
                outcome = future.result()
                outcomes.append(outcome)
        outcomes.sort(key=lambda outcome: outcome.entity_id)
        for outcome in outcomes:
            if outcome.computation is not None:
                for key in [key for key in state.results if key[0] == outcome.entity_id]:
                    del state.results[key]
                for key in [key for key in state.summaries if key[0] == outcome.entity_id]:
                    del state.summaries[key]
                for point in outcome.computation.points:
                    state.results[(point.entity_id, point.feature, point.row_id)] = point
                for summary in outcome.computation.summaries:
                    state.summaries[(summary.entity_id, summary.feature)] = summary
                state.cache_events.extend(outcome.computation.cache_events)
                state.pending.discard(outcome.entity_id)
            else:
                for key in [key for key in state.results if key[0] == outcome.entity_id]:
                    del state.results[key]
                for key in [key for key in state.summaries if key[0] == outcome.entity_id]:
                    del state.summaries[key]
                state.failures.extend(outcome.failures)
                state.pending.discard(outcome.entity_id)
                if outcome.error_code in {E_LEASE_EXPIRED, E_LEASE_LOST}:
                    state.pending.add(outcome.entity_id)
        state.failures = sorted_failures(state.failures)
        state.cache_events = sorted(
            state.cache_events,
            key=lambda event: (
                event.get("entity_id", ""),
                event.get("feature", ""),
                event.get("row_id", ""),
                event.get("kind", ""),
                event.get("key", ""),
            ),
        )
        hits = {item["injection_id"]: item for item in state.injection_hits}
        hits.update(
            {item["injection_id"]: item for item in self.injection_runtime.hits_sorted()}
        )
        state.injection_hits = [hits[key] for key in sorted(hits)]
        state.lease_events = list(self.lease_manager.events)

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------
    def run(self, options: RunOptions | None = None) -> dict[str, Any]:
        options = options or RunOptions()
        versions = self._versions_for_state(self.base_rows, self.base_features)
        state = EngineState(
            rows=list(self.base_rows),
            features=dict(self.base_features),
            failures=sorted_failures(self.base_failures),
            pending=set(rows_by_entity(self.base_rows)),
            lease_versions=versions,
            injection_hits=self.injection_runtime.hits_sorted(),
        )
        resume_error: AppError | None = None
        if options.resume:
            try:
                self._load_resume_state(state)
            except AppError as exc:
                resume_error = exc
                raise
        completed = set(state.completed_batches)
        last_committed_version: int | None = None
        if state.resumed:
            last_committed_version = state.global_version
        for index, batch in enumerate(self.plan):
            if batch.batch_id in completed:
                continue
            if any(
                operation.get("op") == "rollback" for operation in batch.operations
            ):
                old_entities = sorted(rows_by_entity(state.rows))
                rows, features, plan_failures = self._materialize_rollback_state(
                    batch_index=index
                )
                affected = set(rows_by_entity(state.rows)) | set(rows_by_entity(rows))
                removed_blocks = [
                    {
                        "entity_id": point.entity_id,
                        "feature": point.feature,
                        "row_id": point.row_id,
                        "old_fingerprint": point.fingerprint,
                    }
                    for point in state.results.values()
                ]
                state.rows = rows
                state.features = features
                state.results.clear()
                state.summaries.clear()
                state.failures.extend(plan_failures)
                state.pending = set(rows_by_entity(rows))
                state.invalidation_events.append(
                    {
                        "batch_id": batch.batch_id,
                        "operation": "rollback",
                        "reason": "batch_rollback_discards_all_derived_state",
                        "entities": old_entities,
                        "invalidated_blocks": sorted(
                            removed_blocks,
                            key=lambda item: (
                                item["entity_id"],
                                item["feature"],
                                item["row_id"],
                            ),
                        ),
                    }
                )
            else:
                old_rows = list(state.rows)
                old_features = dict(state.features)
                (
                    new_rows,
                    new_features,
                    edit_failures,
                    affected,
                ) = apply_operations(
                    state.rows,
                    state.features,
                    batch.operations,
                    config=self.config,
                    raw_fields=self.raw_fields,
                )
                if edit_failures:
                    state.failures.extend(edit_failures)
                    # A failed batch is rejected atomically.
                    affected = set()
                else:
                    state.rows = new_rows
                    state.features = new_features
                    invalidated_entities = self._invalidate_edit(
                        state=state,
                        batch=batch,
                        old_rows=old_rows,
                        old_features=old_features,
                    )
                    state.pending.update(invalidated_entities)
                    self._prune_orphan_results(state)
            state.failures = sorted_failures(state.failures)
            versions = self._versions_for_state(state.rows, state.features)
            changed_versions = {
                entity
                for entity, version in versions.items()
                if state.lease_versions.get(entity) != version
            }
            for entity in sorted(changed_versions):
                state.pending.add(entity)
                for key in [key for key in state.results if key[0] == entity]:
                    del state.results[key]
                for key in [key for key in state.summaries if key[0] == entity]:
                    del state.summaries[key]
            state.lease_versions = versions
            grouped = rows_by_entity(state.rows)
            ready = sorted(
                entity
                for entity in (set(state.pending) | set(affected))
                if entity in grouped
            )
            allocated, blocked, used_points = self._allocate(
                state=state,
                batch_id=batch.batch_id,
                ready_entities=ready,
                grouped=grouped,
            )
            state.lease_versions = versions
            self._run_batch_workers(
                state=state,
                batch_id=batch.batch_id,
                allocated=allocated,
                grouped=grouped,
                versions=versions,
            )
            state.lease_events = list(self.lease_manager.events)
            state.consumed_points_by_batch[batch.batch_id] = used_points
            state.completed_batches.append(batch.batch_id)
            state.global_version += 1
            if options.crash_before_commit_batch == batch.batch_id:
                self._commit(state, incomplete_after_stage="window_features")
                os._exit(137)
            if (
                len(state.completed_batches) % self.config.snapshot_every_batches == 0
                or index == len(self.plan) - 1
            ):
                self._commit(state)
                last_committed_version = state.global_version
        if not self.plan and state.pending:
            # An empty plan is the explicit full-backfill baseline. It still
            # obeys one batch of row/point limits via implicit batch id 0.
            grouped = rows_by_entity(state.rows)
            versions = self._versions_for_state(state.rows, state.features)
            state.lease_versions = versions
            ready = sorted(entity for entity in state.pending if entity in grouped)
            allocated, _blocked, used_points = self._allocate(
                state=state,
                batch_id=0,
                ready_entities=ready,
                grouped=grouped,
            )
            self._run_batch_workers(
                state=state,
                batch_id=0,
                allocated=allocated,
                grouped=grouped,
                versions=versions,
            )
            state.consumed_points_by_batch[0] = used_points
            state.global_version += 1
            self._commit(state)
            last_committed_version = state.global_version
        if (
            self.snapshot_store.load_current() is None
            or last_committed_version != state.global_version
        ):
            self._commit(state)
        report = self._build_report(state, options)
        if options.write_report:
            self.write_report(report, options.report_path)
        return report

    def _materialize_rollback_state(
        self, *, batch_index: int
    ) -> tuple[list[Row], dict[str, FeatureDef], list[dict[str, Any]]]:
        batch = self.plan[batch_index]
        rollback = next(
            operation for operation in batch.operations if operation.get("op") == "rollback"
        )
        target = rollback.get("to_batch")
        target_index: int | None = None
        if target is not None:
            target_index = next(
                (
                    idx
                    for idx, candidate in enumerate(self.plan[:batch_index])
                    if candidate.batch_id == target
                ),
                None,
            )
        else:
            back = rollback.get("batches_back")
            if isinstance(back, int) and not isinstance(back, bool) and back > 0:
                target_index = batch_index - back - 1
                if target_index < 0:
                    target_index = -1
        if target_index is None:
            return (
                list(self.base_rows),
                dict(self.base_features),
                [
                    make_failure(
                        E_PLAN_INVALID,
                        "rollback target does not identify an earlier batch",
                        stage="plan",
                        details={"batch_id": batch.batch_id, "operation": rollback},
                    )
                ],
            )
        completed_batches = [
            candidate.batch_id for candidate in self.plan[: target_index + 1]
        ]
        rows, features, failures = materialize_plan_state(
            self.base_rows,
            self.base_features,
            self.plan,
            completed_batches=completed_batches,
            config=self.config,
            raw_fields=self.raw_fields,
        )
        return rows, features, failures

    def _build_report(self, state: EngineState, options: RunOptions) -> dict[str, Any]:
        payload = self._state_payload(state)
        failures = list(payload["failures"])
        for entity_id in sorted(state.pending):
            failures.append(
                make_failure(
                    E_BATCH_QUOTA_EXHAUSTED,
                    "entity remains pending because batch resources were exhausted",
                    stage="quota",
                    entity_id=entity_id,
                    details={"pending_queue_position": sorted(state.pending).index(entity_id)},
                )
            )
        failures = sorted_failures(failures)
        current = self.snapshot_store.load_current()
        generation = current[2] if current is not None else None
        if state.pending:
            status = "partial"
        elif failures:
            status = "completed_with_failures"
        else:
            status = "completed"
        cache_stats = self.cache.stats()
        recomputed_blocks = sorted(
            {
                event["key"]
                for event in state.cache_events
                if event.get("kind") == "recompute"
            }
        )
        reused_blocks = sorted(
            {
                event["key"]
                for event in state.cache_events
                if event.get("kind") == "reuse"
            }
        )
        return {
            "status": status,
            "run": {
                "resumed": state.resumed,
                "resumed_from_generation": state.resumed_from_generation,
                "fresh_after_uncommitted": state.fresh_after_uncommitted,
                "uncommitted_temporary_generations": self.snapshot_store.uncommitted_temporary_generations(),
                "resume_requested": options.resume,
                "global_version": state.global_version,
                "snapshot_generation": generation,
                "snapshot_root": str(self.snapshot_root),
                "config_hash": self.config_hash,
                "plan_hash": self.plan_hash,
                "base_input_hash": self.base_input_hash,
                "clock_source": self.config.clock_source,
                "max_workers": self.config.max_workers,
                "max_observed_concurrency": self._max_observed_workers,
                "snapshot_every_batches": self.config.snapshot_every_batches,
                "entity_shards": {
                    entity: shard_for_entity(entity, self.config.entity_shards)
                    for entity in sorted(rows_by_entity(state.rows))
                },
            },
            "entities": payload["entities"],
            "completed_entities": payload["completed_entities"],
            "results": payload["results"],
            "summaries": payload["summaries"],
            "skips": payload["skips"],
            "pending": payload["pending"],
            "failures": failures,
            "quota_allocations": state.quota_allocations,
            "quota_events": state.quota_events,
            "consumed_points_by_batch": payload["consumed_points_by_batch"],
            "completed_batches": state.completed_batches,
            "lease_versions": payload["lease_versions"],
            "lease_version": payload["lease_version"],
            "lease_events": state.lease_events,
            "invalidation_manifest": state.invalidation_events,
            "cache": {
                **cache_stats,
                "recomputed_blocks": recomputed_blocks,
                "reused_blocks": reused_blocks,
                "block_executions": state.cache_events,
                "events": self.cache.sorted_events(),
            },
            "injection_hits": sorted(
                state.injection_hits, key=lambda item: item["injection_id"]
            ),
            "base_failures": self.base_failures,
        }

    def write_report(self, report: Mapping[str, Any], path: str | Path | None) -> None:
        if path is None:
            return
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        os.replace(temporary, target)


def canonical_comparison(report: Mapping[str, Any]) -> dict[str, Any]:
    """Fields that must be byte-identical for incremental/full equivalence."""

    return {
        "results": report.get("results", []),
        "summaries": report.get("summaries", []),
        "skips": report.get("skips", []),
        "pending": report.get("pending", []),
        "failures": report.get("failures", []),
        "lease_versions": report.get("lease_versions", {}),
    }


def execution_comparison(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **canonical_comparison(report),
        "quota_allocations": report.get("quota_allocations", []),
        "consumed_points_by_batch": report.get("consumed_points_by_batch", {}),
        "completed_batches": report.get("completed_batches", []),
    }


def compare_reports(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_canonical = canonical_comparison(left)
    right_canonical = canonical_comparison(right)
    left_text = canonical_json(left_canonical)
    right_text = canonical_json(right_canonical)
    mismatches: list[str] = []
    if left_text != right_text:
        for key in left_canonical:
            if canonical_json(left_canonical[key]) != canonical_json(right_canonical[key]):
                mismatches.append(key)
    return {
        "equal": not mismatches,
        "mismatches": mismatches,
        "left_sha256": sha256_json(left_canonical),
        "right_sha256": sha256_json(right_canonical),
    }
