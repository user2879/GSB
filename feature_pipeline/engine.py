"""End-to-end staged incremental feature/sample engine."""

from __future__ import annotations

import copy
import signal
import time
from pathlib import Path
from typing import Any

from .canonical import (
    deep_set,
    sha256_value,
    sorted_failures,
)
from .compute import compute_feature_levels, sort_feature_records
from .injections import InjectionTracker, load_injections
from .models import FeatureTable, LabelDefinition, PipelineConfig, definition_hash
from .samples import generate_samples, sort_samples
from .state import RuntimeState, SnapshotStore, canonical_cache_ledger as cache_ledger
from .validation import (
    E_EVENT_INVALID,
    E_SCHEMA_UNKNOWN_FIELD,
    E_TIMESTAMP_INVALID,
    failure,
    load_event_file,
    load_json_document,
    sort_events,
    validate_config,
    validate_event,
    validate_feature_table,
    validate_label_definition,
)

EDIT_COMMON_FIELDS = {"kind"}
EDIT_FIELDS = {
    "backfill": {"kind", "event_id", "patch", "timestamp"},
    "feature_window": {"kind", "feature", "window_ms"},
    "feature_aggregation": {"kind", "feature", "aggregation"},
    "label_delay": {"kind", "delay_ms"},
}


class StopController:
    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None
        self._previous: dict[int, Any] = {}

    def install(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle)
            except (ValueError, OSError):
                pass

    def restore(self) -> None:
        for signum, handler in self._previous.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                pass

    def _handle(self, signum: int, _frame: Any) -> None:
        self.requested = True
        self.signal_number = signum


class PipelineEngine:
    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.store = SnapshotStore(self.state_dir)

    def run(
        self,
        *,
        config_path: Path,
        events_path: Path,
        features_path: Path,
        labels_path: Path,
        edits_path: Path | None = None,
        injections_path: Path | None = None,
        reset: bool = False,
        stop_after: str | None = None,
    ) -> tuple[dict[str, Any], int]:
        if stop_after not in {None, "events", "features", "samples"}:
            raise ValueError("stop_after must be events, features, or samples")
        if reset:
            self.store.reset()
        base_state, recovery = self.store.recover()
        stop = StopController()
        stop.install()
        try:
            return self._run_inner(
                base_state=base_state,
                recovery=recovery,
                config_path=Path(config_path),
                events_path=Path(events_path),
                features_path=Path(features_path),
                labels_path=Path(labels_path),
                edits_path=Path(edits_path) if edits_path else None,
                injections_path=Path(injections_path) if injections_path else None,
                stop=stop,
                stop_after=stop_after,
                reset=reset,
            )
        finally:
            stop.restore()

    def _run_inner(
        self,
        *,
        base_state: RuntimeState,
        recovery: dict[str, Any],
        config_path: Path,
        events_path: Path,
        features_path: Path,
        labels_path: Path,
        edits_path: Path | None,
        injections_path: Path | None,
        stop: StopController,
        stop_after: str | None,
        reset: bool,
    ) -> tuple[dict[str, Any], int]:
        stage_times: dict[str, float] = {}
        failures: list[dict[str, Any]] = []
        started = time.perf_counter()

        config_raw, config_failures = load_json_document(config_path, "config")
        config, config_validation_failures = validate_config(config_raw)
        failures.extend(config_failures)
        failures.extend(config_validation_failures)
        if config is None:
            return self._fatal_report(recovery, failures, stage_times), 2

        tracker, injection_failures = load_injections(injections_path, load_json_document)
        failures.extend(injection_failures)

        feature_raw, feature_load_failures = load_json_document(features_path, "schema")
        label_raw, label_load_failures = load_json_document(labels_path, "schema")
        failures.extend(feature_load_failures)
        failures.extend(label_load_failures)
        label_raw = tracker.inject_label(label_raw)

        edits, edit_failures = self._load_edits(edits_path)
        failures.extend(edit_failures)
        feature_raw = self._apply_definition_edits(feature_raw, edits, failures)
        label_raw = self._apply_label_edits(label_raw, edits, failures)

        feature_table, feature_validation_failures = validate_feature_table(feature_raw)
        failures.extend(feature_validation_failures)
        if feature_table is None:
            feature_table = FeatureTable.empty()
        label_definition, label_validation_failures = validate_label_definition(
            label_raw, feature_table
        )
        failures.extend(label_validation_failures)

        def transform(raw: Any, _line_number: int) -> Any:
            return tracker.transform_source_event(raw)

        source_events, source_failures = load_event_file(
            events_path, set(config.allowed_event_types), transform=transform
        )
        failures.extend(source_failures)
        working_events, backfills, changed_ids, new_ids, event_failures = self._merge_events(
            base_state.events,
            source_events,
            edits,
            tracker,
            set(config.allowed_event_types),
            base_state.event_cursor,
        )
        failures.extend(event_failures)
        working_events = sort_events(working_events)
        event_cursor = max((event["timestamp"] for event in working_events), default=-1)
        final_labels = (
            label_definition.normalized()
            if label_definition is not None
            else (label_raw if isinstance(label_raw, dict) else {})
        )
        final_definitions = {
            "features": feature_table.normalized_raw()["features"],
            "labels": final_labels,
        }
        final_definition_hash = definition_hash(feature_table, final_labels)
        invalidated, reasons = self._invalidation_set(
            base_state,
            feature_table,
            label_definition,
            backfills,
            reset,
        )
        for backfill in backfills:
            backfill["affected_features"] = sorted(
                name
                for name in invalidated
                if name in feature_table.definitions
                and self._feature_matches_event(
                    feature_table.definitions[name], backfill.get("_old_event", {}), backfill.get("_new_event", {})
                )
            )

        state = base_state.clone()
        state.events = copy.deepcopy(working_events)
        state.definitions = copy.deepcopy(final_definitions)
        state.backfills = self._compose_backfill_ledger(base_state.backfills, backfills)
        state = self.store.commit(
            state,
            "events",
            definition_hash=final_definition_hash,
            event_cursor=event_cursor,
            completed_stages=["events"],
        )
        stage_times["events"] = _elapsed_ms(started)
        if self._should_stop(stop, config, stop_after, "events"):
            return (
                self._build_report(
                    state=state,
                    recovery=recovery,
                    failures=failures,
                    stage_times=stage_times,
                    invalidated=invalidated,
                    reused=set(feature_table.definitions) - invalidated,
                    computed=[],
                    reasons=reasons,
                    regenerated=[],
                    invalidated_samples=[],
                    reused_samples=[],
                    backfills=backfills,
                    tracker=tracker,
                    stop=stop,
                    stop_after=stop_after,
                ),
                0,
            )

        feature_started = time.perf_counter()
        previous_feature_by_key = {
            (record["feature"], record["event_id"]): record for record in state.features
        }
        feature_records, computed_features, feature_failures = compute_feature_levels(
            feature_table,
            working_events,
            state.features,
            invalidated,
            max_workers=config.max_workers,
        )
        failures.extend(feature_failures)
        state.features = sort_feature_records(feature_records)
        state.cache_ledger = cache_ledger(state.features)
        state = self.store.commit(
            state,
            "features",
            definition_hash=final_definition_hash,
            event_cursor=event_cursor,
            completed_stages=["events", "features"],
        )
        stage_times["features"] = _elapsed_ms(feature_started)
        if self._should_stop(stop, config, stop_after, "features"):
            return (
                self._build_report(
                    state=state,
                    recovery=recovery,
                    failures=failures,
                    stage_times=stage_times,
                    invalidated=invalidated,
                    reused=set(feature_table.definitions) - invalidated,
                    computed=computed_features,
                    reasons=reasons,
                    regenerated=[],
                    invalidated_samples=[],
                    reused_samples=[],
                    backfills=backfills,
                    tracker=tracker,
                    stop=stop,
                    stop_after=stop_after,
                ),
                0,
            )

        sample_started = time.perf_counter()
        if label_definition is None:
            samples: list[dict[str, Any]] = []
            regenerated: list[str] = []
            reused_samples: list[str] = []
            invalidated_samples = sorted(sample["sample_id"] for sample in state.samples)
            if not label_validation_failures:
                failures.append(
                    failure(
                        E_EVENT_INVALID,
                        "sample",
                        "<labels>",
                        "LabelUnavailableError",
                        "label definition is unavailable",
                    )
                )
        else:
            (
                samples,
                regenerated,
                reused_samples,
                invalidated_samples,
                sample_failures,
            ) = generate_samples(
                working_events,
                state.features,
                label_definition,
                event_cursor,
                state.samples,
            )
            failures.extend(sample_failures)
        state.samples = sort_samples(samples)
        self._attach_backfill_sample_impacts(
            backfills,
            previous_feature_by_key,
            state.features,
            invalidated_samples,
            label_definition,
        )
        state.backfills = self._compose_backfill_ledger(base_state.backfills, backfills)
        state = self.store.commit(
            state,
            "samples",
            definition_hash=final_definition_hash,
            event_cursor=event_cursor,
            completed_stages=["events", "features", "samples"],
        )
        stage_times["samples"] = _elapsed_ms(sample_started)
        report = self._build_report(
            state=state,
            recovery=recovery,
            failures=failures,
            stage_times=stage_times,
            invalidated=invalidated,
            reused=set(feature_table.definitions) - invalidated,
            computed=computed_features,
            reasons=reasons,
            regenerated=regenerated,
            invalidated_samples=invalidated_samples,
            reused_samples=reused_samples,
            backfills=backfills,
            tracker=tracker,
            stop=stop,
            stop_after=stop_after,
        )
        return report, 0

    def _load_edits(
        self, path: Path | None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if path is None:
            return [], []
        raw, failures = load_json_document(path, "edit")
        if failures:
            return [], failures
        if isinstance(raw, list):
            raw = {"edits": raw}
        if not isinstance(raw, dict) or set(raw) - {"edits"}:
            return [], [
                failure(
                    E_SCHEMA_UNKNOWN_FIELD,
                    "edit",
                    "<edits>",
                    "SchemaError",
                    "edit file must contain only an edits list",
                )
            ]
        items = raw.get("edits")
        if not isinstance(items, list):
            return [], [
                failure(E_EVENT_INVALID, "edit", "<edits>", "TypeError", "edits must be a list")
            ]
        valid: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            location = f"<edit:{index}>"
            if not isinstance(item, dict) or item.get("kind") not in EDIT_FIELDS:
                failures.append(
                    failure(
                        E_SCHEMA_UNKNOWN_FIELD,
                        "edit",
                        location,
                        "EditKindError",
                        "edit kind must be backfill, feature_window, feature_aggregation, or label_delay",
                    )
                )
                continue
            kind = item["kind"]
            unknown = sorted(set(item) - EDIT_FIELDS[kind])
            for field in unknown:
                failures.append(
                    failure(
                        E_SCHEMA_UNKNOWN_FIELD,
                        "edit",
                        location,
                        "UnknownFieldError",
                        f"unknown {kind} edit field: {field}",
                        field=field,
                    )
                )
            if unknown:
                continue
            if kind == "backfill":
                if not isinstance(item.get("event_id"), str) or not item["event_id"]:
                    failures.append(
                        failure(E_EVENT_INVALID, "edit", location, "IdentifierError", "event_id is required")
                    )
                    continue
                if "timestamp" in item and (
                    isinstance(item["timestamp"], bool)
                    or not isinstance(item["timestamp"], int)
                    or item["timestamp"] < 0
                ):
                    failures.append(
                        failure(
                            E_TIMESTAMP_INVALID,
                            "edit",
                            item["event_id"],
                            "TimestampError",
                            "backfill timestamp must be a non-negative integer",
                        )
                    )
                    continue
                patch = item.get("patch", {})
                if not isinstance(patch, dict) or any(not isinstance(key, str) for key in patch):
                    failures.append(
                        failure(E_SCHEMA_UNKNOWN_FIELD, "edit", location, "PatchError", "patch must be an object")
                    )
                    continue
            elif kind == "feature_window":
                if not isinstance(item.get("feature"), str) or not item["feature"]:
                    failures.append(
                        failure(E_EVENT_INVALID, "edit", location, "FeatureError", "feature is required")
                    )
                    continue
                if (
                    isinstance(item.get("window_ms"), bool)
                    or not isinstance(item.get("window_ms"), int)
                    or item["window_ms"] < 0
                ):
                    failures.append(
                        failure(E_EVENT_INVALID, "edit", location, "WindowError", "window_ms must be >= 0")
                    )
                    continue
            elif kind == "feature_aggregation":
                if not isinstance(item.get("feature"), str) or not item["feature"]:
                    failures.append(
                        failure(E_EVENT_INVALID, "edit", location, "FeatureError", "feature is required")
                    )
                    continue
                if item.get("aggregation") not in {"count", "sum", "mean", "latest"}:
                    failures.append(
                        failure(E_SCHEMA_UNKNOWN_FIELD, "edit", location, "AggregationError", "invalid aggregation")
                    )
                    continue
            elif kind == "label_delay":
                if (
                    isinstance(item.get("delay_ms"), bool)
                    or not isinstance(item.get("delay_ms"), int)
                    or item["delay_ms"] < 0
                ):
                    failures.append(
                        failure(E_TIMESTAMP_INVALID, "edit", location, "DelayError", "delay_ms must be >= 0")
                    )
                    continue
            valid.append(copy.deepcopy(item))
        return valid, failures

    def _apply_definition_edits(
        self, raw: Any, edits: list[dict[str, Any]], failures: list[dict[str, Any]]
    ) -> Any:
        if raw is None:
            return raw
        result = copy.deepcopy(raw)
        if not isinstance(result, dict) or not isinstance(result.get("features"), list):
            return result
        by_name = {
            feature.get("name"): feature
            for feature in result["features"]
            if isinstance(feature, dict)
        }
        for edit in edits:
            if edit["kind"] not in {"feature_window", "feature_aggregation"}:
                continue
            name = edit["feature"]
            if name not in by_name:
                failures.append(
                    failure(
                        E_SCHEMA_UNKNOWN_FIELD,
                        "edit",
                        f"<feature:{name}>",
                        "MissingFeatureError",
                        f"edit references unknown feature: {name}",
                    )
                )
                continue
            if edit["kind"] == "feature_window":
                by_name[name]["window_ms"] = edit["window_ms"]
            else:
                by_name[name]["aggregation"] = edit["aggregation"]
        return result

    def _apply_label_edits(
        self, raw: Any, edits: list[dict[str, Any]], failures: list[dict[str, Any]]
    ) -> Any:
        if raw is None:
            return raw
        if not any(edit["kind"] == "label_delay" for edit in edits):
            return raw
        if not isinstance(raw, dict):
            failures.append(
                failure(
                    E_SCHEMA_UNKNOWN_FIELD,
                    "edit",
                    "<labels>",
                    "LabelSchemaError",
                    "label definition must be an object for label_delay edit",
                )
            )
            return raw
        result = copy.deepcopy(raw)
        for edit in edits:
            if edit["kind"] == "label_delay":
                result["delay_ms"] = edit["delay_ms"]
        return result

    def _merge_events(
        self,
        base_events: list[dict[str, Any]],
        source_events: list[dict[str, Any]],
        edits: list[dict[str, Any]],
        tracker: InjectionTracker,
        allowed_event_types: set[str],
        base_cursor: int,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        set[str],
        set[str],
        list[dict[str, Any]],
    ]:
        base_by_id = {event["event_id"]: event for event in base_events}
        source_by_id = {event["event_id"]: event for event in source_events}
        desired: dict[str, dict[str, Any]] = {}
        failures: list[dict[str, Any]] = []
        for event_id, event in source_by_id.items():
            if event_id not in base_by_id or event != base_by_id[event_id]:
                desired[event_id] = copy.deepcopy(event)
        for edit in edits:
            if edit["kind"] != "backfill":
                continue
            event_id = edit["event_id"]
            if event_id in desired:
                target = desired[event_id]
            elif event_id in source_by_id:
                target = copy.deepcopy(source_by_id[event_id])
            elif event_id in base_by_id:
                target = copy.deepcopy(base_by_id[event_id])
            else:
                failures.append(
                    failure(
                        E_EVENT_INVALID,
                        "backfill",
                        event_id,
                        "MissingEventError",
                        f"backfill event not found: {event_id}",
                    )
                )
                continue
            if "timestamp" in edit:
                target["timestamp"] = edit["timestamp"]
            for dotted_path, value in edit.get("patch", {}).items():
                try:
                    deep_set(target, dotted_path, value)
                except (KeyError, TypeError) as exc:
                    failures.append(
                        failure(
                            E_SCHEMA_UNKNOWN_FIELD,
                            "backfill",
                            event_id,
                            "PatchError",
                            f"cannot apply patch {dotted_path}: {exc}",
                        )
                    )
            target, _injection = tracker.apply_backfill(target)
            desired[event_id] = target

        merged = {event_id: copy.deepcopy(event) for event_id, event in base_by_id.items()}
        backfills: list[dict[str, Any]] = []
        changed_ids: set[str] = set()
        new_ids: set[str] = set()
        for event_id in sorted(desired):
            candidate = desired[event_id]
            validated, event_failures = validate_event(
                candidate, allowed_event_types, set(), None
            )
            if event_failures:
                failures.extend(event_failures)
                continue
            assert validated is not None
            old_event = base_by_id.get(event_id)
            if old_event is None:
                if base_cursor >= 0 and validated["timestamp"] <= base_cursor:
                    failures.append(
                        failure(
                            E_TIMESTAMP_INVALID,
                            "event_load",
                            event_id,
                            "WatermarkError",
                            f"new event timestamp {validated['timestamp']} is not after cursor {base_cursor}",
                        )
                    )
                    continue
                if validated["timestamp"] in {
                    event["timestamp"] for event in merged.values()
                }:
                    failures.append(
                        failure(
                            E_TIMESTAMP_INVALID,
                            "event_load",
                            event_id,
                            "TimestampCollisionError",
                            "event timestamp is not unique",
                        )
                    )
                    continue
                merged[event_id] = validated
                new_ids.add(event_id)
                continue
            if validated == old_event:
                continue
            timestamp_changed = validated["timestamp"] != old_event["timestamp"]
            if timestamp_changed and validated["timestamp"] < base_cursor:
                failures.append(
                    failure(
                        E_TIMESTAMP_INVALID,
                        "backfill",
                        event_id,
                        "WatermarkError",
                        f"corrected timestamp {validated['timestamp']} is before cursor {base_cursor}",
                    )
                )
                continue
            timestamps_excluding_self = {
                event["timestamp"]
                for other_id, event in merged.items()
                if other_id != event_id
            }
            if validated["timestamp"] in timestamps_excluding_self:
                failures.append(
                    failure(
                        E_TIMESTAMP_INVALID,
                        "backfill",
                        event_id,
                        "TimestampCollisionError",
                        "corrected timestamp is not unique",
                    )
                )
                continue
            merged[event_id] = validated
            changed_ids.add(event_id)
            backfills.append(
                {
                    "event_id": event_id,
                    "previous_timestamp": old_event["timestamp"],
                    "corrected_timestamp": validated["timestamp"],
                    "affected_user_id": validated["user_id"],
                    "affected_item_id": validated["item_id"],
                    "affected_features": [],
                    "affected_samples": [],
                    "_old_event": old_event,
                    "_new_event": validated,
                }
            )
        return list(merged.values()), backfills, changed_ids, new_ids, failures

    def _invalidation_set(
        self,
        base_state: RuntimeState,
        table: FeatureTable,
        label: LabelDefinition | None,
        backfills: list[dict[str, Any]],
        reset: bool,
    ) -> tuple[set[str], dict[str, list[str]]]:
        reasons: dict[str, list[str]] = {name: [] for name in table.definitions}
        direct: set[str] = set()
        base_features = {
            feature.get("name"): feature
            for feature in base_state.definitions.get("features", [])
            if isinstance(feature, dict)
        }
        final_features = {definition.name: definition.normalized() for definition in table.definitions.values()}
        if base_state.generation > 0 and "features" not in base_state.completed_stages:
            for name in sorted(final_features):
                direct.add(name)
                reasons.setdefault(name, []).append("recovery_recompute")
        for name in sorted(final_features):
            if reset or name not in base_features or base_features[name] != final_features[name]:
                direct.add(name)
                reasons.setdefault(name, []).append("feature_definition_change")
        for backfill in backfills:
            old_event = backfill.get("_old_event", {})
            new_event = backfill.get("_new_event", {})
            for definition in table.definitions.values():
                if self._feature_matches_event(definition, old_event, new_event):
                    direct.add(definition.name)
                    reasons.setdefault(definition.name, []).append(
                        f"event_backfill:{backfill['event_id']}"
                    )
        invalidated = table.descendants(direct)
        for feature in sorted(invalidated):
            if feature not in direct:
                parents = [
                    definition.name
                    for definition in table.definitions.values()
                    if feature in table.descendants({definition.name})
                    and definition.name != feature
                ]
                if parents:
                    reasons.setdefault(feature, []).append(
                        "downstream_of:" + ",".join(sorted(parents))
                    )
        return invalidated, {key: sorted(set(value)) for key, value in sorted(reasons.items())}

    @staticmethod
    def _feature_matches_event(
        definition: Any, old_event: dict[str, Any], new_event: dict[str, Any]
    ) -> bool:
        if not old_event or not new_event:
            return True
        field = "user_id" if definition.entity == "user" else "item_id"
        if old_event.get(field) != new_event.get(field):
            return True
        if old_event.get("timestamp") != new_event.get("timestamp"):
            return True
        if old_event.get("event_type") != new_event.get("event_type"):
            if (
                definition.source_field == "event_type"
                or (
                    definition.event_filter is not None
                    and definition.event_filter.field == "event_type"
                )
            ):
                return True
        old_context = old_event.get("context", {})
        new_context = new_event.get("context", {})
        changed_keys = {
            key
            for key in set(old_context) | set(new_context)
            if old_context.get(key) != new_context.get(key)
        }
        for key in changed_keys:
            dotted = f"context.{key}"
            if definition.source_field == dotted:
                return True
            if (
                definition.event_filter is not None
                and definition.event_filter.field == dotted
            ):
                return True
        return False

    @staticmethod
    def _public_backfills(backfills: list[dict[str, Any]]) -> list[dict[str, Any]]:
        public: list[dict[str, Any]] = []
        for backfill in sorted(backfills, key=lambda item: item["event_id"]):
            public.append(
                {
                    key: copy.deepcopy(value)
                    for key, value in backfill.items()
                    if not key.startswith("_")
                }
            )
        return public

    def _compose_backfill_ledger(
        self, previous: list[dict[str, Any]], current: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        ledger = copy.deepcopy(previous)
        for offset, record in enumerate(self._public_backfills(current), 1):
            record["ledger_seq"] = len(previous) + offset
            ledger.append(record)
        return ledger

    def _attach_backfill_sample_impacts(
        self,
        backfills: list[dict[str, Any]],
        old_features: dict[tuple[str, str], dict[str, Any]],
        new_features: list[dict[str, Any]],
        invalidated_samples: list[str],
        label: LabelDefinition | None,
    ) -> None:
        new_lookup = {(record["feature"], record["event_id"]): record for record in new_features}
        changed_features_by_anchor: set[tuple[str, str]] = set()
        for key, new_record in new_lookup.items():
            old_record = old_features.get(key)
            if old_record != new_record:
                changed_features_by_anchor.add(key)
        label_features = set(label.feature_names) if label else set()
        invalidated_set = set(invalidated_samples)
        for backfill in backfills:
            impacted: set[str] = set()
            affected_features = set(backfill.get("affected_features", []))
            for feature_name, event_id in changed_features_by_anchor:
                if feature_name not in affected_features:
                    continue
                if feature_name in label_features:
                    sample_id = f"sample:{label.name}:{event_id}" if label else ""
                    if sample_id in invalidated_set:
                        impacted.add(sample_id)
            if label is not None:
                anchored = f"sample:{label.name}:{backfill['event_id']}"
                if anchored in invalidated_set:
                    impacted.add(anchored)
            backfill["affected_samples"] = sorted(impacted)

    def _build_report(
        self,
        *,
        state: RuntimeState,
        recovery: dict[str, Any],
        failures: list[dict[str, Any]],
        stage_times: dict[str, float],
        invalidated: set[str],
        reused: set[str],
        computed: list[str],
        reasons: dict[str, list[str]],
        regenerated: list[str],
        invalidated_samples: list[str],
        reused_samples: list[str],
        backfills: list[dict[str, Any]],
        tracker: InjectionTracker,
        stop: StopController,
        stop_after: str | None,
    ) -> dict[str, Any]:
        ordered_failures = sorted_failures(failures)
        public_backfills = self._public_backfills(backfills)
        feature_snapshot_version = sha256_value(state.cache_ledger)
        sample_snapshot_version = sha256_value(state.samples)
        stopped = (
            stop.requested
            or (stop_after is not None and stop_after in state.completed_stages)
        )
        return {
            "report_version": 1,
            "global_version": state.global_version,
            "snapshot_versions": state.manifest.get("stage_versions", {}),
            "completed_stages": state.completed_stages,
            "definition_hash": state.manifest.get("definition_hash"),
            "event_cursor": state.event_cursor,
            "event_count": len(state.events),
            "feature_count": len(state.features),
            "sample_count": len(state.samples),
            "empty_window_feature_count": sum(1 for record in state.features if record["empty"]),
            "feature_snapshot_version": feature_snapshot_version,
            "sample_snapshot_version": sample_snapshot_version,
            "invalidation_ledger": {
                "invalidated_features": sorted(invalidated),
                "reused_features": sorted(reused),
                "computed_features": sorted(set(computed)),
                "reasons": [
                    {"feature": feature, "reasons": values}
                    for feature, values in sorted(reasons.items())
                    if feature in invalidated and values
                ],
                "invalidated_samples": sorted(invalidated_samples),
                "regenerated_samples": sorted(regenerated),
                "reused_samples": sorted(reused_samples),
            },
            "sample_regeneration_list": sorted(regenerated),
            "backfill_impacts": public_backfills,
            "backfill_ledger": copy.deepcopy(state.backfills),
            "stage_durations_ms": {
                key: round(value, 3) for key, value in sorted(stage_times.items())
            },
            "injected_items": tracker.report(ordered_failures),
            "failures": ordered_failures,
            "recovery_point": recovery,
            "stopped_gracefully": bool(stopped and state.completed_stages),
            "stop_reason": (
                "signal" if stop.requested else ("stop_after:" + stop_after if stop_after else None)
            ),
        }

    def _fatal_report(
        self, recovery: dict[str, Any], failures: list[dict[str, Any]], stage_times: dict[str, float]
    ) -> dict[str, Any]:
        return {
            "report_version": 1,
            "global_version": 0,
            "snapshot_versions": {},
            "completed_stages": [],
            "definition_hash": None,
            "event_cursor": -1,
            "event_count": 0,
            "feature_count": 0,
            "sample_count": 0,
            "empty_window_feature_count": 0,
            "feature_snapshot_version": None,
            "sample_snapshot_version": None,
            "invalidation_ledger": {
                "invalidated_features": [],
                "reused_features": [],
                "computed_features": [],
                "reasons": [],
                "invalidated_samples": [],
                "regenerated_samples": [],
                "reused_samples": [],
            },
            "sample_regeneration_list": [],
            "backfill_impacts": [],
            "backfill_ledger": [],
            "stage_durations_ms": stage_times,
            "injected_items": [],
            "failures": sorted_failures(failures),
            "recovery_point": recovery,
            "stopped_gracefully": False,
            "stop_reason": None,
        }

    @staticmethod
    def _should_stop(
        stop: StopController,
        config: PipelineConfig,
        stop_after: str | None,
        stage: str,
    ) -> bool:
        if config.stage_delay_ms > 0:
            deadline = time.monotonic() + config.stage_delay_ms / 1000.0
            while time.monotonic() < deadline and not stop.requested:
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        return stop.requested or stop_after == stage


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
