from __future__ import annotations

import copy
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Any, Iterable

from .canonical import canonical_json, decimal_text, fingerprint
from .fixedpoint import (
    convert_fixed,
    decimal_to_fixed,
    fixed_decimal,
    fixed_text,
    round_fixed_to_quantum,
)
from .models import (
    Event,
    FeatureDef,
    FeatureValue,
    PendingPoint,
    ScoreConfig,
    ScoreContribution,
    ScoreValue,
    failure,
    failure_sort_key,
)
from .validation import (
    FaultPlan,
    RegistryBuildResult,
    build_registry,
    parse_feature_definition,
    validate_event_payload,
)


class InjectedFault(RuntimeError):
    pass


@dataclass(frozen=True)
class Registry:
    definitions: dict[str, FeatureDef]
    topo: tuple[str, ...]
    downstream: dict[str, tuple[str, ...]]

    @classmethod
    def from_build(cls, result: RegistryBuildResult) -> "Registry":
        return cls(result.definitions, tuple(result.topo), result.downstream)

    @classmethod
    def from_definitions(cls, definitions: dict[str, FeatureDef]) -> "Registry":
        result = build_registry(definitions)
        if result.failures:
            raise ValueError("invalid feature dependency graph")
        return cls.from_build(result)

    def closure(self, names: Iterable[str]) -> set[str]:
        found: set[str] = set()
        stack = list(names)
        while stack:
            name = stack.pop()
            if name in found or name not in self.definitions:
                continue
            found.add(name)
            stack.extend(self.downstream.get(name, ()))
        return found


class EventStore:
    def __init__(self, events: Iterable[Event]):
        self.events: dict[str, Event] = {}
        self.order: list[Event] = []
        for event in events:
            self.events[event.event_id] = event
            self.order.append(event)
        self.order.sort(key=lambda item: (item.timestamp_ms, item.event_id))
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        self.side_entities: dict[str, set[str]] = {"user": set(), "item": set()}
        self.side_events: dict[str, dict[str, list[Event]]] = {"user": {}, "item": {}}
        self.side_timestamps: dict[str, dict[str, list[int]]] = {"user": {}, "item": {}}
        for event in self.order:
            self.side_entities["user"].add(event.user_id)
            self.side_entities["item"].add(event.item_id)
            self.side_events["user"].setdefault(event.user_id, []).append(event)
            self.side_events["item"].setdefault(event.item_id, []).append(event)
        for side in ("user", "item"):
            for entity_id, events in self.side_events[side].items():
                if side == "user":
                    ids = entity_id
                else:
                    ids = entity_id
                self.side_timestamps[side][ids] = [event.timestamp_ms for event in events]
        self.event_order = {event.event_id: index for index, event in enumerate(self.order)}

    @property
    def cursor(self) -> int:
        return self.order[-1].timestamp_ms if self.order else 0

    def entities(self, side: str) -> tuple[str, ...]:
        return tuple(sorted(self.side_entities[side]))

    def append(self, event: Event) -> None:
        if event.event_id in self.events:
            raise ValueError("duplicate event id")
        self.events[event.event_id] = event
        self.order.append(event)
        self.order.sort(key=lambda item: (item.timestamp_ms, item.event_id))
        self._rebuild_indexes()

    def replace(self, event: Event) -> None:
        if event.event_id not in self.events:
            raise KeyError(event.event_id)
        self.events[event.event_id] = event
        self.order = [event if item.event_id == event.event_id else item for item in self.order]
        self.order.sort(key=lambda item: (item.timestamp_ms, item.event_id))
        self._rebuild_indexes()

    def window_events(self, side: str, entity_id: str, cursor: int, window_ms: int) -> tuple[Event, ...]:
        events = self.side_events[side].get(entity_id)
        if not events:
            return ()
        timestamps = self.side_timestamps[side][entity_id]
        left = bisect_left(timestamps, cursor - window_ms)
        right = bisect_right(timestamps, cursor)
        return tuple(events[left:right])

    def ordered_ids(self, event_ids: Iterable[str]) -> tuple[str, ...]:
        return tuple(sorted(set(event_ids), key=lambda item: self.event_order.get(item, 10**30)))


@dataclass(frozen=True)
class PreparedFeature:
    definition: FeatureDef
    side: str
    entity_id: str
    window_events: tuple[Event, ...]
    upstream_values: tuple[FeatureValue, ...]
    event_ids: tuple[str, ...]
    event_window_fingerprint: str
    upstream_fingerprints: tuple[tuple[str, str], ...]
    cache_key: str
    cost: int

    @property
    def unit(self) -> tuple[str, str, str]:
        return (self.definition.name, self.side, self.entity_id)

    @property
    def stable_key(self) -> tuple[str, str, str, str]:
        first_event = self.event_ids[0] if self.event_ids else ""
        return (self.definition.name, first_event, self.side, self.entity_id)


@dataclass(frozen=True)
class PreparedScore:
    side: str
    entity_id: str
    values: tuple[FeatureValue, ...]
    cache_key: str
    cost: int = 1

    @property
    def stable_key(self) -> tuple[str, str]:
        return (self.side, self.entity_id)


@dataclass
class RunStats:
    recomputed_features: list[str]
    reused_features: list[str]
    invalidated_features: list[str]
    recomputed_scores: list[str]
    reused_scores: list[str]
    pending_features: list[PendingPoint]
    pending_scores: list[PendingPoint]

    def as_dict(self) -> dict[str, Any]:
        return {
            "recomputed_feature_count": len(self.recomputed_features),
            "reused_feature_count": len(self.reused_features),
            "recomputed_score_count": len(self.recomputed_scores),
            "reused_score_count": len(self.reused_scores),
            "pending_feature_count": len(self.pending_features),
            "pending_score_count": len(self.pending_scores),
            "recomputed_features": sorted(set(self.recomputed_features)),
            "invalidated_features": sorted(set(self.invalidated_features)),
        }


def _runtime_key(item: dict[str, Any]) -> str:
    return canonical_json(
        [
            item.get("stage"),
            item.get("feature_name"),
            item.get("entity_type"),
            item.get("entity_id"),
            item.get("code"),
            item.get("injection_id"),
        ]
    )


class Workbench:
    def __init__(
        self,
        *,
        config: ScoreConfig,
        definitions: dict[str, FeatureDef],
        topo: list[str],
        downstream: dict[str, tuple[str, ...]],
        events: Iterable[Event],
        load_failures: list[dict[str, Any]],
        faults: FaultPlan | None = None,
    ):
        self.config = config
        self.registry = Registry(definitions, tuple(topo), downstream)
        self.events = EventStore(events)
        self.faults = faults or FaultPlan({}, {})
        self.load_failures = list(load_failures)
        self.feature_values: dict[tuple[str, str, str], FeatureValue] = {}
        self.feature_cache: dict[str, FeatureValue] = {}
        self.score_values: dict[tuple[str, str], ScoreValue] = {}
        self.score_cache: dict[str, ScoreValue] = {}
        self.runtime_failures: dict[str, dict[str, Any]] = {}
        self.operation_failures: list[dict[str, Any]] = []
        self.batch_summaries: list[dict[str, Any]] = []
        self.pending_features: set[tuple[str, str, str]] = set()
        self.pending_scores: set[tuple[str, str]] = set()
        self.pending_feature_details: dict[tuple[str, str, str], tuple[str, ...]] = {}
        self.pending_score_details: dict[tuple[str, str], tuple[str, ...]] = {}
        self.run_count = 0

    @property
    def cursor(self) -> int:
        return self.events.cursor

    def _set_runtime_failure(self, item: dict[str, Any]) -> None:
        self.runtime_failures[_runtime_key(item)] = item

    def _clear_runtime_failures_for_unit(self, feature: str, side: str, entity_id: str) -> None:
        for key, item in list(self.runtime_failures.items()):
            if (
                item.get("stage") in {"feature", "feature_upstream"}
                and item.get("feature_name") == feature
                and item.get("entity_type") == side
                and item.get("entity_id") == entity_id
            ):
                self.runtime_failures.pop(key, None)

    def _clear_runtime_failures_for_score(self, side: str, entity_id: str) -> None:
        for key, item in list(self.runtime_failures.items()):
            if (
                item.get("stage") == "scoring"
                and item.get("entity_type") == side
                and item.get("entity_id") == entity_id
            ):
                self.runtime_failures.pop(key, None)

    def failures(self) -> list[dict[str, Any]]:
        all_failures = list(self.load_failures) + list(self.runtime_failures.values())
        unique = {canonical_json(item): item for item in all_failures}
        return sorted(unique.values(), key=failure_sort_key)

    def all_units(self) -> set[tuple[str, str, str]]:
        units: set[tuple[str, str, str]] = set()
        for name in self.registry.topo:
            definition = self.registry.definitions[name]
            for side in definition.sides:
                for entity_id in self.events.entities(side):
                    units.add((name, side, entity_id))
        return units

    def _prepare_feature(
        self, definition: FeatureDef, side: str, entity_id: str
    ) -> PreparedFeature | None:
        window = self.events.window_events(side, entity_id, self.cursor, definition.window_ms)
        upstream_values: list[FeatureValue] = []
        for upstream_name in definition.upstream_features:
            record = self.feature_values.get((upstream_name, side, entity_id))
            if record is None or record.stale:
                return None
            upstream_values.append(record)
        window_fp = fingerprint(
            "feature-event-window-v1",
            {
                "side": side,
                "entity_id": entity_id,
                "cursor_ms": self.cursor,
                "window_ms": definition.window_ms,
                "events": [event.content_fingerprint for event in window],
            },
        )
        upstream_fps = tuple(
            (record.feature_name, record.content_fingerprint)
            for record in sorted(upstream_values, key=lambda item: item.feature_name)
        )
        cache_key = fingerprint(
            "feature-cache-key-v1",
            {
                "parameter_fingerprint": definition.parameter_fingerprint,
                "event_window_fingerprint": window_fp,
                "upstream_fingerprints": [list(item) for item in upstream_fps],
            },
        )
        participating: list[str] = []
        if definition.aggregate == "count":
            participating.extend(event.event_id for event in window)
        elif definition.source_field is not None:
            if definition.aggregate == "latest":
                for event in reversed(window):
                    if definition.source_field in event.context:
                        participating.append(event.event_id)
                        break
            else:
                participating.extend(
                    event.event_id
                    for event in window
                    if definition.source_field in event.context
                )
        for record in upstream_values:
            participating.extend(record.event_ids)
        event_ids = self.events.ordered_ids(participating)
        return PreparedFeature(
            definition=definition,
            side=side,
            entity_id=entity_id,
            window_events=window,
            upstream_values=tuple(upstream_values),
            event_ids=event_ids,
            event_window_fingerprint=window_fp,
            upstream_fingerprints=upstream_fps,
            cache_key=cache_key,
            cost=max(1, len(event_ids)),
        )

    def _compute_feature(self, prepared: PreparedFeature) -> FeatureValue:
        fault = self.faults.feature_faults.get(prepared.definition.name)
        if fault:
            injection_id, exception_type = fault
            raise InjectedFault(f"{exception_type}:{injection_id}")
        definition = prepared.definition
        with localcontext() as context:
            context.prec = 80
            values: list[Decimal] = []
            if definition.aggregate == "count":
                base = Decimal(len(prepared.window_events))
            else:
                field = definition.source_field
                if field is not None:
                    for event in prepared.window_events:
                        if field in event.context:
                            values.append(Decimal(str(event.context[field])))
                if definition.aggregate == "sum":
                    base = sum(values, Decimal(0))
                elif definition.aggregate == "mean":
                    base = (
                        sum(values, Decimal(0)) / Decimal(len(values))
                        if values
                        else Decimal(0)
                    )
                elif definition.aggregate == "latest":
                    base = values[-1] if values else Decimal(0)
                else:
                    raise ValueError(f"unsupported aggregate: {definition.aggregate}")
            result_fixed = decimal_to_fixed(base, definition.decimal_places, definition.quantum)
            for upstream in prepared.upstream_values:
                result_fixed += convert_fixed(
                    upstream.value_fixed, upstream.decimal_places, definition.decimal_places
                )
        result_fixed = round_fixed_to_quantum(
            result_fixed, definition.decimal_places, definition.quantum
        )
        content_fingerprint = fingerprint(
            "feature-value-v1",
            {"cache_key": prepared.cache_key, "value_fixed": result_fixed},
        )
        return FeatureValue(
            feature_name=definition.name,
            side=prepared.side,
            entity_id=prepared.entity_id,
            value_fixed=result_fixed,
            decimal_places=definition.decimal_places,
            quantization_step=decimal_text(definition.quantum),
            event_ids=prepared.event_ids,
            event_window_fingerprint=prepared.event_window_fingerprint,
            parameter_fingerprint=definition.parameter_fingerprint,
            upstream_fingerprints=prepared.upstream_fingerprints,
            cache_key=prepared.cache_key,
            content_fingerprint=content_fingerprint,
            stale=False,
        )

    def _mark_feature_stale(self, unit: tuple[str, str, str]) -> None:
        current = self.feature_values.get(unit)
        if current is not None and not current.stale:
            self.feature_values[unit] = FeatureValue(
                **{**current.__dict__, "stale": True}
            )

    def _mark_score_stale(self, unit: tuple[str, str]) -> None:
        current = self.score_values.get(unit)
        if current is not None and not current.stale:
            self.score_values[unit] = ScoreValue(**{**current.__dict__, "stale": True})

    def _feature_pending_point(
        self, unit: tuple[str, str, str], event_ids: tuple[str, ...]
    ) -> PendingPoint:
        return PendingPoint(unit[0], unit[1], unit[2], event_ids, stage="feature")

    def _score_pending_point(self, unit: tuple[str, str], event_ids: tuple[str, ...]) -> PendingPoint:
        return PendingPoint("*", unit[0], unit[1], event_ids, stage="score")

    def run_incremental(
        self,
        *,
        force_all: bool = False,
        forced_units: Iterable[tuple[str, str, str]] | None = None,
    ) -> RunStats:
        self.run_count += 1
        forced = set(forced_units or ())
        if force_all or self.run_count == 1:
            candidates = self.all_units()
        else:
            candidates = forced | set(self.pending_features)
            for unit in self.all_units():
                if unit not in self.feature_values or self.feature_values[unit].stale:
                    candidates.add(unit)
        candidates = {unit for unit in candidates if unit[0] in self.registry.definitions}

        recomputed_features: list[str] = []
        all_units = self.all_units()
        reused_features: list[str] = [
            f"{unit[0]}|{unit[1]}|{unit[2]}"
            for unit in sorted(all_units - candidates)
            if unit in self.feature_values and not self.feature_values[unit].stale
        ]
        recomputed_scores: list[str] = []
        reused_scores: list[str] = []
        pending_feature_points: list[PendingPoint] = []
        pending_score_points: list[PendingPoint] = []
        next_pending_features: set[tuple[str, str, str]] = set()
        next_pending_feature_details: dict[tuple[str, str, str], tuple[str, ...]] = {}
        next_pending_scores: set[tuple[str, str]] = set()
        next_pending_score_details: dict[tuple[str, str], tuple[str, ...]] = {}
        invalidated_features = {unit[0] for unit in candidates}
        remaining_budget = self.config.point_budget_per_batch
        budget_exhausted = False

        executor = ThreadPoolExecutor(max_workers=self.config.concurrency)
        try:
            for feature_name in self.registry.topo:
                definition = self.registry.definitions[feature_name]
                units = [
                    (feature_name, side, entity_id)
                    for side in definition.sides
                    for entity_id in self.events.entities(side)
                    if (feature_name, side, entity_id) in candidates
                ]
                prepared_items: list[PreparedFeature] = []
                for unit in units:
                    _, side, entity_id = unit
                    missing_upstream = next(
                        (
                            upstream
                            for upstream in definition.upstream_features
                            if (upstream, side, entity_id) not in self.feature_values
                            or self.feature_values[(upstream, side, entity_id)].stale
                        ),
                        None,
                    )
                    if missing_upstream is not None:
                        upstream_unit = (missing_upstream, side, entity_id)
                        if upstream_unit in self.pending_features:
                            next_pending_features.add(unit)
                            detail = self.pending_feature_details.get(upstream_unit, ())
                            next_pending_feature_details[unit] = detail
                            pending_feature_points.append(self._feature_pending_point(unit, detail))
                            self._mark_feature_stale(unit)
                        else:
                            self._mark_feature_stale(unit)
                            self._set_runtime_failure(
                                failure(
                                    code="E_UPSTREAM_UNAVAILABLE",
                                    stage="feature_upstream",
                                    message=f"upstream feature {missing_upstream} has no reusable value",
                                    feature_name=feature_name,
                                    entity_type=side,
                                    entity_id=entity_id,
                                )
                            )
                        continue
                    prepared = self._prepare_feature(definition, side, entity_id)
                    if prepared is None:
                        self._mark_feature_stale(unit)
                        self._set_runtime_failure(
                            failure(
                                code="E_UPSTREAM_UNAVAILABLE",
                                stage="feature_upstream",
                                message="an upstream feature is stale or unavailable",
                                feature_name=feature_name,
                                entity_type=side,
                                entity_id=entity_id,
                            )
                        )
                        continue
                    cached = self.feature_cache.get(prepared.cache_key)
                    if cached is not None:
                        reused = FeatureValue(**{**cached.__dict__, "stale": False})
                        self.feature_values[unit] = reused
                        reused_features.append(f"{feature_name}|{side}|{entity_id}")
                        self._clear_runtime_failures_for_unit(feature_name, side, entity_id)
                        continue
                    prepared_items.append(prepared)

                prepared_items.sort(key=lambda item: item.stable_key)
                accepted: list[PreparedFeature] = []
                for prepared in prepared_items:
                    if budget_exhausted or prepared.cost > remaining_budget:
                        budget_exhausted = True
                        unit = prepared.unit
                        next_pending_features.add(unit)
                        next_pending_feature_details[unit] = prepared.event_ids
                        pending_feature_points.append(
                            self._feature_pending_point(unit, prepared.event_ids)
                        )
                        self._mark_feature_stale(unit)
                        continue
                    remaining_budget -= prepared.cost
                    accepted.append(prepared)

                futures = [
                    (prepared, executor.submit(self._compute_feature, prepared))
                    for prepared in accepted
                ]
                for prepared, future in futures:
                    unit = prepared.unit
                    try:
                        record = future.result()
                    except Exception as exc:  # noqa: BLE001 - diagnostic isolation is intentional
                        self._mark_feature_stale(unit)
                        injection_id = None
                        fault = self.faults.feature_faults.get(prepared.definition.name)
                        if fault:
                            injection_id = fault[0]
                        self._set_runtime_failure(
                            failure(
                                code="E_FEATURE_COMPUTE",
                                stage="feature",
                                message=str(exc) or "feature calculation failed",
                                feature_name=prepared.definition.name,
                                entity_type=prepared.side,
                                entity_id=prepared.entity_id,
                                exception_type=type(exc).__name__,
                                injection_id=injection_id,
                            )
                        )
                        continue
                    self.feature_values[unit] = record
                    self.feature_cache[record.cache_key] = record
                    self._clear_runtime_failures_for_unit(*unit)
                    recomputed_features.append(f"{unit[0]}|{unit[1]}|{unit[2]}")
        finally:
            executor.shutdown(wait=True)

        self.pending_features = next_pending_features
        self.pending_feature_details = next_pending_feature_details

        for (feature_name, side, entity_id), record in list(self.feature_values.items()):
            if feature_name not in self.registry.definitions:
                self.feature_values.pop((feature_name, side, entity_id), None)

        score_units = [
            (side, entity_id)
            for side in ("item", "user")
            for entity_id in self.events.entities(side)
        ]
        prepared_scores: list[PreparedScore] = []
        for side, entity_id in score_units:
            definition_names = sorted(
                name
                for name in self.registry.topo
                if side in self.registry.definitions[name].sides
            )
            values: list[FeatureValue] = []
            unavailable_feature: str | None = None
            missing_event_ids: list[str] = []
            for name in definition_names:
                record = self.feature_values.get((name, side, entity_id))
                if record is None or record.stale:
                    unavailable_feature = name
                    if record is not None:
                        missing_event_ids.extend(record.event_ids)
                    break
                values.append(record)
            score_unit = (side, entity_id)
            if unavailable_feature is not None:
                if (unavailable_feature, side, entity_id) in self.pending_features:
                    detail = self.pending_feature_details.get(
                        (unavailable_feature, side, entity_id), ()
                    )
                    next_pending_scores.add(score_unit)
                    next_pending_score_details[score_unit] = detail
                    pending_score_points.append(
                        self._score_pending_point(score_unit, detail)
                    )
                    self._mark_score_stale(score_unit)
                else:
                    self._mark_score_stale(score_unit)
                    self._set_runtime_failure(
                        failure(
                            code="E_SCORE_UNAVAILABLE",
                            stage="scoring",
                            message=f"feature {unavailable_feature} is unavailable for scoring",
                            feature_name=unavailable_feature,
                            entity_type=side,
                            entity_id=entity_id,
                        )
                    )
                continue
            feature_fingerprints = tuple(
                (record.feature_name, record.content_fingerprint) for record in values
            )
            contributions_payload = [
                [record.feature_name, record.content_fingerprint] for record in values
            ]
            cache_key = fingerprint(
                "score-cache-key-v1",
                {
                    "scoring_fingerprint": self.config.scoring_fingerprint,
                    "feature_fingerprints": contributions_payload,
                },
            )
            cached_score = self.score_cache.get(cache_key)
            if cached_score is not None:
                reused_score = ScoreValue(**{**cached_score.__dict__, "stale": False})
                self.score_values[score_unit] = reused_score
                reused_scores.append(f"{side}|{entity_id}")
                self._clear_runtime_failures_for_score(side, entity_id)
                continue
            if budget_exhausted or remaining_budget < 1:
                budget_exhausted = True
                next_pending_scores.add(score_unit)
                all_event_ids = self.events.ordered_ids(
                    event_id for record in values for event_id in record.event_ids
                )
                next_pending_score_details[score_unit] = all_event_ids
                pending_score_points.append(self._score_pending_point(score_unit, all_event_ids))
                self._mark_score_stale(score_unit)
                continue
            remaining_budget -= 1
            prepared_scores.append(
                PreparedScore(side, entity_id, tuple(values), cache_key, cost=1)
            )

        score_executor = ThreadPoolExecutor(max_workers=self.config.concurrency)
        try:
            score_futures = [
                (prepared, score_executor.submit(self._compute_score, prepared))
                for prepared in sorted(prepared_scores, key=lambda item: item.stable_key)
            ]
            for prepared, future in score_futures:
                score_unit = (prepared.side, prepared.entity_id)
                try:
                    score = future.result()
                except Exception as exc:  # noqa: BLE001 - diagnostic isolation is intentional
                    self._mark_score_stale(score_unit)
                    fault = self.faults.score_faults.get(score_unit) or self.faults.any_score_fault.get("*")
                    injection_id = fault[0] if fault else None
                    target_feature = fault[2] if fault and len(fault) > 2 else None
                    if target_feature is None and prepared.values:
                        target_feature = prepared.values[0].feature_name
                    self._set_runtime_failure(
                        failure(
                            code="E_SCORE_COMPUTE",
                            stage="scoring",
                            message=str(exc) or "scoring failed",
                            feature_name=target_feature,
                            entity_type=prepared.side,
                            entity_id=prepared.entity_id,
                            exception_type=type(exc).__name__,
                            injection_id=injection_id,
                        )
                    )
                    continue
                self.score_values[score_unit] = score
                self.score_cache[score.cache_key] = score
                self._clear_runtime_failures_for_score(*score_unit)
                recomputed_scores.append(f"{prepared.side}|{prepared.entity_id}")
        finally:
            score_executor.shutdown(wait=True)

        self.pending_scores = next_pending_scores
        self.pending_score_details = next_pending_score_details
        return RunStats(
            recomputed_features=recomputed_features,
            reused_features=reused_features,
            invalidated_features=sorted(invalidated_features),
            recomputed_scores=recomputed_scores,
            reused_scores=reused_scores,
            pending_features=pending_feature_points,
            pending_scores=pending_score_points,
        )

    def _compute_score(self, prepared: PreparedScore) -> ScoreValue:
        score_unit = (prepared.side, prepared.entity_id)
        fault = self.faults.score_faults.get(score_unit) or self.faults.any_score_fault.get("*")
        if fault:
            injection_id, exception_type = fault[0], fault[1]
            raise InjectedFault(f"{exception_type}:{injection_id}")
        contributions: list[ScoreContribution] = []
        total = Decimal(0)
        with localcontext() as context:
            context.prec = 80
            for record in sorted(prepared.values, key=lambda item: item.feature_name):
                weight = self.config.weights.get(record.feature_name, Decimal(0))
                contribution = record.value_decimal * weight / self.config.fusion_temperature
                total += contribution
                contribution_fixed = decimal_to_fixed(
                    contribution,
                    self.config.score_decimal_places,
                    self.config.score_quantum,
                )
                contributions.append(
                    ScoreContribution(
                        feature_name=record.feature_name,
                        feature_value_fixed=record.value_fixed,
                        feature_decimal_places=record.decimal_places,
                        feature_value_decimal=decimal_text(record.value_decimal),
                        feature_content_fingerprint=record.content_fingerprint,
                        weight=decimal_text(weight),
                        contribution_decimal=decimal_text(contribution),
                        contribution_fixed=contribution_fixed,
                    )
                )
            fused = self.config.fusion_bias + total
            clamped = min(max(fused, self.config.clamp_min), self.config.clamp_max)
            score_fixed = decimal_to_fixed(
                clamped,
                self.config.score_decimal_places,
                self.config.score_quantum,
            )
        event_ids = self.events.ordered_ids(
            event_id for record in prepared.values for event_id in record.event_ids
        )
        feature_fingerprints = tuple(
            (record.feature_name, record.content_fingerprint) for record in prepared.values
        )
        content_fingerprint = fingerprint(
            "score-value-v1",
            {"cache_key": prepared.cache_key, "score_fixed": score_fixed},
        )
        return ScoreValue(
            entity_type=prepared.side,
            entity_id=prepared.entity_id,
            score_fixed=score_fixed,
            score_decimal_places=self.config.score_decimal_places,
            score_quantum=decimal_text(self.config.score_quantum),
            score_decimal=fixed_text(score_fixed, self.config.score_decimal_places),
            bias_contribution=decimal_text(self.config.fusion_bias),
            contributions=tuple(contributions),
            participating_event_count=len(event_ids),
            feature_fingerprints=feature_fingerprints,
            cache_key=prepared.cache_key,
            content_fingerprint=content_fingerprint,
            stale=False,
        )

    def ranked_scores(self) -> list[ScoreValue]:
        values = [
            value
            for value in self.score_values.values()
            if value.entity_type in {"user", "item"}
        ]
        return sorted(
            values,
            key=lambda item: (
                -item.score_fixed,
                -item.participating_event_count,
                item.entity_id,
                item.entity_type,
            ),
        )

    def features_public(self) -> list[dict[str, Any]]:
        return [
            value.public()
            for _, value in sorted(
                self.feature_values.items(),
                key=lambda item: (
                    item[1].feature_name,
                    item[1].side,
                    item[1].entity_id,
                ),
            )
        ]

    def scores_public(self) -> list[dict[str, Any]]:
        return [
            value.public(rank=index + 1)
            for index, value in enumerate(self.ranked_scores())
        ]

    def pending_public(self, stage: str | None = None) -> list[dict[str, Any]]:
        points = []
        if stage in {None, "feature"}:
            points.extend(
                self._feature_pending_point(unit, self.pending_feature_details.get(unit, ()))
                for unit in self.pending_features
            )
        if stage in {None, "score"}:
            points.extend(
                self._score_pending_point(unit, self.pending_score_details.get(unit, ()))
                for unit in self.pending_scores
            )
        return [
            item.public()
            for item in sorted(
                points,
                key=lambda point: (
                    point.stage,
                    point.feature_name,
                    point.event_ids[0] if point.event_ids else "",
                    point.side,
                    point.entity_id,
                ),
            )
        ]

    def projection(self) -> dict[str, Any]:
        return {
            "features": self.features_public(),
            "scores": self.scores_public(),
            "failures": self.failures(),
            "pending": self.pending_public(),
        }

    def apply_batch(self, batch: Any, batch_index: int) -> tuple[bool, set[tuple[str, str, str]]]:
        failures: list[dict[str, Any]] = []
        force_all = False
        forced_units: set[tuple[str, str, str]] = set()
        if not isinstance(batch, dict):
            self.operation_failures.append(
                failure(
                    code="E_EDIT_PLAN_INVALID",
                    stage="plan_apply",
                    message="batch must be an object",
                    detail={"batch_index": batch_index},
                )
            )
            return False, forced_units

        appends = batch.get("appends", [])
        if appends is None:
            appends = []
        if not isinstance(appends, list):
            failures.append(
                failure(
                    code="E_EDIT_PLAN_INVALID",
                    stage="plan_apply",
                    message="appends must be a list",
                    detail={"batch_index": batch_index},
                )
            )
            appends = []
        for raw in appends:
            event, error = validate_event_payload(
                raw,
                existing_ids=set(self.events.events),
                previous_timestamp=self.cursor,
                allowed_behaviors=self.config.allowed_behaviors,
            )
            if error:
                error["stage"] = "plan_apply"
                failures.append(error)
                continue
            assert event is not None
            self.events.append(event)
            force_all = True

        backfills = batch.get("backfills", [])
        if backfills is None:
            backfills = []
        if not isinstance(backfills, list):
            failures.append(
                failure(
                    code="E_EDIT_PLAN_INVALID",
                    stage="plan_apply",
                    message="backfills must be a list",
                    detail={"batch_index": batch_index},
                )
            )
            backfills = []
        for item in backfills:
            if not isinstance(item, dict):
                failures.append(
                    failure(
                        code="E_EDIT_PLAN_INVALID",
                        stage="plan_apply",
                        message="backfill item must be an object",
                        detail={"batch_index": batch_index},
                    )
                )
                continue
            injection_id = item.get("_injection_id", item.get("injection_id"))
            if "late_event" in item:
                late_raw = item["late_event"]
                if isinstance(late_raw, dict) and injection_id and "_injection_id" not in late_raw:
                    late_raw = dict(late_raw)
                    late_raw["_injection_id"] = injection_id
                if isinstance(late_raw, dict) and isinstance(late_raw.get("timestamp_ms"), int):
                    if late_raw["timestamp_ms"] <= self.cursor:
                        failures.append(
                            failure(
                                code="E_EVENT_INVALID",
                                stage="plan_apply",
                                message="backfill event timestamp is older than or equal to the completed event cursor",
                                event_id=str(late_raw.get("event_id")) if late_raw.get("event_id") else None,
                                injection_id=str(injection_id) if injection_id else None,
                            )
                        )
                        continue
                event, error = validate_event_payload(
                    late_raw,
                    existing_ids=set(self.events.events),
                    previous_timestamp=self.cursor,
                    allowed_behaviors=self.config.allowed_behaviors,
                )
                if error:
                    error["stage"] = "plan_apply"
                    failures.append(error)
                    continue
                assert event is not None
                self.events.append(event)
                force_all = True
                continue

            event_id = item.get("event_id")
            old = self.events.events.get(event_id) if isinstance(event_id, str) else None
            if old is None:
                failures.append(
                    failure(
                        code="E_EVENT_INVALID",
                        stage="plan_apply",
                        message="backfill refers to an unknown event_id",
                        event_id=str(event_id) if event_id is not None else None,
                        injection_id=str(injection_id) if injection_id else None,
                    )
                )
                continue
            replacement = item.get("replacement")
            if replacement is None:
                replacement = old.public(include_injection=True)
                patch = item.get("patch", {})
                if not isinstance(patch, dict):
                    failures.append(
                        failure(
                            code="E_EDIT_PLAN_INVALID",
                            stage="plan_apply",
                            message="backfill patch must be an object",
                            event_id=old.event_id,
                            injection_id=str(injection_id) if injection_id else None,
                        )
                    )
                    continue
                replacement = copy.deepcopy(replacement)
                for key, value in patch.items():
                    if key == "context" and isinstance(value, dict):
                        replacement.setdefault("context", {}).update(value)
                    else:
                        replacement[key] = value
            if not isinstance(replacement, dict):
                failures.append(
                    failure(
                        code="E_EVENT_INVALID",
                        stage="plan_apply",
                        message="backfill replacement must be an object",
                        event_id=old.event_id,
                        injection_id=str(injection_id) if injection_id else None,
                    )
                )
                continue
            replacement = dict(replacement)
            replacement.setdefault("event_id", old.event_id)
            if injection_id and "_injection_id" not in replacement:
                replacement["_injection_id"] = injection_id
            identity_changed = (
                replacement.get("event_id") != old.event_id
                or replacement.get("user_id") != old.user_id
                or replacement.get("item_id") != old.item_id
                or replacement.get("timestamp_ms") != old.timestamp_ms
            )
            if identity_changed:
                timestamp_changed = replacement.get("timestamp_ms") != old.timestamp_ms
                message = (
                    "backfill event timestamp is older than the completed event cursor"
                    if timestamp_changed
                    and isinstance(replacement.get("timestamp_ms"), int)
                    and replacement["timestamp_ms"] < self.cursor
                    else "backfill may not change event identity or timestamp"
                )
                failures.append(
                    failure(
                        code="E_EVENT_INVALID",
                        stage="plan_apply",
                        message=message,
                        event_id=old.event_id,
                        injection_id=str(injection_id) if injection_id else None,
                    )
                )
                continue
            new_event, error = validate_event_payload(
                replacement,
                existing_ids=set(self.events.events) - {old.event_id},
                previous_timestamp=None,
                allowed_behaviors=self.config.allowed_behaviors,
                require_strictly_after_previous=False,
            )
            if error:
                error["stage"] = "plan_apply"
                failures.append(error)
                continue
            assert new_event is not None
            if new_event.content_fingerprint == old.content_fingerprint:
                continue
            self.events.replace(new_event)
            forced_units |= self._event_affected_units(new_event)

        definition_changes = batch.get("definition_changes", [])
        if definition_changes is None:
            definition_changes = []
        if not isinstance(definition_changes, list):
            failures.append(
                failure(
                    code="E_EDIT_PLAN_INVALID",
                    stage="plan_apply",
                    message="definition_changes must be a list",
                    detail={"batch_index": batch_index},
                )
            )
            definition_changes = []
        if definition_changes:
            applied, changed_names = self._apply_definition_changes(definition_changes, failures)
            if applied:
                closure = self.registry.closure(changed_names)
                for name in closure:
                    definition = self.registry.definitions[name]
                    for side in definition.sides:
                        for entity_id in self.events.entities(side):
                            forced_units.add((name, side, entity_id))

        self.operation_failures.extend(failures)
        return force_all, forced_units

    def _event_affected_units(self, event: Event) -> set[tuple[str, str, str]]:
        roots: set[str] = set()
        units: set[tuple[str, str, str]] = set()
        for name, definition in self.registry.definitions.items():
            if name not in self.registry.topo:
                continue
            matching: list[tuple[str, str]] = []
            if "user" in definition.sides:
                matching.append(("user", event.user_id))
            if "item" in definition.sides:
                matching.append(("item", event.item_id))
            for side, entity_id in matching:
                left = self.cursor - definition.window_ms
                if left <= event.timestamp_ms <= self.cursor:
                    roots.add(name)
                    units.add((name, side, entity_id))
        closure = self.registry.closure(roots)
        for name in closure:
            definition = self.registry.definitions.get(name)
            if definition is None:
                continue
            if "user" in definition.sides:
                units.add((name, "user", event.user_id))
            if "item" in definition.sides:
                units.add((name, "item", event.item_id))
        return units

    def _apply_definition_changes(
        self, changes: list[Any], failures: list[dict[str, Any]]
    ) -> tuple[bool, set[str]]:
        candidate_raw = {
            name: copy.deepcopy(definition.raw)
            for name, definition in self.registry.definitions.items()
        }
        parsed_changes: list[tuple[str, FeatureDef | None, str | None]] = []
        for change in changes:
            if not isinstance(change, dict):
                failures.append(
                    failure(
                        code="E_EDIT_PLAN_INVALID",
                        stage="plan_apply",
                        message="definition change must be an object",
                    )
                )
                return False, set()
            op = change.get("op", "upsert")
            injection_id = change.get("_injection_id", change.get("injection_id"))
            if op == "upsert":
                raw_definition = change.get("definition")
                if not isinstance(raw_definition, dict) and isinstance(change.get("name"), str):
                    raw_definition = {key: value for key, value in change.items() if key not in {"op", "definition"}}
                definition, error = parse_feature_definition(raw_definition)
                if error:
                    if injection_id:
                        error["injection_id"] = str(injection_id)
                    error["stage"] = "plan_apply"
                    failures.append(error)
                    return False, set()
                assert definition is not None
                parsed_changes.append((definition.name, definition, None))
                candidate_raw[definition.name] = copy.deepcopy(definition.raw)
            elif op == "remove":
                name = change.get("name")
                if not isinstance(name, str) or name not in candidate_raw:
                    failures.append(
                        failure(
                            code="E_DEFINITION_INVALID",
                            stage="plan_apply",
                            message="remove refers to an unknown feature name",
                            feature_name=str(name) if name is not None else None,
                            injection_id=str(injection_id) if injection_id else None,
                        )
                    )
                    return False, set()
                parsed_changes.append((name, None, name))
                candidate_raw.pop(name, None)
            else:
                failures.append(
                    failure(
                        code="E_DEFINITION_INVALID",
                        stage="plan_apply",
                        message="definition change op must be upsert or remove",
                        injection_id=str(injection_id) if injection_id else None,
                    )
                )
                return False, set()

        candidate_definitions: dict[str, FeatureDef] = {}
        for name, raw in candidate_raw.items():
            definition, error = parse_feature_definition(raw)
            if error or definition is None:
                if error:
                    error["stage"] = "plan_apply"
                    failures.append(error)
                return False, set()
            candidate_definitions[name] = definition
        for name, definition in candidate_definitions.items():
            for upstream in definition.upstream_features:
                if upstream not in candidate_definitions:
                    raw_injection = definition.raw.get(
                        "_injection_id", definition.raw.get("injection_id")
                    )
                    failures.append(
                        failure(
                            code="E_DEFINITION_INVALID",
                            stage="plan_apply",
                            message=f"definition change leaves unknown upstream feature {upstream}",
                            feature_name=name,
                            injection_id=str(raw_injection) if raw_injection is not None else None,
                        )
                    )
                    return False, set()
        result = build_registry(candidate_definitions)
        if result.failures:
            failures.extend({**item, "stage": "plan_apply"} for item in result.failures)
            return False, set()
        old_fps = {
            name: definition.parameter_fingerprint
            for name, definition in self.registry.definitions.items()
        }
        changed_names = {
            name
            for name, definition in result.definitions.items()
            if old_fps.get(name) != definition.parameter_fingerprint
        }
        removed_names = set(old_fps) - set(result.definitions)
        for name in removed_names:
            for unit in list(self.feature_values):
                if unit[0] == name:
                    self.feature_values.pop(unit, None)
            self.pending_features = {unit for unit in self.pending_features if unit[0] != name}
            for key, item in list(self.runtime_failures.items()):
                if item.get("feature_name") == name:
                    self.runtime_failures.pop(key, None)
        self.registry = Registry.from_build(result)
        return True, changed_names

    def export_load_stage(self) -> dict[str, Any]:
        return {
            "events": [event.public() for event in self.events.order],
            "definitions": [
                copy.deepcopy(self.registry.definitions[name].raw) for name in self.registry.topo
            ],
            "load_failures": self.load_failures,
            "operation_failures": self.operation_failures,
            "batch_summaries": self.batch_summaries,
        }

    def export_feature_stage(self) -> dict[str, Any]:
        return {
            "feature_values": self.features_public(),
            "runtime_failures": sorted(self.runtime_failures.values(), key=failure_sort_key),
            "pending": self.pending_public("feature"),
        }

    def export_score_stage(self) -> dict[str, Any]:
        return {"score_values": self.scores_public(), "pending": self.pending_public("score")}

    @classmethod
    def from_stages(
        cls,
        *,
        config: ScoreConfig,
        load_stage: dict[str, Any],
        feature_stage: dict[str, Any],
        score_stage: dict[str, Any],
        faults: FaultPlan | None = None,
    ) -> "Workbench":
        definitions: dict[str, FeatureDef] = {}
        for index, raw in enumerate(load_stage.get("definitions", [])):
            definition, error = parse_feature_definition(raw, index)
            if error or definition is None:
                raise ValueError(f"invalid snapshot definition: {error}")
            definitions[definition.name] = definition
        registry = Registry.from_definitions(definitions)
        events: list[Event] = []
        previous: int | None = None
        for index, raw in enumerate(load_stage.get("events", [])):
            event, error = validate_event_payload(
                raw,
                existing_ids={item.event_id for item in events},
                previous_timestamp=previous,
                allowed_behaviors=config.allowed_behaviors,
                source_index=index,
            )
            if error or event is None:
                raise ValueError(f"invalid snapshot event: {error}")
            events.append(event)
            previous = event.timestamp_ms
        instance = cls(
            config=config,
            definitions=registry.definitions,
            topo=list(registry.topo),
            downstream=registry.downstream,
            events=events,
            load_failures=list(load_stage.get("load_failures", [])),
            faults=faults,
        )
        instance.operation_failures = list(load_stage.get("operation_failures", []))
        instance.batch_summaries = list(load_stage.get("batch_summaries", []))
        for raw in feature_stage.get("feature_values", []):
            record = _feature_value_from_public(raw)
            instance.feature_values[(record.feature_name, record.side, record.entity_id)] = record
            if not record.stale:
                instance.feature_cache[record.cache_key] = record
        for raw in feature_stage.get("runtime_failures", []):
            instance.runtime_failures[_runtime_key(raw)] = raw
        for raw in feature_stage.get("pending", []) + score_stage.get("pending", []):
            if raw.get("stage") == "feature":
                unit = (raw.get("feature_name"), raw.get("side"), raw.get("entity_id"))
                instance.pending_features.add(unit)
                instance.pending_feature_details[unit] = tuple(raw.get("event_ids", []))
            elif raw.get("stage") == "score":
                unit = (raw.get("side"), raw.get("entity_id"))
                instance.pending_scores.add(unit)
                instance.pending_score_details[unit] = tuple(raw.get("event_ids", []))
        for raw in score_stage.get("score_values", []):
            record = _score_value_from_public(raw)
            instance.score_values[(record.entity_type, record.entity_id)] = record
            if not record.stale:
                instance.score_cache[record.cache_key] = record
        instance.run_count = 1
        return instance


def _feature_value_from_public(raw: dict[str, Any]) -> FeatureValue:
    return FeatureValue(
        feature_name=raw["feature_name"],
        side=raw["side"],
        entity_id=raw["entity_id"],
        value_fixed=int(raw["value_fixed"]),
        decimal_places=int(raw["decimal_places"]),
        quantization_step=str(raw["quantization_step"]),
        event_ids=tuple(raw.get("event_ids", [])),
        event_window_fingerprint=str(raw["event_window_fingerprint"]),
        parameter_fingerprint=str(raw["parameter_fingerprint"]),
        upstream_fingerprints=tuple(tuple(item) for item in raw.get("upstream_fingerprints", [])),
        cache_key=str(raw["cache_key"]),
        content_fingerprint=str(raw["content_fingerprint"]),
        stale=bool(raw.get("stale", False)),
    )


def _score_value_from_public(raw: dict[str, Any]) -> ScoreValue:
    contributions = tuple(
        ScoreContribution(
            feature_name=item["feature_name"],
            feature_value_fixed=int(item["feature_value_fixed"]),
            feature_decimal_places=int(item["feature_decimal_places"]),
            feature_value_decimal=str(item["feature_value_decimal"]),
            feature_content_fingerprint=str(item["feature_content_fingerprint"]),
            weight=str(item["weight"]),
            contribution_decimal=str(item["contribution_decimal"]),
            contribution_fixed=int(item["contribution_fixed"]),
        )
        for item in raw.get("contributions", [])
    )
    return ScoreValue(
        entity_type=raw["entity_type"],
        entity_id=raw["entity_id"],
        score_fixed=int(raw["score_fixed"]),
        score_decimal_places=int(raw["score_decimal_places"]),
        score_quantum=str(raw["score_quantum"]),
        score_decimal=str(raw["score_decimal"]),
        bias_contribution=str(raw["bias_contribution"]),
        contributions=contributions,
        participating_event_count=int(raw["participating_event_count"]),
        feature_fingerprints=tuple(tuple(item) for item in raw.get("feature_fingerprints", [])),
        cache_key=str(raw["cache_key"]),
        content_fingerprint=str(raw["content_fingerprint"]),
        stale=bool(raw.get("stale", False)),
    )
