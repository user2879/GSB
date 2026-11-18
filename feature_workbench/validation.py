from __future__ import annotations

import heapq
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Iterable

import yaml

from .canonical import (
    bytes_fingerprint,
    canonical_json,
    decimal_text,
    fingerprint,
)
from .models import Event, FeatureDef, ScoreConfig, failure, failure_sort_key


AGGREGATES = frozenset({"count", "sum", "mean", "latest"})
SIDES = frozenset({"user", "item", "both"})
DEFAULT_BEHAVIORS = ("view", "click", "favorite", "cart", "purchase", "refund")
MAX_DECIMAL_PLACES = 18
MIN_QUANTUM = Decimal("1e-12")
MAX_QUANTUM = Decimal("1e12")


@dataclass
class DefinitionLoadResult:
    definitions: dict[str, FeatureDef]
    topo: list[str]
    downstream: dict[str, tuple[str, ...]]
    failures: list[dict[str, Any]]
    raw_document: dict[str, Any]
    source_hash: str


@dataclass
class EventLoadResult:
    events: dict[str, Event]
    failures: list[dict[str, Any]]
    cursor: int
    source_hash: str


@dataclass
class RegistryBuildResult:
    definitions: dict[str, FeatureDef]
    topo: list[str]
    downstream: dict[str, tuple[str, ...]]
    failures: list[dict[str, Any]]


@dataclass(frozen=True)
class FaultPlan:
    feature_faults: dict[str, tuple[str, str]]
    score_faults: dict[tuple[str, str], tuple[str, str, str | None]]
    any_score_fault: dict[str, tuple[str, str, str | None]] = field(default_factory=dict)


@dataclass
class InjectionBundle:
    expected: list[dict[str, Any]]
    faults: FaultPlan
    raw: dict[str, Any]


def _injection_id(raw: dict[str, Any]) -> str | None:
    value = raw.get("_injection_id", raw.get("injection_id"))
    return str(value) if value is not None else None


def _definition_failure(raw: Any, message: str, index: int | None = None) -> dict[str, Any]:
    raw_dict = raw if isinstance(raw, dict) else {}
    name = raw_dict.get("name")
    detail: dict[str, Any] = {}
    if index is not None:
        detail["entry_index"] = index
    return failure(
        code="E_DEFINITION_INVALID",
        stage="definition_load",
        message=message,
        feature_name=str(name) if name is not None else None,
        injection_id=_injection_id(raw_dict),
        detail=detail or None,
    )


def parse_decimal(value: Any, field_name: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field_name} must be a finite number")
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not number.is_finite():
        raise ValueError(f"{field_name} must be a finite number")
    return number


def validate_quantum(decimal_places: int, quantum: Decimal) -> None:
    if quantum < MIN_QUANTUM or quantum > MAX_QUANTUM:
        raise ValueError(
            f"quantization_step must be in [{decimal_text(MIN_QUANTUM)}, {decimal_text(MAX_QUANTUM)}]"
        )
    with localcontext() as context:
        context.prec = 80
        scaled = quantum.scaleb(decimal_places)
        representable = scaled == scaled.to_integral_value()
    if not representable:
        raise ValueError(
            "quantization_step is not representable at the configured decimal_places"
        )


def parse_feature_definition(raw: Any, index: int | None = None) -> tuple[FeatureDef | None, dict[str, Any] | None]:
    if not isinstance(raw, dict):
        return None, _definition_failure(raw, "feature definition must be an object", index)
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return None, _definition_failure(raw, "name must be a non-empty string", index)
    side = raw.get("side", "both")
    if side not in SIDES:
        return None, _definition_failure(raw, "side must be one of user, item, both", index)
    window_ms = raw.get("window_ms")
    if isinstance(window_ms, bool) or not isinstance(window_ms, int) or window_ms <= 0:
        return None, _definition_failure(raw, "window_ms must be a positive integer", index)
    aggregate = raw.get("aggregate")
    if aggregate not in AGGREGATES:
        return None, _definition_failure(raw, "aggregate must be count, sum, mean, or latest", index)
    decimal_places = raw.get("decimal_places", raw.get("fixed_decimal_places"))
    if (
        isinstance(decimal_places, bool)
        or not isinstance(decimal_places, int)
        or decimal_places < 0
        or decimal_places > MAX_DECIMAL_PLACES
    ):
        return None, _definition_failure(
            raw, f"decimal_places must be an integer in [0, {MAX_DECIMAL_PLACES}]", index
        )
    quantum_raw = raw.get(
        "quantization_step", raw.get("quantum", raw.get("quantization_level"))
    )
    if quantum_raw is None:
        return None, _definition_failure(raw, "quantization_step is required", index)
    try:
        quantum = parse_decimal(quantum_raw, "quantization_step")
        if quantum <= 0:
            raise ValueError("quantization_step must be positive")
        validate_quantum(decimal_places, quantum)
    except ValueError as exc:
        return None, _definition_failure(raw, str(exc), index)
    source_field = raw.get("source_field")
    if source_field is not None and (not isinstance(source_field, str) or not source_field.strip()):
        return None, _definition_failure(raw, "source_field must be a non-empty string or null", index)
    upstream_raw = raw.get("upstream_features", [])
    if upstream_raw is None:
        upstream_raw = []
    if not isinstance(upstream_raw, list) or any(
        not isinstance(item, str) or not item.strip() for item in upstream_raw
    ):
        return None, _definition_failure(raw, "upstream_features must be a list of feature names", index)
    if len(set(upstream_raw)) != len(upstream_raw):
        return None, _definition_failure(raw, "upstream_features contains duplicates", index)
    if name in upstream_raw:
        return None, _definition_failure(raw, "a feature cannot depend on itself", index)
    if aggregate != "count" and source_field is None and not upstream_raw:
        return None, _definition_failure(
            raw, "non-count features require source_field or upstream_features", index
        )
    semantic = {
        "name": name,
        "side": side,
        "source_field": source_field,
        "window_ms": window_ms,
        "aggregate": aggregate,
        "decimal_places": decimal_places,
        "quantization_step": decimal_text(quantum),
        "upstream_features": sorted(upstream_raw),
    }
    parameter_fingerprint = fingerprint("feature-definition-v1", semantic)
    definition = FeatureDef(
        name=name,
        side=side,
        source_field=source_field,
        window_ms=window_ms,
        aggregate=aggregate,
        decimal_places=decimal_places,
        quantum=quantum,
        upstream_features=tuple(sorted(upstream_raw)),
        parameter_fingerprint=parameter_fingerprint,
        raw=dict(raw),
    )
    return definition, None


def _compatible_sides(definition: FeatureDef, upstream: FeatureDef) -> bool:
    if definition.side == "both":
        return upstream.side == "both"
    return upstream.side in {"both", definition.side}


def build_registry(definitions: dict[str, FeatureDef]) -> RegistryBuildResult:
    valid = dict(definitions)
    failures: list[dict[str, Any]] = []

    changed = True
    while changed:
        changed = False
        removed: set[str] = set()
        for name, definition in sorted(valid.items()):
            missing = [item for item in definition.upstream_features if item not in valid]
            if missing:
                removed.add(name)
                failures.append(
                    _definition_failure(
                        definition.raw,
                        f"unknown upstream feature(s): {', '.join(sorted(missing))}",
                    )
                )
                continue
            incompatible = [
                item
                for item in definition.upstream_features
                if not _compatible_sides(definition, valid[item])
            ]
            if incompatible:
                removed.add(name)
                failures.append(
                    _definition_failure(
                        definition.raw,
                        f"incompatible side for upstream feature(s): {', '.join(sorted(incompatible))}",
                    )
                )
        if removed:
            for name in removed:
                valid.pop(name, None)
            # A removed invalid upstream changes the dependency set. Retry until stable.
            changed = True
            continue

        indegree = {name: len(definition.upstream_features) for name, definition in valid.items()}
        reverse: dict[str, list[str]] = {name: [] for name in valid}
        for name, definition in valid.items():
            for upstream in definition.upstream_features:
                reverse[upstream].append(name)
        heap = [name for name, degree in indegree.items() if degree == 0]
        heapq.heapify(heap)
        topo: list[str] = []
        while heap:
            name = heapq.heappop(heap)
            topo.append(name)
            for child in sorted(reverse[name]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(heap, child)
        unresolved = sorted(set(valid) - set(topo))
        if unresolved:
            for name in unresolved:
                failures.append(
                    _definition_failure(valid[name].raw, "feature dependency graph contains a cycle")
                )
                valid.pop(name, None)
            changed = True

    return _finish_registry(valid, failures)


def _finish_registry(
    valid: dict[str, FeatureDef], failures: list[dict[str, Any]]
) -> RegistryBuildResult:
    indegree = {name: len(definition.upstream_features) for name, definition in valid.items()}
    reverse: dict[str, list[str]] = {name: [] for name in valid}
    for name, definition in valid.items():
        for upstream in definition.upstream_features:
            reverse[upstream].append(name)
    heap = [name for name, degree in indegree.items() if degree == 0]
    heapq.heapify(heap)
    topo: list[str] = []
    while heap:
        name = heapq.heappop(heap)
        topo.append(name)
        for child in sorted(reverse[name]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(heap, child)
    if len(topo) != len(valid):
        unresolved = sorted(set(valid) - set(topo))
        for name in unresolved:
            failures.append(
                _definition_failure(valid[name].raw, "feature dependency graph contains a cycle")
            )
            valid.pop(name, None)
        return _finish_registry(valid, failures)
    downstream = {name: list(reverse[name]) for name in valid}
    for name in downstream:
        downstream[name].sort()
    unique_failures = {canonical_json(item): item for item in failures}
    ordered_failures = sorted(unique_failures.values(), key=failure_sort_key)
    return RegistryBuildResult(valid, topo, {k: tuple(v) for k, v in downstream.items()}, ordered_failures)


def load_definitions(path: Path) -> DefinitionLoadResult:
    try:
        text = path.read_text(encoding="utf-8")
        raw_document = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        document = {"features": []}
        return DefinitionLoadResult(
            definitions={},
            topo=[],
            downstream={},
            failures=[_definition_failure({}, f"cannot load definition table: {exc}")],
            raw_document=document,
            source_hash=bytes_fingerprint("definition-file-bytes-v1", path.read_bytes())
            if path.exists()
            else fingerprint("empty-definition-v1", {}),
        )
    source_hash = fingerprint("definition-document-v1", raw_document)
    if not isinstance(raw_document, dict) or not isinstance(raw_document.get("features"), list):
        return DefinitionLoadResult(
            definitions={},
            topo=[],
            downstream={},
            failures=[_definition_failure(raw_document, "definition table must contain a features list")],
            raw_document=raw_document if isinstance(raw_document, dict) else {"features": []},
            source_hash=source_hash,
        )
    parsed: list[tuple[int, FeatureDef | None, dict[str, Any] | None]] = []
    failures: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_document["features"]):
        definition, error = parse_feature_definition(raw, index)
        parsed.append((index, definition, error))
        if error:
            failures.append(error)
    counts = Counter(definition.name for _, definition, _ in parsed if definition is not None)
    valid_candidates: dict[str, FeatureDef] = {}
    for index, definition, _ in parsed:
        if definition is None:
            continue
        if counts[definition.name] > 1:
            failures.append(
                _definition_failure(definition.raw, "feature name is not unique", index)
            )
            continue
        valid_candidates[definition.name] = definition
    registry = build_registry(valid_candidates)
    failures.extend(registry.failures)
    unique_failures = {canonical_json(item): item for item in failures}
    return DefinitionLoadResult(
        definitions=registry.definitions,
        topo=registry.topo,
        downstream=registry.downstream,
        failures=sorted(unique_failures.values(), key=failure_sort_key),
        raw_document=raw_document,
        source_hash=source_hash,
    )


def validate_event_payload(
    raw: Any,
    *,
    existing_ids: set[str],
    previous_timestamp: int | None,
    allowed_behaviors: Iterable[str],
    source_index: int | None = None,
    require_strictly_after_previous: bool = True,
) -> tuple[Event | None, dict[str, Any] | None]:
    raw_dict = raw if isinstance(raw, dict) else {}
    injection_id = _injection_id(raw_dict)
    detail = {"source_index": source_index} if source_index is not None else None

    def bad(message: str) -> tuple[None, dict[str, Any]]:
        return None, failure(
            code="E_EVENT_INVALID",
            stage="event_load",
            message=message,
            event_id=str(raw_dict.get("event_id")) if raw_dict.get("event_id") is not None else None,
            injection_id=injection_id,
            detail=detail,
        )

    if not isinstance(raw, dict):
        return bad("event must be an object")
    event_id = raw.get("event_id")
    if not isinstance(event_id, str) or not event_id.strip():
        return bad("event_id must be a non-empty string")
    if event_id in existing_ids:
        return bad("event_id is not unique")
    user_id = raw.get("user_id")
    item_id = raw.get("item_id")
    if not isinstance(user_id, str) or not user_id.strip():
        return bad("user_id must be a non-empty string")
    if not isinstance(item_id, str) or not item_id.strip():
        return bad("item_id must be a non-empty string")
    behavior = raw.get("behavior")
    if not isinstance(behavior, str) or behavior not in set(allowed_behaviors):
        return bad("unknown behavior type")
    timestamp_ms = raw.get("timestamp_ms")
    if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int) or timestamp_ms < 0:
        return bad("timestamp_ms must be a non-negative integer")
    if (
        require_strictly_after_previous
        and previous_timestamp is not None
        and timestamp_ms <= previous_timestamp
    ):
        return bad("timestamp_ms must be strictly greater than the previous accepted event")
    context = raw.get("context", {})
    if not isinstance(context, dict):
        return bad("context must be an object")
    for key, value in context.items():
        if not isinstance(key, str) or not key:
            return bad("context keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return bad(f"context value for {key} must be numeric")
        try:
            finite = math.isfinite(float(value))
        except (OverflowError, ValueError):
            finite = False
        if not finite:
            return bad(f"context value for {key} must be finite")
    content_fingerprint = fingerprint(
        "event-content-v1",
        {
            "event_id": event_id,
            "user_id": user_id,
            "item_id": item_id,
            "behavior": behavior,
            "timestamp_ms": timestamp_ms,
            "context": context,
        },
    )
    return (
        Event(
            event_id=event_id,
            user_id=user_id,
            item_id=item_id,
            behavior=behavior,
            timestamp_ms=timestamp_ms,
            context=dict(context),
            content_fingerprint=content_fingerprint,
            injection_id=injection_id,
        ),
        None,
    )


def load_events(path: Path, allowed_behaviors: Iterable[str]) -> EventLoadResult:
    if not path.exists():
        return EventLoadResult(
            events={},
            failures=[
                failure(
                    code="E_EVENT_INVALID",
                    stage="event_load",
                    message=f"event file does not exist: {path.name}",
                )
            ],
            cursor=0,
            source_hash=bytes_fingerprint("empty-event-file-v1", b""),
        )
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        return EventLoadResult(
            events={},
            failures=[
                failure(
                    code="E_EVENT_INVALID",
                    stage="event_load",
                    message=f"cannot read event file: {exc}",
                )
            ],
            cursor=0,
            source_hash=bytes_fingerprint("unreadable-event-file-v1", b""),
        )
    source_hash = bytes_fingerprint("event-file-bytes-v1", raw_bytes)
    events: dict[str, Event] = {}
    failures: list[dict[str, Any]] = []
    cursor = 0
    previous_timestamp: int | None = None
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        return EventLoadResult(
            events={},
            failures=[
                failure(
                    code="E_EVENT_INVALID",
                    stage="event_load",
                    message=f"event file is not valid UTF-8: {exc}",
                )
            ],
            cursor=0,
            source_hash=source_hash,
        )
    for source_index, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(
                failure(
                    code="E_EVENT_INVALID",
                    stage="event_load",
                    message=f"invalid JSON line: {exc.msg}",
                    detail={"source_index": source_index},
                )
            )
            continue
        event, error = validate_event_payload(
            raw,
            existing_ids=set(events),
            previous_timestamp=previous_timestamp,
            allowed_behaviors=allowed_behaviors,
            source_index=source_index,
        )
        if error:
            failures.append(error)
            continue
        assert event is not None
        events[event.event_id] = event
        previous_timestamp = event.timestamp_ms
        cursor = event.timestamp_ms
    unique_failures = {canonical_json(item): item for item in failures}
    return EventLoadResult(
        events=events,
        failures=sorted(unique_failures.values(), key=failure_sort_key),
        cursor=cursor,
        source_hash=source_hash,
    )


def _config_failure(message: str) -> dict[str, Any]:
    return failure(code="E_CONFIG_INVALID", stage="config_load", message=message)


def _decimal_or_none(value: Any) -> Decimal | None:
    try:
        return parse_decimal(value, "value")
    except ValueError:
        return None


def load_config(path: Path) -> tuple[ScoreConfig | None, dict[str, Any] | None, dict[str, Any]]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        return None, _config_failure(f"cannot load scoring config: {exc}"), {}
    if not isinstance(raw, dict):
        return None, _config_failure("scoring config must be a mapping"), raw if isinstance(raw, dict) else {}
    version = raw.get("version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        return None, _config_failure("version must be a positive integer"), raw
    weights_raw = raw.get("weights")
    if not isinstance(weights_raw, dict) or not weights_raw:
        return None, _config_failure("weights must be a non-empty mapping"), raw
    weights: dict[str, Decimal] = {}
    for name, value in sorted(weights_raw.items()):
        if not isinstance(name, str) or not name:
            return None, _config_failure("weight names must be non-empty strings"), raw
        number = _decimal_or_none(value)
        if number is None:
            return None, _config_failure(f"weight for {name} must be finite"), raw
        weights[name] = number
    temperature = _decimal_or_none(raw.get("fusion_temperature", 1))
    if temperature is None or temperature <= 0:
        return None, _config_failure("fusion_temperature must be positive and finite"), raw
    bias = _decimal_or_none(raw.get("fusion_bias", 0))
    if bias is None:
        return None, _config_failure("fusion_bias must be finite"), raw
    clamp = raw.get("clamp_interval", raw.get("clamp"))
    if isinstance(clamp, dict):
        clamp_min_raw, clamp_max_raw = clamp.get("min"), clamp.get("max")
    elif isinstance(clamp, list) and len(clamp) == 2:
        clamp_min_raw, clamp_max_raw = clamp
    else:
        clamp_min_raw, clamp_max_raw = raw.get("clamp_min", -100), raw.get("clamp_max", 100)
    clamp_min = _decimal_or_none(clamp_min_raw)
    clamp_max = _decimal_or_none(clamp_max_raw)
    if clamp_min is None or clamp_max is None or clamp_min >= clamp_max:
        return None, _config_failure("clamp interval must contain finite min < max"), raw
    concurrency = raw.get("concurrency", 1)
    point_budget = raw.get("point_budget_per_batch", raw.get("point_budget", 10000))
    snapshot_interval = raw.get("snapshot_commit_interval", 1)
    for field_name, value in (
        ("concurrency", concurrency),
        ("point_budget_per_batch", point_budget),
        ("snapshot_commit_interval", snapshot_interval),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return None, _config_failure(f"{field_name} must be a positive integer"), raw
    score_decimal_places = raw.get("score_decimal_places", 4)
    if (
        isinstance(score_decimal_places, bool)
        or not isinstance(score_decimal_places, int)
        or score_decimal_places < 0
        or score_decimal_places > MAX_DECIMAL_PLACES
    ):
        return None, _config_failure("score_decimal_places is out of range"), raw
    score_quantum = _decimal_or_none(raw.get("score_quantization_step", "0.0001"))
    if score_quantum is None or score_quantum <= 0:
        return None, _config_failure("score_quantization_step must be positive"), raw
    try:
        validate_quantum(score_decimal_places, score_quantum)
    except ValueError as exc:
        return None, _config_failure(str(exc)), raw
    behavior_raw = raw.get("allowed_behaviors", raw.get("behaviors", list(DEFAULT_BEHAVIORS)))
    if not isinstance(behavior_raw, list) or not behavior_raw or any(
        not isinstance(item, str) or not item for item in behavior_raw
    ):
        return None, _config_failure("allowed_behaviors must be a non-empty string list"), raw
    scoring_semantic = {
        "version": version,
        "scoring_version": 1,
        "weights": {name: decimal_text(value) for name, value in weights.items()},
        "fusion_temperature": decimal_text(temperature),
        "fusion_bias": decimal_text(bias),
        "clamp_min": decimal_text(clamp_min),
        "clamp_max": decimal_text(clamp_max),
        "score_decimal_places": score_decimal_places,
        "score_quantization_step": decimal_text(score_quantum),
        "allowed_behaviors": behavior_raw,
    }
    semantic = {
        **scoring_semantic,
        "concurrency": concurrency,
        "point_budget_per_batch": point_budget,
        "snapshot_commit_interval": snapshot_interval,
    }
    config = ScoreConfig(
        version=version,
        weights=weights,
        fusion_temperature=temperature,
        fusion_bias=bias,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        concurrency=concurrency,
        point_budget_per_batch=point_budget,
        snapshot_commit_interval=snapshot_interval,
        score_decimal_places=score_decimal_places,
        score_quantum=score_quantum,
        allowed_behaviors=tuple(behavior_raw),
        config_fingerprint=fingerprint(
            "score-config-v1", {"resolved": semantic, "source": raw}
        ),
        scoring_fingerprint=fingerprint("score-semantics-v1", scoring_semantic),
        raw=raw,
    )
    return config, None, raw


def load_plan(path: Path) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"batches": []}, fingerprint("empty-edit-plan-v1", {}), failure(
            code="E_EDIT_PLAN_INVALID",
            stage="plan_load",
            message=f"cannot load edit plan: {exc}",
        )
    if not isinstance(raw, dict) or not isinstance(raw.get("batches", []), list):
        return {"batches": []}, fingerprint("edit-plan-v1", raw), failure(
            code="E_EDIT_PLAN_INVALID",
            stage="plan_load",
            message="edit plan must contain a batches list",
        )
    return raw, fingerprint("edit-plan-v1", raw), None


def load_injection_bundle(path: Path | None) -> InjectionBundle:
    if path is None:
        return InjectionBundle(expected=[], faults=FaultPlan({}, {}), raw={})
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return InjectionBundle(expected=[], faults=FaultPlan({}, {}), raw={})
    expected = raw.get("injections", []) if isinstance(raw, dict) else []
    faults_raw = raw.get("faults", []) if isinstance(raw, dict) else []
    feature_faults: dict[str, tuple[str, str]] = {}
    score_faults: dict[tuple[str, str], tuple[str, str, str | None]] = {}
    any_score_fault: dict[str, tuple[str, str, str | None]] = {}
    for item in faults_raw:
        if not isinstance(item, dict):
            continue
        injection_id = str(item.get("id", ""))
        stage = item.get("stage")
        exception_type = str(item.get("exception_type", "InjectedFault"))
        if stage == "feature" and isinstance(item.get("feature"), str):
            feature_faults[item["feature"]] = (injection_id, exception_type)
        elif stage == "score":
            entity_type = item.get("entity_type")
            entity_id = item.get("entity_id")
            target_feature = item.get("feature") if isinstance(item.get("feature"), str) else None
            fault_value = (injection_id, exception_type, target_feature)
            if entity_type == "*" and entity_id == "*":
                any_score_fault["*"] = fault_value
            elif isinstance(entity_type, str) and isinstance(entity_id, str):
                score_faults[(entity_type, entity_id)] = fault_value
    return InjectionBundle(
        expected=[item for item in expected if isinstance(item, dict)],
        faults=FaultPlan(feature_faults, score_faults, any_score_fault),
        raw=raw if isinstance(raw, dict) else {},
    )
