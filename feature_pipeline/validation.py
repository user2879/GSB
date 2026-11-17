"""Input loading and business validation.

Validation is intentionally non-throwing at the file level: each bad row or
definition produces a stable failure record so later valid rows can proceed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from .canonical import is_finite_number
from .models import (
    EventFilter,
    FeatureDefinition,
    FeatureTable,
    LabelDefinition,
    PipelineConfig,
)

E_EVENT_INVALID = "E_EVENT_INVALID"
E_SCHEMA_UNKNOWN_FIELD = "E_SCHEMA_UNKNOWN_FIELD"
E_FEATURE_CYCLE = "E_FEATURE_CYCLE"
E_TIMESTAMP_INVALID = "E_TIMESTAMP_INVALID"
E_FEATURE_COMPUTE = "E_FEATURE_COMPUTE"
E_LABEL_UNDEFINED = "E_LABEL_UNDEFINED"

EVENT_FIELDS = {
    "event_id",
    "user_id",
    "item_id",
    "event_type",
    "timestamp",
    "context",
}
CONTEXT_FIELDS = {"event_id", "user_id", "item_id", "event_type", "timestamp", "context"}
FEATURE_TABLE_FIELDS = {"features"}
FEATURE_FIELDS = {
    "name",
    "entity",
    "window_ms",
    "aggregation",
    "source_field",
    "source_feature",
    "upstream_features",
    "empty_value",
    "filter",
}
FILTER_FIELDS = {"field", "equals"}
LABEL_FIELDS = {
    "name",
    "trigger_event_types",
    "delay_ms",
    "feature_names",
    "positive_rule",
    "negative_rule",
}
CONFIG_FIELDS = {"allowed_event_types", "max_workers", "stage_delay_ms"}
RULE_GROUP_FIELDS = {"all", "any", "not", "else"}
RULE_CONDITION_FIELDS = {"feature", "operator", "value"}
OPERATORS = {"gt", "ge", "lt", "le", "eq", "ne"}
AGGREGATIONS = {"count", "sum", "mean", "latest"}
ENTITIES = {"user", "item"}


def failure(
    code: str,
    stage: str,
    event_id: str | None = None,
    error_type: str = "ValidationError",
    message: str = "",
    **extra: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "code": code,
        "stage": stage,
        "event_id": event_id or "<none>",
        "error_type": error_type,
        "message": message,
    }
    record.update(extra)
    return record


def _unknown_fields(
    value: dict[str, Any], allowed: set[str], stage: str, event_id: str | None = None
) -> list[dict[str, Any]]:
    return [
        failure(
            E_SCHEMA_UNKNOWN_FIELD,
            stage,
            event_id,
            "UnknownFieldError",
            f"unknown field: {field}",
            field=field,
        )
        for field in sorted(set(value) - allowed)
    ]


def load_json_document(path: Path, stage: str) -> tuple[Any | None, list[dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), []
    except json.JSONDecodeError as exc:
        return None, [
            failure(
                E_EVENT_INVALID,
                stage,
                f"<{path.name}>",
                "JSONDecodeError",
                str(exc),
            )
        ]
    except OSError as exc:
        return None, [
            failure(E_EVENT_INVALID, stage, f"<{path.name}>", type(exc).__name__, str(exc))
        ]


def validate_config(raw: Any) -> tuple[PipelineConfig | None, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        return None, [failure(E_EVENT_INVALID, "config", "<config>", "TypeError", "config must be an object")]
    failures.extend(_unknown_fields(raw, CONFIG_FIELDS, "config", "<config>"))
    allowed = raw.get("allowed_event_types")
    if (
        not isinstance(allowed, list)
        or not allowed
        or any(not isinstance(item, str) or not item for item in allowed)
        or len(set(allowed)) != len(allowed)
    ):
        failures.append(
            failure(
                E_EVENT_INVALID,
                "config",
                "<config>",
                "EnumError",
                "allowed_event_types must be a non-empty unique string list",
            )
        )
        allowed = []
    max_workers = raw.get("max_workers", 4)
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        failures.append(
            failure(E_EVENT_INVALID, "config", "<config>", "ValueError", "max_workers must be >= 1")
        )
        max_workers = 1
    stage_delay_ms = raw.get("stage_delay_ms", 0)
    if (
        isinstance(stage_delay_ms, bool)
        or not isinstance(stage_delay_ms, int)
        or stage_delay_ms < 0
    ):
        failures.append(
            failure(
                E_EVENT_INVALID,
                "config",
                "<config>",
                "ValueError",
                "stage_delay_ms must be a non-negative integer",
            )
        )
        stage_delay_ms = 0
    if failures:
        return None, failures
    return (
        PipelineConfig(
            allowed_event_types=tuple(allowed),
            max_workers=max_workers,
            stage_delay_ms=stage_delay_ms,
            raw=dict(raw),
        ),
        failures,
    )


def validate_event(
    raw: Any,
    allowed_event_types: set[str],
    seen_event_ids: set[str],
    line_number: int | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    line_event = f"<line:{line_number}>" if line_number is not None else "<event>"
    if not isinstance(raw, dict):
        return None, [
            failure(E_EVENT_INVALID, "event_load", line_event, "TypeError", "event must be an object")
        ]
    event_id = raw.get("event_id")
    failure_event_id = event_id if isinstance(event_id, str) and event_id else line_event
    failures.extend(_unknown_fields(raw, EVENT_FIELDS, "event_load", failure_event_id))
    required = ("event_id", "user_id", "item_id", "event_type", "timestamp", "context")
    missing = [field for field in required if field not in raw]
    if missing:
        failures.append(
            failure(
                E_EVENT_INVALID,
                "event_load",
                failure_event_id,
                "MissingFieldError",
                f"missing fields: {','.join(missing)}",
                fields=missing,
            )
        )
    for field in ("event_id", "user_id", "item_id"):
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            failures.append(
                failure(
                    E_EVENT_INVALID,
                    "event_load",
                    failure_event_id,
                    "IdentifierError",
                    f"{field} must be a non-empty string",
                    field=field,
                )
            )
    if isinstance(event_id, str) and event_id in seen_event_ids:
        failures.append(
            failure(
                E_EVENT_INVALID,
                "event_load",
                failure_event_id,
                "DuplicateEventError",
                f"duplicate event_id: {event_id}",
            )
        )
    timestamp = raw.get("timestamp")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        failures.append(
            failure(
                E_TIMESTAMP_INVALID,
                "event_load",
                failure_event_id,
                "TimestampError",
                "timestamp must be a non-negative integer",
            )
        )
    event_type = raw.get("event_type")
    if not isinstance(event_type, str) or event_type not in allowed_event_types:
        failures.append(
            failure(
                E_EVENT_INVALID,
                "event_load",
                failure_event_id,
                "EnumError",
                f"unknown event_type: {event_type!r}",
                field="event_type",
            )
        )
    context = raw.get("context")
    if not isinstance(context, dict):
        failures.append(
            failure(
                E_EVENT_INVALID,
                "event_load",
                failure_event_id,
                "ContextError",
                "context must be an object of finite numeric values",
            )
        )
    else:
        for key in sorted(context):
            if not isinstance(key, str) or not key:
                failures.append(
                    failure(
                        E_EVENT_INVALID,
                        "event_load",
                        failure_event_id,
                        "ContextError",
                        "context keys must be non-empty strings",
                    )
                )
            elif not is_finite_number(context[key]):
                failures.append(
                    failure(
                        E_EVENT_INVALID,
                        "event_load",
                        failure_event_id,
                        "NonFiniteNumberError",
                        f"context.{key} must be a finite number",
                        field=f"context.{key}",
                    )
                )
    if failures:
        return None, failures
    normalized = {
        "event_id": raw["event_id"],
        "user_id": raw["user_id"],
        "item_id": raw["item_id"],
        "event_type": raw["event_type"],
        "timestamp": raw["timestamp"],
        "context": dict(raw["context"]),
    }
    return normalized, failures


def validate_feature_table(raw: Any) -> tuple[FeatureTable | None, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        return None, [
            failure(E_EVENT_INVALID, "schema", "<features>", "TypeError", "feature table must be an object")
        ]
    failures.extend(_unknown_fields(raw, FEATURE_TABLE_FIELDS, "schema", "<features>"))
    items = raw.get("features")
    if not isinstance(items, list):
        failures.append(
            failure(E_EVENT_INVALID, "schema", "<features>", "TypeError", "features must be a list")
        )
        return None, failures

    definitions: dict[str, FeatureDefinition] = {}
    for index, item in enumerate(items):
        location = f"<feature:{index}>"
        if not isinstance(item, dict):
            failures.append(
                failure(E_EVENT_INVALID, "schema", location, "TypeError", "feature must be an object")
            )
            continue
        name = item.get("name")
        failure_event = f"<feature:{name}>" if isinstance(name, str) and name else location
        failures.extend(_unknown_fields(item, FEATURE_FIELDS, "schema", failure_event))
        local_failures: list[dict[str, Any]] = []
        if not isinstance(name, str) or not name:
            local_failures.append(
                failure(E_EVENT_INVALID, "schema", failure_event, "FeatureNameError", "feature name must be non-empty")
            )
        elif name in definitions:
            local_failures.append(
                failure(
                    E_SCHEMA_UNKNOWN_FIELD,
                    "schema",
                    failure_event,
                    "DuplicateFeatureError",
                    f"duplicate feature name: {name}",
                )
            )
        entity = item.get("entity")
        if entity not in ENTITIES:
            local_failures.append(
                failure(
                    E_SCHEMA_UNKNOWN_FIELD,
                    "schema",
                    failure_event,
                    "EntityError",
                    "entity must be user or item",
                )
            )
        window_ms = item.get("window_ms")
        if isinstance(window_ms, bool) or not isinstance(window_ms, int) or window_ms < 0:
            local_failures.append(
                failure(
                    E_EVENT_INVALID,
                    "schema",
                    failure_event,
                    "WindowError",
                    "window_ms must be a non-negative integer",
                )
            )
        aggregation = item.get("aggregation")
        if aggregation not in AGGREGATIONS:
            local_failures.append(
                failure(
                    E_SCHEMA_UNKNOWN_FIELD,
                    "schema",
                    failure_event,
                    "AggregationError",
                    "aggregation must be count, sum, mean, or latest",
                )
            )
        source_field = item.get("source_field")
        source_feature = item.get("source_feature")
        if source_field is not None and (
            not isinstance(source_field, str)
            or not source_field
            or source_field.startswith(".")
            or ".." in source_field
        ):
            local_failures.append(
                failure(
                    E_SCHEMA_UNKNOWN_FIELD,
                    "schema",
                    failure_event,
                    "SourceFieldError",
                    "source_field must be a dotted field path",
                )
            )
        if source_field is not None and source_feature is not None:
            local_failures.append(
                failure(
                    E_SCHEMA_UNKNOWN_FIELD,
                    "schema",
                    failure_event,
                    "SourceError",
                    "only one of source_field and source_feature is allowed",
                )
            )
        if aggregation in {"sum", "mean", "latest"} and source_field is None and source_feature is None:
            local_failures.append(
                failure(
                    E_SCHEMA_UNKNOWN_FIELD,
                    "schema",
                    failure_event,
                    "SourceError",
                    f"aggregation {aggregation} requires source_field or source_feature",
                )
            )
        upstream = item.get("upstream_features", [])
        if (
            not isinstance(upstream, list)
            or any(not isinstance(up, str) or not up for up in upstream)
            or len(set(upstream)) != len(upstream)
        ):
            local_failures.append(
                failure(
                    E_SCHEMA_UNKNOWN_FIELD,
                    "schema",
                    failure_event,
                    "DependencyError",
                    "upstream_features must be a unique string list",
                )
            )
            upstream = []
        if isinstance(source_feature, str) and source_feature not in upstream:
            local_failures.append(
                failure(
                    E_SCHEMA_UNKNOWN_FIELD,
                    "schema",
                    failure_event,
                    "DependencyError",
                    "source_feature must be listed in upstream_features",
                )
            )
        if isinstance(name, str) and name in upstream:
            local_failures.append(
                failure(
                    E_FEATURE_CYCLE,
                    "schema",
                    failure_event,
                    "CycleError",
                    "feature cannot depend on itself",
                )
            )
        empty_value = item.get("empty_value", 0)
        if not is_finite_number(empty_value):
            local_failures.append(
                failure(
                    E_EVENT_INVALID,
                    "schema",
                    failure_event,
                    "EmptyValueError",
                    "empty_value must be a finite number",
                )
            )
        event_filter: EventFilter | None = None
        filter_raw = item.get("filter")
        if filter_raw is not None:
            if not isinstance(filter_raw, dict):
                local_failures.append(
                    failure(E_SCHEMA_UNKNOWN_FIELD, "schema", failure_event, "FilterError", "filter must be an object")
                )
            else:
                failures.extend(_unknown_fields(filter_raw, FILTER_FIELDS, "schema", failure_event))
                filter_field = filter_raw.get("field")
                if not isinstance(filter_field, str) or not filter_field:
                    local_failures.append(
                        failure(
                            E_SCHEMA_UNKNOWN_FIELD,
                            "schema",
                            failure_event,
                            "FilterError",
                            "filter.field must be a non-empty path",
                        )
                    )
                elif "equals" not in filter_raw:
                    local_failures.append(
                        failure(
                            E_SCHEMA_UNKNOWN_FIELD,
                            "schema",
                            failure_event,
                            "FilterError",
                            "filter.equals is required",
                        )
                    )
                elif not isinstance(filter_raw["equals"], (str, int, float, bool)) or (
                    isinstance(filter_raw["equals"], float) and not is_finite_number(filter_raw["equals"])
                ):
                    local_failures.append(
                        failure(
                            E_SCHEMA_UNKNOWN_FIELD,
                            "schema",
                            failure_event,
                            "FilterError",
                            "filter.equals must be a JSON scalar",
                        )
                    )
                else:
                    event_filter = EventFilter(filter_field, filter_raw["equals"])
        failures.extend(local_failures)
        if local_failures:
            continue
        definitions[name] = FeatureDefinition(
            name=name,
            entity=entity,
            window_ms=window_ms,
            aggregation=aggregation,
            source_field=source_field,
            source_feature=source_feature,
            upstream_features=tuple(upstream),
            empty_value=empty_value,
            event_filter=event_filter,
        )

    all_names = set(definitions)
    dependency_failures: list[dict[str, Any]] = []
    for definition in definitions.values():
        for upstream in definition.upstream_features:
            if upstream not in all_names:
                dependency_failures.append(
                    failure(
                        E_SCHEMA_UNKNOWN_FIELD,
                        "schema",
                        f"<feature:{definition.name}>",
                        "MissingDependencyError",
                        f"unknown upstream feature: {upstream}",
                    )
                )
        if definition.source_feature is not None:
            upstream_definition = definitions.get(definition.source_feature)
            if upstream_definition is not None and upstream_definition.entity != definition.entity:
                dependency_failures.append(
                    failure(
                        E_SCHEMA_UNKNOWN_FIELD,
                        "schema",
                        f"<feature:{definition.name}>",
                        "EntityDependencyError",
                        "source_feature must have the same entity type",
                    )
                )
    failures.extend(dependency_failures)
    if failures:
        return None, failures

    levels = _dependency_levels(definitions)
    if len([name for level in levels for name in level]) != len(definitions):
        cycle_names = sorted(all_names - {name for level in levels for name in level})
        failures.append(
            failure(
                E_FEATURE_CYCLE,
                "schema",
                "<feature-definitions>",
                "CycleError",
                f"feature dependency cycle: {','.join(cycle_names)}",
                features=cycle_names,
            )
        )
        return None, failures
    ordered = tuple(name for level in levels for name in level)
    positions = {name: index for index, name in enumerate(ordered)}
    table = FeatureTable(
        definitions=definitions,
        topological_order=ordered,
        levels=tuple(tuple(name for name in level) for level in levels),
        positions=positions,
        raw={"features": [definitions[name].normalized() for name in ordered]},
    )
    return table, failures


def _dependency_levels(definitions: dict[str, FeatureDefinition]) -> list[list[str]]:
    levels: dict[str, int] = {}
    remaining = set(definitions)
    while remaining:
        ready = sorted(
            name
            for name in remaining
            if all(upstream in levels for upstream in definitions[name].upstream_features)
        )
        if not ready:
            break
        for name in ready:
            dependencies = definitions[name].upstream_features
            levels[name] = max((levels[up] + 1 for up in dependencies), default=0)
        remaining.difference_update(ready)
    grouped: dict[int, list[str]] = {}
    for name, level in levels.items():
        grouped.setdefault(level, []).append(name)
    return [sorted(grouped[level]) for level in sorted(grouped)]


def validate_label_definition(
    raw: Any, feature_table: FeatureTable
) -> tuple[LabelDefinition | None, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        return None, [
            failure(E_EVENT_INVALID, "schema", "<labels>", "TypeError", "label definition must be an object")
        ]
    failures.extend(_unknown_fields(raw, LABEL_FIELDS, "schema", "<labels>"))
    name = raw.get("name", "default_label")
    if not isinstance(name, str) or not name:
        failures.append(
            failure(E_EVENT_INVALID, "schema", "<labels>", "LabelNameError", "label name must be non-empty")
        )
    trigger_types = raw.get("trigger_event_types", [])
    if (
        not isinstance(trigger_types, list)
        or any(not isinstance(item, str) or not item for item in trigger_types)
        or len(set(trigger_types)) != len(trigger_types)
    ):
        failures.append(
            failure(
                E_SCHEMA_UNKNOWN_FIELD,
                "schema",
                "<labels>",
                "TriggerError",
                "trigger_event_types must be a unique string list when provided",
            )
        )
        trigger_types = []
    delay_ms = raw.get("delay_ms")
    if isinstance(delay_ms, bool) or not isinstance(delay_ms, int) or delay_ms < 0:
        failures.append(
            failure(E_TIMESTAMP_INVALID, "schema", "<labels>", "DelayError", "delay_ms must be non-negative")
        )
    feature_names = raw.get("feature_names")
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or any(not isinstance(item, str) or not item for item in feature_names)
        or len(set(feature_names)) != len(feature_names)
    ):
        failures.append(
            failure(
                E_SCHEMA_UNKNOWN_FIELD,
                "schema",
                "<labels>",
                "FeatureListError",
                "feature_names must be a non-empty unique string list",
            )
        )
        feature_names = []
    referenced: set[str] = set()
    for field in ("positive_rule", "negative_rule"):
        rule = raw.get(field)
        rule_failures, field_references = _validate_rule(rule, field, feature_table)
        failures.extend(rule_failures)
        referenced.update(field_references)
    for feature_name in sorted(feature_names):
        if feature_name not in feature_table.definitions:
            failures.append(
                failure(
                    E_SCHEMA_UNKNOWN_FIELD,
                    "schema",
                    "<labels>",
                    "MissingFeatureReferenceError",
                    f"label references unknown feature: {feature_name}",
                    feature=feature_name,
                )
            )
    for feature_name in sorted(referenced - set(feature_names)):
        failures.append(
            failure(
                E_SCHEMA_UNKNOWN_FIELD,
                "schema",
                "<labels>",
                "FeatureScopeError",
                f"rule feature is not listed in feature_names: {feature_name}",
                feature=feature_name,
            )
        )
    if failures:
        return None, failures
    return (
        LabelDefinition(
            name=name,
            trigger_event_types=tuple(trigger_types),
            delay_ms=delay_ms,
            feature_names=tuple(sorted(feature_names)),
            positive_rule=raw["positive_rule"],
            negative_rule=raw["negative_rule"],
            raw=dict(raw),
        ),
        failures,
    )


def _validate_rule(
    rule: Any, path: str, feature_table: FeatureTable
) -> tuple[list[dict[str, Any]], set[str]]:
    failures: list[dict[str, Any]] = []
    references: set[str] = set()
    if not isinstance(rule, dict):
        return [
            failure(E_EVENT_INVALID, "schema", "<labels>", "RuleError", f"{path} must be an object")
        ], references
    keys = set(rule)
    if keys == {"else"} and rule["else"] is True:
        return failures, references
    if keys == {"not"}:
        nested_failures, nested_references = _validate_rule(rule["not"], f"{path}.not", feature_table)
        return nested_failures, nested_references
    if len(keys) == 1 and next(iter(keys)) in {"all", "any"}:
        group_key = next(iter(keys))
        children = rule[group_key]
        if not isinstance(children, list) or not children:
            failures.append(
                failure(
                    E_SCHEMA_UNKNOWN_FIELD,
                    "schema",
                    "<labels>",
                    "RuleError",
                    f"{path}.{group_key} must be a non-empty list",
                )
            )
            return failures, references
        for index, child in enumerate(children):
            child_failures, child_references = _validate_rule(
                child, f"{path}.{group_key}[{index}]", feature_table
            )
            failures.extend(child_failures)
            references.update(child_references)
        return failures, references
    if not set(rule).issubset(RULE_CONDITION_FIELDS):
        for field in sorted(keys - RULE_CONDITION_FIELDS):
            failures.append(
                failure(
                    E_SCHEMA_UNKNOWN_FIELD,
                    "schema",
                    "<labels>",
                    "UnknownFieldError",
                    f"unknown rule field: {path}.{field}",
                    field=field,
                )
            )
        return failures, references
    feature_name = rule.get("feature")
    operator = rule.get("operator")
    if not isinstance(feature_name, str) or not feature_name:
        failures.append(
            failure(E_EVENT_INVALID, "schema", "<labels>", "RuleError", f"{path}.feature is required")
        )
    elif feature_name not in feature_table.definitions:
        failures.append(
            failure(
                E_SCHEMA_UNKNOWN_FIELD,
                "schema",
                "<labels>",
                "MissingFeatureReferenceError",
                f"label references unknown feature: {feature_name}",
                feature=feature_name,
            )
        )
    else:
        references.add(feature_name)
    if operator not in OPERATORS:
        failures.append(
            failure(E_SCHEMA_UNKNOWN_FIELD, "schema", "<labels>", "OperatorError", f"unknown operator: {operator}")
        )
    if "value" not in rule or not is_finite_number(rule.get("value")):
        failures.append(
            failure(E_EVENT_INVALID, "schema", "<labels>", "RuleValueError", f"{path}.value must be finite")
        )
    return failures, references


def load_event_file(
    path: Path,
    allowed_event_types: set[str],
    transform: Callable[[Any, int], Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous_timestamp: int | None = None
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        return [], [failure(E_EVENT_INVALID, "event_load", f"<{path.name}>", type(exc).__name__, str(exc))]
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(
                    failure(
                        E_EVENT_INVALID,
                        "event_load",
                        f"<line:{line_number}>",
                        "JSONDecodeError",
                        str(exc),
                    )
                )
                continue
            if transform is not None:
                raw = transform(raw, line_number)
            event, event_failures = validate_event(raw, allowed_event_types, seen, line_number)
            if event_failures:
                failures.extend(event_failures)
                continue
            timestamp = event["timestamp"]
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                failures.append(
                    failure(
                        E_TIMESTAMP_INVALID,
                        "event_load",
                        event["event_id"],
                        "TimestampOrderError",
                        f"timestamp {timestamp} is not strictly greater than {previous_timestamp}",
                    )
                )
                continue
            previous_timestamp = timestamp
            seen.add(event["event_id"])
            events.append(event)
    return events, failures


def apply_rule(rule: dict[str, Any], values: dict[str, int | float]) -> bool:
    if "else" in rule:
        return bool(rule["else"])
    if "not" in rule:
        return not apply_rule(rule["not"], values)
    if "all" in rule:
        return all(apply_rule(child, values) for child in rule["all"])
    if "any" in rule:
        return any(apply_rule(child, values) for child in rule["any"])
    left = float(values[rule["feature"]])
    right = float(rule["value"])
    operator = rule["operator"]
    if operator == "gt":
        return left > right
    if operator == "ge":
        return left >= right
    if operator == "lt":
        return left < right
    if operator == "le":
        return left <= right
    if operator == "eq":
        return left == right
    if operator == "ne":
        return left != right
    raise ValueError(f"unknown operator: {operator}")


def sort_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(events, key=lambda event: (event["timestamp"], event["event_id"]))
