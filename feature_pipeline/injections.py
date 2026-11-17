"""Fault injection support with explicit business-code accounting."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .canonical import deep_set
from .validation import (
    E_EVENT_INVALID,
    E_SCHEMA_UNKNOWN_FIELD,
    E_TIMESTAMP_INVALID,
    failure,
)

INJECTION_FIELDS = {"id", "event_id", "stage", "kind", "field", "value", "feature"}
INJECTION_KINDS = {
    "non_finite_context",
    "unknown_event_type",
    "unknown_event_field",
    "backfill_before_cursor",
    "missing_label_feature",
}
EXPECTED_CODES = {
    "non_finite_context": E_EVENT_INVALID,
    "unknown_event_type": E_EVENT_INVALID,
    "unknown_event_field": E_SCHEMA_UNKNOWN_FIELD,
    "backfill_before_cursor": E_TIMESTAMP_INVALID,
    "missing_label_feature": E_SCHEMA_UNKNOWN_FIELD,
}


@dataclass(frozen=True)
class Injection:
    injection_id: str
    event_id: str | None
    stage: str
    kind: str
    field: str | None
    value: Any
    feature: str | None


class InjectionTracker:
    def __init__(self, injections: list[Injection]):
        self.injections = injections
        self.matched: set[str] = set()
        self.messages: dict[str, str] = {}

    def transform_source_event(self, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        event_id = raw.get("event_id")
        injection = self._find(event_id, "parse")
        if injection is None:
            return raw
        mutated = copy.deepcopy(raw)
        if injection.kind == "non_finite_context":
            field = injection.field or "context.price"
            deep_set(mutated, field, float("nan"))
        elif injection.kind == "unknown_event_type":
            mutated["event_type"] = str(injection.value or "__INJECTED_UNKNOWN__")
        elif injection.kind == "unknown_event_field":
            mutated[injection.field or "__injected_unknown_field__"] = injection.value
        else:
            return raw
        self.matched.add(injection.injection_id)
        self.messages[injection.injection_id] = injection.kind
        return mutated

    def apply_backfill(self, event: dict[str, Any]) -> tuple[dict[str, Any], Injection | None]:
        injection = self._find(event.get("event_id"), "backfill", "backfill_before_cursor")
        if injection is None:
            return event, None
        mutated = copy.deepcopy(event)
        mutated["timestamp"] = 1
        self.matched.add(injection.injection_id)
        self.messages[injection.injection_id] = injection.kind
        return mutated, injection

    def inject_label(self, raw_labels: Any) -> Any:
        injection = self._find(None, "label", "missing_label_feature")
        if injection is None or not isinstance(raw_labels, dict):
            return raw_labels
        mutated = copy.deepcopy(raw_labels)
        missing = injection.feature or "__injected_missing_feature__"
        feature_names = mutated.get("feature_names")
        if not isinstance(feature_names, list):
            feature_names = []
        if missing not in feature_names:
            feature_names.append(missing)
        mutated["feature_names"] = feature_names
        condition = {"feature": missing, "operator": "ge", "value": 0}
        positive = mutated.get("positive_rule")
        if isinstance(positive, dict) and set(positive) == {"all"}:
            positive["all"].append(condition)
        else:
            mutated["positive_rule"] = {"all": [positive or {"else": True}, condition]}
        self.matched.add(injection.injection_id)
        self.messages[injection.injection_id] = injection.kind
        return mutated

    def report(self, failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for injection in sorted(self.injections, key=lambda item: item.injection_id):
            expected = EXPECTED_CODES[injection.kind]
            matching = [
                item
                for item in failures
                if item.get("code") == expected
                and (
                    injection.event_id is None
                    or item.get("event_id") == injection.event_id
                    or item.get("injection_id") == injection.injection_id
                )
            ]
            reports.append(
                {
                    "injection_id": injection.injection_id,
                    "event_id": injection.event_id,
                    "stage": injection.stage,
                    "kind": injection.kind,
                    "matched": injection.injection_id in self.matched,
                    "hit": injection.injection_id in self.matched and bool(matching),
                    "business_code": matching[0]["code"] if matching else expected,
                }
            )
        return reports

    def _find(self, event_id: Any, stage: str, kind: str | None = None) -> Injection | None:
        for injection in self.injections:
            if injection.stage != stage:
                continue
            if kind is not None and injection.kind != kind:
                continue
            if injection.event_id is not None and injection.event_id != event_id:
                continue
            return injection
        return None


def load_injections(
    path: Path | None, load_json: Callable[[Path, str], tuple[Any | None, list[dict[str, Any]]]]
) -> tuple[InjectionTracker, list[dict[str, Any]]]:
    if path is None:
        return InjectionTracker([]), []
    raw, failures = load_json(path, "injection")
    if failures:
        return InjectionTracker([]), failures
    if not isinstance(raw, dict) or set(raw) - {"injections"}:
        return InjectionTracker([]), [
            failure(
                E_SCHEMA_UNKNOWN_FIELD,
                "injection",
                "<injections>",
                "SchemaError",
                "injection configuration must contain only an injections list",
            )
        ]
    items = raw.get("injections")
    if not isinstance(items, list):
        return InjectionTracker([]), [
            failure(E_EVENT_INVALID, "injection", "<injections>", "TypeError", "injections must be a list")
        ]
    injections: list[Injection] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            failures.append(
                failure(E_EVENT_INVALID, "injection", f"<injection:{index}>", "TypeError", "injection must be an object")
            )
            continue
        for field in sorted(set(item) - INJECTION_FIELDS):
            failures.append(
                failure(
                    E_SCHEMA_UNKNOWN_FIELD,
                    "injection",
                    f"<injection:{index}>",
                    "UnknownFieldError",
                    f"unknown injection field: {field}",
                    field=field,
                )
            )
        injection_id = item.get("id")
        kind = item.get("kind")
        stage = item.get("stage")
        if not isinstance(injection_id, str) or not injection_id:
            failures.append(
                failure(E_EVENT_INVALID, "injection", f"<injection:{index}>", "IdentifierError", "id is required")
            )
            continue
        if injection_id in seen:
            failures.append(
                failure(E_EVENT_INVALID, "injection", f"<injection:{injection_id}>", "DuplicateError", "duplicate id")
            )
            continue
        seen.add(injection_id)
        if kind not in INJECTION_KINDS:
            failures.append(
                failure(
                    E_SCHEMA_UNKNOWN_FIELD,
                    "injection",
                    f"<injection:{injection_id}>",
                    "KindError",
                    f"unknown injection kind: {kind}",
                )
            )
            continue
        expected_stage = {
            "non_finite_context": "parse",
            "unknown_event_type": "parse",
            "unknown_event_field": "parse",
            "backfill_before_cursor": "backfill",
            "missing_label_feature": "label",
        }[kind]
        event_id = item.get("event_id")
        if kind != "missing_label_feature" and (not isinstance(event_id, str) or not event_id):
            failures.append(
                failure(
                    E_EVENT_INVALID,
                    "injection",
                    f"<injection:{injection_id}>",
                    "IdentifierError",
                    "event_id is required for this injection",
                )
            )
            continue
        if stage != expected_stage:
            failures.append(
                failure(
                    E_SCHEMA_UNKNOWN_FIELD,
                    "injection",
                    f"<injection:{injection_id}>",
                    "StageError",
                    f"{kind} must use stage {expected_stage}",
                )
            )
            continue
        injections.append(
            Injection(
                injection_id=injection_id,
                event_id=event_id,
                stage=stage,
                kind=kind,
                field=item.get("field"),
                value=item.get("value"),
                feature=item.get("feature"),
            )
        )
    return InjectionTracker(injections), failures
