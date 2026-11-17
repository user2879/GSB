"""Validated, immutable pipeline definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .canonical import sha256_value


@dataclass(frozen=True)
class EventFilter:
    field: str
    equals: Any


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    entity: str
    window_ms: int
    aggregation: str
    source_field: str | None
    source_feature: str | None
    upstream_features: tuple[str, ...]
    empty_value: int | float
    event_filter: EventFilter | None = None

    def normalized(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "entity": self.entity,
            "window_ms": self.window_ms,
            "aggregation": self.aggregation,
            "upstream_features": list(self.upstream_features),
            "empty_value": self.empty_value,
        }
        if self.source_field is not None:
            result["source_field"] = self.source_field
        if self.source_feature is not None:
            result["source_feature"] = self.source_feature
        if self.event_filter is not None:
            result["filter"] = {
                "field": self.event_filter.field,
                "equals": self.event_filter.equals,
            }
        return result

    @property
    def fingerprint(self) -> str:
        return sha256_value({"kind": "feature-definition-v1", **self.normalized()})

    @property
    def aggregation_fingerprint(self) -> str:
        return sha256_value(
            {
                "kind": "aggregation-v1",
                "aggregation": self.aggregation,
                "window_ms": self.window_ms,
            }
        )


@dataclass(frozen=True)
class FeatureTable:
    definitions: dict[str, FeatureDefinition]
    topological_order: tuple[str, ...]
    levels: tuple[tuple[str, ...], ...]
    positions: dict[str, int]
    raw: dict[str, Any]

    @classmethod
    def empty(cls) -> "FeatureTable":
        return cls({}, (), (), {}, {"features": []})

    def descendants(self, roots: set[str] | list[str]) -> set[str]:
        reverse: dict[str, set[str]] = {name: set() for name in self.definitions}
        for definition in self.definitions.values():
            for upstream in definition.upstream_features:
                reverse.setdefault(upstream, set()).add(definition.name)
        affected = set(roots)
        stack = list(roots)
        while stack:
            current = stack.pop()
            for child in reverse.get(current, ()):
                if child not in affected:
                    affected.add(child)
                    stack.append(child)
        return affected

    def normalized_raw(self) -> dict[str, Any]:
        return {"features": [self.definitions[name].normalized() for name in sorted(self.definitions)]}


@dataclass(frozen=True)
class LabelDefinition:
    name: str
    trigger_event_types: tuple[str, ...]
    delay_ms: int
    feature_names: tuple[str, ...]
    positive_rule: dict[str, Any]
    negative_rule: dict[str, Any]
    raw: dict[str, Any]

    def normalized(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trigger_event_types": list(self.trigger_event_types),
            "delay_ms": self.delay_ms,
            "feature_names": list(self.feature_names),
            "positive_rule": self.positive_rule,
            "negative_rule": self.negative_rule,
        }


@dataclass(frozen=True)
class PipelineConfig:
    allowed_event_types: tuple[str, ...]
    max_workers: int = 4
    stage_delay_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> dict[str, Any]:
        return {
            "allowed_event_types": list(self.allowed_event_types),
            "max_workers": self.max_workers,
            "stage_delay_ms": self.stage_delay_ms,
        }


def definition_hash(features: FeatureTable, labels: LabelDefinition | dict[str, Any]) -> str:
    label_value = labels.normalized() if isinstance(labels, LabelDefinition) else labels
    return sha256_value(
        {
            "kind": "pipeline-definitions-v1",
            "features": features.normalized_raw(),
            "labels": label_value,
        }
    )

