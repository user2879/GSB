from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .canonical import decimal_text


@dataclass(frozen=True)
class FeatureDef:
    name: str
    side: str
    source_field: str | None
    window_ms: int
    aggregate: str
    decimal_places: int
    quantum: Decimal
    upstream_features: tuple[str, ...]
    parameter_fingerprint: str
    raw: dict[str, Any] = field(compare=False, hash=False)

    @property
    def sides(self) -> tuple[str, ...]:
        if self.side == "both":
            return ("item", "user")
        return (self.side,)

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "side": self.side,
            "source_field": self.source_field,
            "window_ms": self.window_ms,
            "aggregate": self.aggregate,
            "decimal_places": self.decimal_places,
            "quantization_step": decimal_text(self.quantum),
            "upstream_features": list(self.upstream_features),
            "parameter_fingerprint": self.parameter_fingerprint,
        }


@dataclass(frozen=True)
class Event:
    event_id: str
    user_id: str
    item_id: str
    behavior: str
    timestamp_ms: int
    context: dict[str, float | int]
    content_fingerprint: str
    injection_id: str | None = None

    def public(self, include_injection: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "item_id": self.item_id,
            "behavior": self.behavior,
            "timestamp_ms": self.timestamp_ms,
            "context": dict(self.context),
        }
        if include_injection and self.injection_id:
            payload["_injection_id"] = self.injection_id
        return payload


@dataclass(frozen=True)
class ScoreConfig:
    version: int
    weights: dict[str, Decimal]
    fusion_temperature: Decimal
    fusion_bias: Decimal
    clamp_min: Decimal
    clamp_max: Decimal
    concurrency: int
    point_budget_per_batch: int
    snapshot_commit_interval: int
    score_decimal_places: int
    score_quantum: Decimal
    allowed_behaviors: tuple[str, ...]
    config_fingerprint: str
    scoring_fingerprint: str
    raw: dict[str, Any] = field(compare=False, hash=False)


@dataclass(frozen=True)
class FeatureValue:
    feature_name: str
    side: str
    entity_id: str
    value_fixed: int
    decimal_places: int
    quantization_step: str
    event_ids: tuple[str, ...]
    event_window_fingerprint: str
    parameter_fingerprint: str
    upstream_fingerprints: tuple[tuple[str, str], ...]
    cache_key: str
    content_fingerprint: str
    stale: bool = False

    @property
    def value_decimal(self) -> Decimal:
        return Decimal(self.value_fixed).scaleb(-self.decimal_places)

    def public(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "side": self.side,
            "entity_id": self.entity_id,
            "value_fixed": self.value_fixed,
            "decimal_places": self.decimal_places,
            "quantization_step": self.quantization_step,
            "value_decimal": decimal_text(self.value_decimal),
            "event_ids": list(self.event_ids),
            "event_window_fingerprint": self.event_window_fingerprint,
            "parameter_fingerprint": self.parameter_fingerprint,
            "upstream_fingerprints": [list(item) for item in self.upstream_fingerprints],
            "cache_key": self.cache_key,
            "content_fingerprint": self.content_fingerprint,
            "stale": self.stale,
        }


@dataclass(frozen=True)
class ScoreContribution:
    feature_name: str
    feature_value_fixed: int
    feature_decimal_places: int
    feature_value_decimal: str
    feature_content_fingerprint: str
    weight: str
    contribution_decimal: str
    contribution_fixed: int

    def public(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "feature_value_fixed": self.feature_value_fixed,
            "feature_decimal_places": self.feature_decimal_places,
            "feature_value_decimal": self.feature_value_decimal,
            "feature_content_fingerprint": self.feature_content_fingerprint,
            "weight": self.weight,
            "contribution_decimal": self.contribution_decimal,
            "contribution_fixed": self.contribution_fixed,
        }


@dataclass(frozen=True)
class ScoreValue:
    entity_type: str
    entity_id: str
    score_fixed: int
    score_decimal_places: int
    score_quantum: str
    score_decimal: str
    bias_contribution: str
    contributions: tuple[ScoreContribution, ...]
    participating_event_count: int
    feature_fingerprints: tuple[tuple[str, str], ...]
    cache_key: str
    content_fingerprint: str
    stale: bool = False

    def public(self, rank: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "score_fixed": self.score_fixed,
            "score_decimal_places": self.score_decimal_places,
            "score_quantum": self.score_quantum,
            "score_decimal": self.score_decimal,
            "bias_contribution": self.bias_contribution,
            "contributions": [item.public() for item in self.contributions],
            "participating_event_count": self.participating_event_count,
            "feature_fingerprints": [list(item) for item in self.feature_fingerprints],
            "cache_key": self.cache_key,
            "content_fingerprint": self.content_fingerprint,
            "stale": self.stale,
        }
        if rank is not None:
            payload["rank"] = rank
        return payload


@dataclass(frozen=True)
class PendingPoint:
    feature_name: str
    side: str
    entity_id: str
    event_ids: tuple[str, ...]
    stage: str = "feature"
    code: str = "E_POINT_BUDGET_PENDING"

    def public(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "side": self.side,
            "entity_id": self.entity_id,
            "event_ids": list(self.event_ids),
            "stage": self.stage,
            "code": self.code,
        }


def failure(
    *,
    code: str,
    stage: str,
    message: str,
    feature_name: str | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    event_id: str | None = None,
    exception_type: str | None = None,
    injection_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": code,
        "stage": stage,
        "message": message,
        "feature_name": feature_name,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "event_id": event_id,
        "exception_type": exception_type,
        "injection_id": injection_id,
    }
    if detail:
        payload["detail"] = detail
    return payload


def failure_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item.get("stage") or "",
        item.get("feature_name") or "",
        item.get("entity_type") or "",
        item.get("entity_id") or "",
        item.get("event_id") or "",
        item.get("code") or "",
        item.get("injection_id") or "",
        item.get("message") or "",
        item.get("exception_type") or "",
    )
