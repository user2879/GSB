"""In-memory domain models for events, feature definitions and label rules."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .common import AGG_FUNCTIONS, SIDES


@dataclass(frozen=True)
class Event:
    event_id: str
    user_id: str
    item_id: str
    behavior: str
    ts_ms: int
    context: dict[str, float]

    def with_correction(self, behavior: str | None = None,
                        ts_ms: int | None = None,
                        context: dict[str, float] | None = None) -> "Event":
        return replace(
            self,
            behavior=behavior if behavior is not None else self.behavior,
            ts_ms=ts_ms if ts_ms is not None else self.ts_ms,
            context=context if context is not None else self.context,
        )


@dataclass
class FeatureDef:
    name: str
    side: str                    # user | item
    window_ms: int
    agg: str                     # count | sum | mean | last
    source_field: str | None     # raw event field for level-0 features
    upstream: list[str]          # upstream feature names (derived features)
    empty_value: int | None      # configured raw-space empty-window value
    valid: bool = True
    invalid_reason: str = ""
    # populated during DAG build
    topo_rank: int = -1
    level: int = -1

    @property
    def is_derived(self) -> bool:
        return bool(self.upstream)


@dataclass
class LabelDef:
    name: str
    trigger_behaviors: list[str]
    decision_feature: str
    threshold_raw: str           # kept as string -> Decimal to stay exact
    positive_op: str             # ge | gt | le | lt | eq
    observed_features: list[str]
    delay_ms: int
    valid: bool = True
    invalid_reason: str = ""


@dataclass
class InjectionRule:
    injection_id: str
    event_id: str
    stage: str                   # load | feature | sample
    kind: str                    # non_finite_context | unknown_behavior | backfill_order
    context_field: str = ""
    bad_ts_ms: int = -1


def validate_agg(agg: str) -> bool:
    return agg in AGG_FUNCTIONS


def validate_side(side: str) -> bool:
    return side in SIDES
