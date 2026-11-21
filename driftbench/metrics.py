from __future__ import annotations

import math
from typing import Any

from .concurrency import ParallelRunner
from .fingerprint import digest, labelled_digest, ordered_set_digest
from .fixed import format_fixed, quantize_ratio
from .models import Config, Exposure, FilterCondition, Spec


def _coerce_pair(left: Any, right: Any) -> tuple[Any, Any]:
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left), bool(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left), float(right)
    return left, right


def condition_matches(event: Exposure, condition: FilterCondition) -> bool:
    actual = getattr(event, condition.field, None)
    expected = condition.value
    op = condition.op
    if op in {"in", "not_in"}:
        if not isinstance(expected, (list, tuple, set)):
            return False
        if actual in expected:
            return op == "in"
        return op == "not_in"
    left, right = _coerce_pair(actual, expected)
    try:
        if op == "eq":
            return left == right
        if op == "ne":
            return left != right
        if op == "gt":
            return left > right
        if op == "gte":
            return left >= right
        if op == "lt":
            return left < right
        if op == "lte":
            return left <= right
    except TypeError:
        return False
    return False


def passes_filter(event: Exposure, spec: Spec) -> bool:
    return all(condition_matches(event, condition) for condition in spec.sample_filter)


def rank_weight(rank: int, normalization: str) -> int:
    if normalization == "none":
        return 1
    if normalization == "rank_inverse":
        return 1_000_000 // rank
    if normalization == "position_decay":
        return int(1_000_000 / math.log2(rank + 1))
    raise ValueError(f"unsupported normalization {normalization!r}")


def spec_metric_policy(spec: Spec, config: Config) -> dict[str, Any]:
    return {
        "spec": spec.canonical_dict(),
        "metric_name": spec.metric_name,
        "fixed_decimal_places": config.fixed_decimal_places,
        "quantization_step": config.quantization_step,
    }


def metric_config_fingerprint(config: Config) -> str:
    return labelled_digest(
        "metric-config-v1",
        {
            "control_group": config.control_group,
            "experiment_group": config.experiment_group,
            "fixed_decimal_places": config.fixed_decimal_places,
            "quantization_step": config.quantization_step,
        },
    )


def metric_cache_key(
    version: str,
    group: str,
    events_fingerprint: str,
    spec: Spec,
    config: Config,
) -> str:
    return "metrics:" + labelled_digest(
        "metrics-v1",
        {
            "version": version,
            "group": group,
            "events_fingerprint": events_fingerprint,
            "spec_fingerprint": spec.fingerprint(),
            "config_policy_fingerprint": metric_config_fingerprint(config),
            "policy": spec_metric_policy(spec, config),
        },
    )


def _event_probe(event: Exposure, spec: Spec) -> dict[str, Any]:
    return {
        "stable_id": event.stable_id,
        "request_id": event.request_id,
        "user_id": event.user_id,
        "item_id": event.item_id,
        "rank": event.rank,
        "filtered": passes_filter(event, spec),
        "truncated": event.rank <= spec.truncation_rank,
        "weight": rank_weight(event.rank, spec.normalization),
        "clicked": event.clicked,
        "converted": event.converted,
    }


def compute_metric(
    events: list[Exposure],
    group: str,
    spec: Spec,
    config: Config,
    runner: ParallelRunner,
    *,
    cache_key: str,
    upstream: dict[str, str] | None = None,
) -> dict[str, Any]:
    ordered_events = sorted(events, key=lambda item: (item.request_id, item.item_id))
    probes = runner.map_tasks(
        [
            (f"metrics|{spec.version}|{group}|{event.request_id}|{event.item_id}", lambda e=event: _event_probe(e, spec))
            for event in ordered_events
        ]
    )
    eligible = [probe for probe in probes if probe["filtered"] and probe["truncated"]]
    eligible.sort(key=lambda item: (item["request_id"], item["item_id"]))
    dedup_removed: list[str] = []
    if spec.dedup_count:
        retained: list[dict[str, Any]] = []
        seen_users: set[str] = set()
        for probe in eligible:
            if probe["user_id"] in seen_users:
                dedup_removed.append(probe["stable_id"])
            else:
                seen_users.add(probe["user_id"])
                retained.append(probe)
        participants = retained
    else:
        participants = eligible

    numerator_weight = 0
    denominator_weight = 0
    for probe in participants:
        weight = probe["weight"]
        if spec.metric_name == "ctr":
            denominator_weight += weight
            if probe["clicked"] == 1:
                numerator_weight += weight
        elif spec.metric_name == "cvr":
            if probe["clicked"] == 1:
                denominator_weight += weight
                if probe["converted"] == 1:
                    numerator_weight += weight
        else:
            raise ValueError(f"unsupported metric {spec.metric_name!r}")

    value_fixed = quantize_ratio(
        numerator_weight,
        denominator_weight,
        config.fixed_decimal_places,
        config.quantization_step,
    )
    event_refs = sorted(probe["stable_id"] for probe in participants)
    request_refs = sorted(probe["request_id"] for probe in participants)
    return {
        "stage": "metrics",
        "cache_key": cache_key,
        "group": group,
        "spec_version": spec.version,
        "metric_name": spec.metric_name,
        "spec_fingerprint": spec.fingerprint(),
        "config_fingerprint": metric_config_fingerprint(config),
        "upstream": dict(upstream or {}),
        "event_count": len(events),
        "eligible_before_dedup_count": len(eligible),
        "participant_count": len(participants),
        "dedup_removed_count": len(dedup_removed),
        "dedup_removed_event_refs": dedup_removed,
        "numerator_weight": numerator_weight,
        "denominator_weight": denominator_weight,
        "value_fixed": value_fixed,
        "value": format_fixed(value_fixed, config.fixed_decimal_places),
        "scale": config.scale,
        "quantization_step": config.quantization_step,
        "event_refs": event_refs,
        "request_set_fingerprint": ordered_set_digest(request_refs),
        "content_fingerprint": digest(
            {
                "group": group,
                "spec": spec.canonical_dict(),
                "events": [event.content_dict() for event in ordered_events],
            }
        ),
    }


def compute_metric_for_spec_without_cache(
    events: list[Exposure],
    group: str,
    spec: Spec,
    config: Config,
    runner: ParallelRunner,
) -> dict[str, Any]:
    return compute_metric(
        events,
        group,
        spec,
        config,
        runner,
        cache_key="attribution-intermediate:" + spec.fingerprint()[:16],
        upstream={"purpose": "attribution_intermediate"},
    )
