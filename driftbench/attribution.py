from __future__ import annotations

import itertools
from typing import Any, Callable

from .concurrency import ParallelRunner
from .fingerprint import labelled_digest
from .fixed import decimal_ratio_fixed, format_fixed
from .metrics import compute_metric
from .models import Config, Exposure, SOURCE_ORDER, Spec, changed_spec_sources


def _replace_source(spec: Spec, source: str, target: Spec, values: dict[str, Any]) -> Spec:
    return Spec(
        version="__attrib__",
        metric_name=spec.metric_name,
        truncation_rank=target.truncation_rank if source == "truncation_rank" else values["truncation_rank"],
        normalization=target.normalization if source == "normalization" else values["normalization"],
        dedup_count=target.dedup_count if source == "dedup_count" else values["dedup_count"],
        sample_filter=target.sample_filter if source == "sample_filter" else values["sample_filter"],
    )


def attribution_cache_key(
    from_version: str,
    to_version: str,
    group: str,
    drift_key: str,
    config: Config,
) -> str:
    return "attribution:" + labelled_digest(
        "attribution-v1",
        {
            "from_version": from_version,
            "to_version": to_version,
            "group": group,
            "drift_key": drift_key,
            "order": list(config.attribution_order),
            "drift_tolerance": str(config.drift_tolerance),
            "fixed_decimal_places": config.fixed_decimal_places,
            "quantization_step": config.quantization_step,
        },
    )


def compute_attribution_group(
    events: list[Exposure],
    group: str,
    from_spec: Spec,
    to_spec: Spec,
    from_metric: dict[str, Any],
    to_metric: dict[str, Any],
    drift_artifact: dict[str, Any],
    config: Config,
    runner: ParallelRunner,
    *,
    cache_key: str,
) -> dict[str, Any]:
    changed = list(changed_spec_sources(from_spec, to_spec))
    metric_cache: dict[str, dict[str, Any]] = {}

    def values_for(spec: Spec) -> dict[str, Any]:
        return {
            "truncation_rank": spec.truncation_rank,
            "normalization": spec.normalization,
            "dedup_count": spec.dedup_count,
            "sample_filter": spec.sample_filter,
        }

    def evaluate(spec: Spec) -> dict[str, Any]:
        fingerprint = spec.fingerprint()
        if fingerprint not in metric_cache:
            metric = compute_metric(
                events,
                group,
                spec,
                config,
                runner,
                cache_key="attribution-intermediate:" + fingerprint[:24],
                upstream={"attribution_stage_key": cache_key},
            )
            metric_cache[fingerprint] = metric
        return metric_cache[fingerprint]

    def path(permutation: tuple[str, ...]) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
        current_values = values_for(from_spec)
        current_metric = from_metric
        contributions: dict[str, int] = {}
        steps: list[dict[str, Any]] = [
            {
                "step": 0,
                "source": "__from__",
                "value_fixed": from_metric["value_fixed"],
                "value": from_metric["value"],
                "numerator_weight": from_metric["numerator_weight"],
                "denominator_weight": from_metric["denominator_weight"],
                "participant_count": from_metric["participant_count"],
            }
        ]
        rows: list[dict[str, Any]] = []
        for index, source in enumerate(permutation, 1):
            next_spec = _replace_source(from_spec, source, to_spec, current_values)
            next_metric = evaluate(next_spec)
            contribution = int(next_metric["value_fixed"]) - int(current_metric["value_fixed"])
            contributions[source] = contribution
            rows.append(
                {
                    "source": source,
                    "before_fixed": current_metric["value_fixed"],
                    "before": current_metric["value"],
                    "after_fixed": next_metric["value_fixed"],
                    "after": next_metric["value"],
                    "contribution_fixed": contribution,
                    "contribution": format_fixed(contribution, config.fixed_decimal_places),
                }
            )
            steps.append(
                {
                    "step": index,
                    "source": source,
                    "value_fixed": next_metric["value_fixed"],
                    "value": next_metric["value"],
                    "numerator_weight": next_metric["numerator_weight"],
                    "denominator_weight": next_metric["denominator_weight"],
                    "participant_count": next_metric["participant_count"],
                }
            )
            current_values = values_for(next_spec)
            current_metric = next_metric
        return rows, contributions, steps

    chosen_rows, chosen_contributions, chosen_steps = path(config.attribution_order)
    target_check = evaluate(to_spec)
    total_delta = int(to_metric["value_fixed"]) - int(from_metric["value_fixed"])
    residual = sum(chosen_contributions.get(source, 0) for source in SOURCE_ORDER) - total_delta
    tolerance_fixed = decimal_ratio_fixed(config.drift_tolerance, config.fixed_decimal_places)

    permutation_contributions: dict[str, list[int]] = {source: [] for source in changed}
    permutations_checked: list[list[str]] = []
    if changed:
        for permutation in itertools.permutations(changed):
            _, contributions, _ = path(permutation)
            permutations_checked.append(list(permutation))
            for source in changed:
                permutation_contributions[source].append(contributions[source])
    else:
        permutations_checked.append([])

    sensitivity: dict[str, Any] = {}
    ambiguous: list[str] = []
    for source in changed:
        values = permutation_contributions[source] or [0]
        low, high = min(values), max(values)
        is_sensitive = high - low > tolerance_fixed
        sensitivity[source] = {
            "min_contribution_fixed": low,
            "max_contribution_fixed": high,
            "range_fixed": high - low,
            "order_sensitive": is_sensitive,
        }
        if is_sensitive:
            ambiguous.append(source)

    source_event_refs = {
        source: drift_artifact["sources"][source]["event_refs"] for source in SOURCE_ORDER
    }
    contributions_report = []
    for source in SOURCE_ORDER:
        row = next(item for item in chosen_rows if item["source"] == source) if any(
            item["source"] == source for item in chosen_rows
        ) else {
            "source": source,
            "before_fixed": from_metric["value_fixed"],
            "before": from_metric["value"],
            "after_fixed": from_metric["value_fixed"],
            "after": from_metric["value"],
            "contribution_fixed": 0,
            "contribution": format_fixed(0, config.fixed_decimal_places),
        }
        row = dict(row)
        row["changed"] = source in changed
        row["event_refs"] = source_event_refs[source]
        row["event_refs_fingerprint"] = drift_artifact["sources"][source]["event_refs_fingerprint"]
        contributions_report.append(row)

    # The configured order is a full permutation; unchanged sources contribute zero.
    chosen_full = {
        source: chosen_contributions.get(source, 0) for source in SOURCE_ORDER
    }
    contribution_sum = sum(chosen_full.values())
    return {
        "stage": "attribution",
        "cache_key": cache_key,
        "group": group,
        "from_version": from_spec.version,
        "to_version": to_spec.version,
        "order": list(config.attribution_order),
        "changed_sources": changed,
        "contributions": contributions_report,
        "steps": chosen_steps,
        "from_value_fixed": from_metric["value_fixed"],
        "from_value": from_metric["value"],
        "to_value_fixed": to_metric["value_fixed"],
        "to_value": to_metric["value"],
        "total_delta_fixed": total_delta,
        "total_delta": format_fixed(total_delta, config.fixed_decimal_places),
        "contribution_sum_fixed": contribution_sum,
        "contribution_sum": format_fixed(contribution_sum, config.fixed_decimal_places),
        "residual_fixed": residual,
        "tolerance_fixed": tolerance_fixed,
        "within_tolerance": abs(residual) <= tolerance_fixed,
        "target_check_value_fixed": target_check["value_fixed"],
        "target_check_value": target_check["value"],
        "target_check_ok": target_check["value_fixed"] == to_metric["value_fixed"],
        "order_sensitivity": sensitivity,
        "order_sensitive_sources": ambiguous,
        "resolved_by_order": bool(ambiguous),
        "permutations_checked": permutations_checked,
        "source_event_refs": source_event_refs,
        "upstream_drift_cache_key": drift_artifact["cache_key"],
    }


def attribution_to_report(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "group": artifact["group"],
        "from_version": artifact["from_version"],
        "to_version": artifact["to_version"],
        "order": artifact["order"],
        "changed_sources": artifact["changed_sources"],
        "contributions": artifact["contributions"],
        "steps": artifact["steps"],
        "from_value": artifact["from_value"],
        "to_value": artifact["to_value"],
        "total_delta_fixed": artifact["total_delta_fixed"],
        "total_delta": artifact["total_delta"],
        "contribution_sum_fixed": artifact["contribution_sum_fixed"],
        "contribution_sum": artifact["contribution_sum"],
        "residual_fixed": artifact["residual_fixed"],
        "within_tolerance": artifact["within_tolerance"],
        "target_check_ok": artifact["target_check_ok"],
        "order_sensitive_sources": artifact["order_sensitive_sources"],
        "resolved_by_order": artifact["resolved_by_order"],
        "order_sensitivity": artifact["order_sensitivity"],
        "permutations_checked": artifact["permutations_checked"],
        "source_event_refs": artifact["source_event_refs"],
    }
