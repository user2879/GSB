from __future__ import annotations

from typing import Any

from .concurrency import ParallelRunner
from .fingerprint import digest, labelled_digest
from .fixed import decimal_ratio_fixed, format_fixed, ratio_decimal
from .metrics import passes_filter, rank_weight
from .models import Config, Exposure, SOURCE_ORDER, Spec, changed_spec_sources


def _dedup_dropped_refs(events: list[Exposure], spec: Spec) -> set[str]:
    eligible = [
        event
        for event in sorted(events, key=lambda item: (item.request_id, item.item_id))
        if passes_filter(event, spec) and event.rank <= spec.truncation_rank
    ]
    dropped: set[str] = set()
    if not spec.dedup_count:
        return dropped
    seen_users: set[str] = set()
    for event in eligible:
        if event.user_id in seen_users:
            dropped.add(event.stable_id)
        else:
            seen_users.add(event.user_id)
    return dropped


def _source_refs(
    source: str,
    before: Spec,
    after: Spec,
    events: list[Exposure],
    dedup_before: set[str],
    dedup_after: set[str],
) -> list[str]:
    refs: list[str] = []
    for event in sorted(events, key=lambda item: (item.request_id, item.item_id)):
        if source == "truncation_rank":
            if (event.rank <= before.truncation_rank) != (event.rank <= after.truncation_rank):
                refs.append(event.stable_id)
        elif source == "normalization":
            if rank_weight(event.rank, before.normalization) != rank_weight(event.rank, after.normalization):
                refs.append(event.stable_id)
        elif source == "dedup_count":
            if (event.stable_id in dedup_before) != (event.stable_id in dedup_after):
                refs.append(event.stable_id)
        elif source == "sample_filter":
            if passes_filter(event, before) != passes_filter(event, after):
                refs.append(event.stable_id)
        else:
            raise ValueError(f"unknown source {source!r}")
    return refs


def drift_cache_key(
    from_version: str,
    to_version: str,
    group: str,
    from_metric_key: str,
    to_metric_key: str,
    from_spec: Spec,
    to_spec: Spec,
    config: Config,
) -> str:
    return "drift:" + labelled_digest(
        "drift-v1",
        {
            "from_version": from_version,
            "to_version": to_version,
            "group": group,
            "from_metric_key": from_metric_key,
            "to_metric_key": to_metric_key,
            "from_spec_fingerprint": from_spec.fingerprint(),
            "to_spec_fingerprint": to_spec.fingerprint(),
            "drift_tolerance": str(config.drift_tolerance),
            "min_affected_ratio": str(config.min_affected_ratio),
            "fixed_decimal_places": config.fixed_decimal_places,
            "quantization_step": config.quantization_step,
        },
    )


def compute_drift_group(
    events: list[Exposure],
    group: str,
    from_spec: Spec,
    to_spec: Spec,
    from_metric: dict[str, Any],
    to_metric: dict[str, Any],
    config: Config,
    runner: ParallelRunner,
    *,
    cache_key: str,
) -> dict[str, Any]:
    dedup_before = _dedup_dropped_refs(events, from_spec)
    dedup_after = _dedup_dropped_refs(events, to_spec)
    changed_sources = set(changed_spec_sources(from_spec, to_spec))

    def delta_task() -> dict[str, Any]:
        return {"delta_fixed": int(to_metric["value_fixed"]) - int(from_metric["value_fixed"])}

    def source_task(source: str) -> dict[str, Any]:
        changed = source in changed_sources
        refs = (
            _source_refs(source, from_spec, to_spec, events, dedup_before, dedup_after)
            if changed
            else []
        )
        return {
            "source": source,
            "changed": changed,
            "impacted_count": len(refs),
            "event_refs": refs,
            "event_refs_fingerprint": digest(refs),
        }

    tasks = [(f"drift|{from_spec.version}>{to_spec.version}|{group}|__delta__", delta_task)]
    tasks.extend(
        (
            f"drift|{from_spec.version}>{to_spec.version}|{group}|{source}",
            lambda name=source: source_task(name),
        )
        for source in SOURCE_ORDER
    )
    results = runner.map_tasks(tasks)
    delta_fixed = results[0]["delta_fixed"]
    source_rows = results[1:]
    by_source = {row["source"]: row for row in source_rows}
    affected_refs = sorted(
        {
            ref
            for source in SOURCE_ORDER
            for ref in by_source[source]["event_refs"]
        }
    )
    total_events = max(1, len(events))
    tolerance_fixed = decimal_ratio_fixed(config.drift_tolerance, config.fixed_decimal_places)
    min_ratio_fixed = decimal_ratio_fixed(config.min_affected_ratio, config.fixed_decimal_places)
    affected_ratio_fixed = (len(affected_refs) * config.scale + total_events // 2) // total_events
    boundary = abs(delta_fixed) == tolerance_fixed
    significant = (
        not boundary
        and abs(delta_fixed) > tolerance_fixed
        and affected_ratio_fixed >= min_ratio_fixed
    )
    source_report = {}
    for source in config.attribution_order:
        row = dict(by_source[source])
        row["impacted_ratio"] = ratio_decimal(
            row["impacted_count"], total_events, config.fixed_decimal_places
        )
        source_report[source] = row
    return {
        "stage": "drift",
        "cache_key": cache_key,
        "group": group,
        "from_version": from_spec.version,
        "to_version": to_spec.version,
        "from_spec_fingerprint": from_spec.fingerprint(),
        "to_spec_fingerprint": to_spec.fingerprint(),
        "from_value_fixed": from_metric["value_fixed"],
        "from_value": from_metric["value"],
        "to_value_fixed": to_metric["value_fixed"],
        "to_value": to_metric["value"],
        "delta_fixed": delta_fixed,
        "delta": format_fixed(delta_fixed, config.fixed_decimal_places),
        "tolerance_fixed": tolerance_fixed,
        "tolerance": str(config.drift_tolerance),
        "min_affected_ratio": str(config.min_affected_ratio),
        "affected_count": len(affected_refs),
        "affected_ratio": ratio_decimal(len(affected_refs), total_events, config.fixed_decimal_places),
        "affected_ratio_fixed": affected_ratio_fixed,
        "affected_event_refs": affected_refs,
        "significant": significant,
        "boundary": boundary,
        "boundary_samples": affected_refs if boundary else [],
        "sources": source_report,
        "changed_sources": [source for source in SOURCE_ORDER if source in changed_sources],
        "group_event_count": len(events),
        "upstream": {
            "from_metric_cache_key": from_metric["cache_key"],
            "to_metric_cache_key": to_metric["cache_key"],
        },
    }


def drift_to_report(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "group": artifact["group"],
        "from_version": artifact["from_version"],
        "to_version": artifact["to_version"],
        "from_value": artifact["from_value"],
        "to_value": artifact["to_value"],
        "delta_fixed": artifact["delta_fixed"],
        "delta": artifact["delta"],
        "tolerance_fixed": artifact["tolerance_fixed"],
        "tolerance": artifact["tolerance"],
        "significant": artifact["significant"],
        "boundary": artifact["boundary"],
        "boundary_samples": artifact["boundary_samples"],
        "affected_count": artifact["affected_count"],
        "affected_ratio": artifact["affected_ratio"],
        "changed_sources": artifact["changed_sources"],
        "sources": artifact["sources"],
    }
