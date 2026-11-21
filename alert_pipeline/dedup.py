from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping

from .models import AggregateAlert, NormalizedAlert, build_alert_id


DedupKey = tuple[str, str]
DedupState = dict[DedupKey, tuple[AggregateAlert, ...]]


def _source_record(alert: NormalizedAlert) -> dict:
    return alert.to_ledger_source()


def _new_aggregate(alert: NormalizedAlert) -> AggregateAlert:
    return AggregateAlert(
        alert_id=build_alert_id(alert.service_id, alert.fingerprint, alert.timestamp_ms),
        service_id=alert.service_id,
        level=alert.level,
        fingerprint=alert.fingerprint,
        tags=dict(sorted(alert.tags.items())),
        first_timestamp_ms=alert.timestamp_ms,
        last_timestamp_ms=alert.timestamp_ms,
        suppressed_count=0,
        total_count=1,
        participant_lines=(alert.line_number,),
        silence_window_ms=alert.silence_window_ms,
        window_version=(
            f"v{alert.format_version}:{alert.silence_window_ms}:{alert.strategy_id}"
            f"@decl{alert.declared_version}"
        ),
        strategy_ids=(alert.strategy_id,),
        source_format_versions=(alert.format_version,),
        source_records=(_source_record(alert),),
        declared_format_version=alert.format_version,
        declared_version=alert.declared_version,
    )


def _extend_aggregate(aggregate: AggregateAlert, alert: NormalizedAlert) -> AggregateAlert:
    lines = tuple(sorted(set(aggregate.participant_lines + (alert.line_number,))))
    strategies = tuple(sorted(set(aggregate.strategy_ids + (alert.strategy_id,))))
    versions = tuple(sorted(set(aggregate.source_format_versions + (alert.format_version,))))
    records = tuple(
        sorted(
            aggregate.source_records + (_source_record(alert),),
            key=lambda item: (item["timestamp_ms"], item["line"]),
        )
    )
    return replace(
        aggregate,
        last_timestamp_ms=max(aggregate.last_timestamp_ms, alert.timestamp_ms),
        suppressed_count=aggregate.suppressed_count + 1,
        total_count=aggregate.total_count + 1,
        participant_lines=lines,
        strategy_ids=strategies,
        source_format_versions=versions,
        source_records=records,
        declared_format_version=max(
            aggregate.declared_format_version, alert.format_version
        ),
    )


def process_group(
    alerts: Iterable[NormalizedAlert], existing: tuple[AggregateAlert, ...] = tuple()
) -> tuple[AggregateAlert, ...]:
    ordered = sorted(alerts, key=lambda item: (item.timestamp_ms, item.line_number))
    completed = list(existing)
    for alert in ordered:
        if not completed:
            completed.append(_new_aggregate(alert))
            continue
        current = completed[-1]
        within_window = (
            alert.timestamp_ms - current.first_timestamp_ms < current.silence_window_ms
        )
        if within_window:
            completed[-1] = _extend_aggregate(current, alert)
        else:
            completed.append(_new_aggregate(alert))
    return tuple(completed)


def group_alerts(alerts: Iterable[NormalizedAlert]) -> dict[DedupKey, tuple[NormalizedAlert, ...]]:
    grouped: dict[DedupKey, list[NormalizedAlert]] = {}
    for alert in alerts:
        key = (alert.service_id, alert.fingerprint)
        grouped.setdefault(key, []).append(alert)
    return {
        key: tuple(sorted(items, key=lambda item: (item.timestamp_ms, item.line_number)))
        for key, items in sorted(grouped.items())
    }


def merge_group_results(
    state: Mapping[DedupKey, tuple[AggregateAlert, ...]],
    updates: Mapping[DedupKey, tuple[AggregateAlert, ...]],
) -> DedupState:
    result = {key: value for key, value in state.items()}
    for key in sorted(updates):
        result[key] = updates[key]
    return result


def flatten_state(state: Mapping[DedupKey, tuple[AggregateAlert, ...]]) -> tuple[AggregateAlert, ...]:
    aggregates = [aggregate for _, items in sorted(state.items()) for aggregate in items]
    return tuple(
        sorted(
            aggregates,
            key=lambda item: (
                item.first_timestamp_ms,
                item.service_id,
                item.fingerprint,
                item.alert_id,
            ),
        )
    )


def deduplicate_serial(alerts: Iterable[NormalizedAlert]) -> tuple[AggregateAlert, ...]:
    state: DedupState = {}
    for key, items in group_alerts(alerts).items():
        state[key] = process_group(items)
    return flatten_state(state)
