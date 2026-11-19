"""Resource-budget and overflow accounting."""

from __future__ import annotations

from typing import Any

try:
    import numpy as _np
except ImportError:
    _np = None


def _numeric_peaks(values: list[int]) -> tuple[int, int]:
    if not values:
        return 0, 0
    if _np is not None:
        array = _np.asarray(values, dtype=_np.int64)
        peak = int(_np.max(array))
        peak_index = int(_np.argmax(array))
        return peak, peak_index
    peak = max(values)
    return peak, values.index(peak)


def compute_budget_report(
    event_records: list[dict[str, Any]],
    tick_records: list[dict[str, Any]],
    rules: dict[str, Any],
    violations: list[dict[str, Any]],
) -> dict[str, Any]:
    factions = sorted(rules["factions"])
    report: dict[str, Any] = {}

    for faction in factions:
        faction_events = [
            record for record in event_records if record["actor"] == faction
        ]
        resources = [record["state_after"]["resources"] for record in faction_events]
        queue_depths = [
            len(record["state_after"]["queue"]) for record in faction_events
        ]
        peak_resources, resource_index = _numeric_peaks(resources)
        peak_queue, queue_index = _numeric_peaks(queue_depths)

        total_collected = sum(
            record["resource_delta"]
            for record in faction_events
            if record["action"] == "collect" and record["outcome"] == "applied"
        )
        total_spent = -sum(
            record["resource_delta"]
            for record in faction_events
            if record["action"] == "produce" and record["outcome"] == "applied"
        )
        total_produced = sum(
            record["amount"]
            for record in faction_events
            if record["action"] == "produce" and record["outcome"] == "applied"
        )
        total_dispatched = sum(
            record["amount"]
            for record in faction_events
            if record["action"] == "dispatch" and record["outcome"] == "applied"
        )
        faction_violations = [item for item in violations if item["actor"] == faction]
        reason_counts: dict[str, int] = {}
        for item in faction_violations:
            reason_counts[item["reason"]] = reason_counts.get(item["reason"], 0) + 1
        first_violation = faction_violations[0] if faction_violations else None
        first_resource_violation = next(
            (
                item
                for item in faction_violations
                if item["reason"] in ("resource_cap", "resource_negative")
            ),
            None,
        )
        attempted_values = [peak_resources]
        attempted_values.extend(
            int(item["attempted_resources"])
            for item in faction_violations
            if item.get("attempted_resources") is not None
        )
        peak_attempted = max(attempted_values)
        report[faction] = {
            "event_count": len(faction_events),
            "resource_cap": rules["factions"][faction]["resource_cap"],
            "queue_capacity": rules["factions"][faction]["queue_capacity"],
            "peak_resources": peak_resources,
            "peak_resources_tick": (
                faction_events[resource_index]["tick"] if faction_events else None
            ),
            "peak_resources_event_index": (
                faction_events[resource_index]["event_index"] if faction_events else None
            ),
            "peak_utilization_ppm": (
                peak_resources * 1_000_000 // rules["factions"][faction]["resource_cap"]
            ),
            "peak_attempted_resources": peak_attempted,
            "peak_queue_depth": peak_queue,
            "peak_queue_tick": (
                faction_events[queue_index]["tick"] if faction_events else None
            ),
            "total_collected": total_collected,
            "total_consumed": total_spent,
            "total_produced_units": total_produced,
            "total_dispatched_units": total_dispatched,
            "overflow_count": len(faction_violations),
            "resource_overflow_count": sum(
                count
                for reason, count in reason_counts.items()
                if reason in ("resource_cap", "resource_negative")
            ),
            "queue_overflow_count": reason_counts.get("queue_capacity", 0),
            "queue_underflow_count": reason_counts.get("queue_underflow", 0),
            "overflow_reasons": {key: reason_counts[key] for key in sorted(reason_counts)},
            "first_overflow_event_index": (
                first_violation["event_index"] if first_violation else None
            ),
            "first_overflow_event_key": (
                {
                    "tick": first_violation["tick"],
                    "actor": first_violation["actor"],
                    "local_seq": first_violation["local_seq"],
                }
                if first_violation
                else None
            ),
            "first_resource_overflow_event_index": (
                first_resource_violation["event_index"]
                if first_resource_violation
                else None
            ),
        }

    all_resources = [
        record["state_after"]["resources"] for record in event_records
    ]
    global_peak, _ = _numeric_peaks(all_resources)
    total_overflow = len(violations)
    return {
        "factions": report,
        "global_peak_resources": global_peak,
        "global_overflow_count": total_overflow,
        "global_first_overflow_event_index": (
            violations[0]["event_index"] if violations else None
        ),
        "has_overflow": bool(violations),
        "overflow_conclusion": (
            "E_RESOURCE_OVERFLOW" if violations else "within_limits"
        ),
        "tick_count": len(tick_records),
    }
