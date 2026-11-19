"""Parallel/serial business-state comparison and first-divergence evidence."""

from __future__ import annotations

from typing import Any

from .canonical import hash_value, hash_without
from .constants import PIPELINE_VERSION, SCHEMA_VERSION
from .errors import PipelineError
from .ledger import verify_ledger
from .rules import faction_hash

EVENT_BUSINESS_FIELDS = (
    "event_index",
    "tick",
    "actor",
    "local_seq",
    "action",
    "target",
    "amount",
    "rng_seed_delta",
    "rng",
    "outcome",
    "violation_reason",
    "resource_delta",
    "queue_delta",
    "state_after",
    "resource_queue_hash",
)


def _event_view(record: dict[str, Any]) -> dict[str, Any]:
    return {field: record.get(field) for field in EVENT_BUSINESS_FIELDS}


def _tick_map(ledger: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {record["tick"]: record for record in ledger["tick_records"]}


def _faction_state(tick_record: dict[str, Any], faction: str) -> dict[str, Any] | None:
    return tick_record.get("snapshot", {}).get("factions", {}).get(faction)


def _boundary_for_tick(ledger: dict[str, Any], tick: int) -> dict[str, Any] | None:
    for boundary in ledger.get("chunk_boundaries", []):
        if boundary["start_tick"] <= tick <= boundary["end_tick"]:
            return boundary
    return None


def _participants(ledger: dict[str, Any], tick: int) -> list[dict[str, Any]]:
    return [
        _event_view(record)
        for record in ledger.get("event_records", [])
        if record["tick"] == tick
    ]


def _evidence(
    code: str,
    *,
    tick: int | None,
    faction: str | None,
    parallel: dict[str, Any],
    serial: dict[str, Any],
    detail: dict[str, Any],
) -> dict[str, Any]:
    boundary = _boundary_for_tick(parallel, tick) if tick is not None else None
    return {
        "code": code,
        "tick": tick,
        "faction": faction,
        "detail": detail,
        "participating_events": _participants(parallel, tick) if tick is not None else [],
        "serial_participating_events": _participants(serial, tick) if tick is not None else [],
        "boundary_snapshot": boundary,
        "degradation_records": parallel.get("scheduler", {}).get("degradations", []),
        "parallel_chunk_scheduler": parallel.get("scheduler"),
    }


def compare_ledgers(parallel: dict[str, Any], serial: dict[str, Any]) -> dict[str, Any]:
    verify_ledger(parallel)
    verify_ledger(serial)
    compared_faction_tick_pairs = 0
    compared_events = min(
        len(parallel.get("event_records", [])), len(serial.get("event_records", []))
    )
    first: dict[str, Any] | None = None

    for field in ("rules_hash", "events_hash"):
        if parallel.get(field) != serial.get(field):
            first = _evidence(
                "E_LEDGER_HASH",
                tick=None,
                faction=None,
                parallel=parallel,
                serial=serial,
                detail={
                    "kind": "metadata_mismatch",
                    "field": field,
                    "parallel": parallel.get(field),
                    "serial": serial.get(field),
                },
            )
            break

    tick_left = _tick_map(parallel)
    tick_right = _tick_map(serial)
    tick_numbers = sorted(set(tick_left) | set(tick_right))
    if first is None:
        for tick in tick_numbers:
            left = tick_left.get(tick)
            right = tick_right.get(tick)
            if left is None or right is None:
                first = _evidence(
                    "E_LEDGER_HASH",
                    tick=tick,
                    faction=None,
                    parallel=parallel,
                    serial=serial,
                    detail={
                        "kind": "missing_tick",
                        "parallel_present": left is not None,
                        "serial_present": right is not None,
                    },
                )
                break
            factions = sorted(
                set(left.get("snapshot", {}).get("factions", {}))
                | set(right.get("snapshot", {}).get("factions", {}))
            )
            for faction in factions:
                compared_faction_tick_pairs += 1
                left_state = _faction_state(left, faction)
                right_state = _faction_state(right, faction)
                left_hash = left.get("faction_hashes", {}).get(faction)
                right_hash = right.get("faction_hashes", {}).get(faction)
                state_equal = left_state == right_state
                hash_equal = left_hash == right_hash
                recalculated_left = faction_hash(left_state) if left_state else None
                recalculated_right = faction_hash(right_state) if right_state else None
                if not (state_equal and hash_equal and left_hash == recalculated_left and right_hash == recalculated_right):
                    first = _evidence(
                        "E_LEDGER_HASH",
                        tick=tick,
                        faction=faction,
                        parallel=parallel,
                        serial=serial,
                        detail={
                            "kind": "faction_state_mismatch",
                            "parallel_resource_queue_hash": left_hash,
                            "serial_resource_queue_hash": right_hash,
                            "recalculated_parallel_hash": recalculated_left,
                            "recalculated_serial_hash": recalculated_right,
                            "parallel_state": left_state,
                            "serial_state": right_state,
                        },
                    )
                    break
            if first is not None:
                break
            if left.get("state_hash") != right.get("state_hash"):
                first = _evidence(
                    "E_LEDGER_HASH",
                    tick=tick,
                    faction=None,
                    parallel=parallel,
                    serial=serial,
                    detail={
                        "kind": "tick_state_hash_mismatch",
                        "parallel_state_hash": left.get("state_hash"),
                        "serial_state_hash": right.get("state_hash"),
                    },
                )
                break

    if first is None and parallel.get("initial_snapshot_hash") != serial.get("initial_snapshot_hash"):
        first = _evidence(
            "E_LEDGER_HASH",
            tick=None,
            faction=None,
            parallel=parallel,
            serial=serial,
            detail={"kind": "initial_snapshot_mismatch"},
        )
    if first is None and parallel.get("exit_snapshot_hash") != serial.get("exit_snapshot_hash"):
        first = _evidence(
            "E_LEDGER_HASH",
            tick=parallel.get("game", {}).get("tick_end"),
            faction=None,
            parallel=parallel,
            serial=serial,
            detail={"kind": "exit_snapshot_mismatch"},
        )

    events_left = parallel.get("event_records", [])
    events_right = serial.get("event_records", [])
    if first is None:
        for index in range(max(len(events_left), len(events_right))):
            left = _event_view(events_left[index]) if index < len(events_left) else None
            right = _event_view(events_right[index]) if index < len(events_right) else None
            if left != right:
                rng_drift = (
                    left is not None
                    and right is not None
                    and left.get("rng") != right.get("rng")
                )
                tick = (
                    left["tick"] if left is not None else right["tick"]
                    if right is not None
                    else None
                )
                faction = (
                    left["actor"] if left is not None else right["actor"]
                    if right is not None
                    else None
                )
                first = _evidence(
                    "E_RNG_DRIFT" if rng_drift else "E_LEDGER_HASH",
                    tick=tick,
                    faction=faction,
                    parallel=parallel,
                    serial=serial,
                    detail={
                        "kind": "event_record_mismatch",
                        "position": index,
                        "parallel_event": left,
                        "serial_event": right,
                    },
                )
                break

    if first is None and parallel.get("violations") != serial.get("violations"):
        failures = parallel.get("violations") or serial.get("violations") or []
        first_item = failures[0] if failures else {}
        first = _evidence(
            "E_RESOURCE_OVERFLOW",
            tick=first_item.get("tick"),
            faction=first_item.get("actor"),
            parallel=parallel,
            serial=serial,
            detail={
                "kind": "overflow_conclusion_mismatch",
                "parallel_overflow_count": len(parallel.get("violations", [])),
                "serial_overflow_count": len(serial.get("violations", [])),
            },
        )
    if first is None and parallel.get("budget_report") != serial.get("budget_report"):
        first = _evidence(
            "E_LEDGER_HASH",
            tick=None,
            faction=None,
            parallel=parallel,
            serial=serial,
            detail={"kind": "budget_report_mismatch"},
        )
    if first is None and parallel.get("business_hash") != serial.get("business_hash"):
        first = _evidence(
            "E_LEDGER_HASH",
            tick=None,
            faction=None,
            parallel=parallel,
            serial=serial,
            detail={"kind": "business_hash_mismatch"},
        )

    full_consistency = first is None
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "comparison": "parallel_vs_serial",
        "full_consistency": full_consistency,
        "result_code": "OK" if full_consistency else first["code"],
        "explicit_conclusion": (
            "FULL_MATCH: every (faction, tick) resource/queue state, tick hash, "
            "event derivation, overflow conclusion, and budget statistic matched."
            if full_consistency
            else "DIVERGENCE: the first differing business state is recorded below."
        ),
        "compared": {
            "tick_count": len(tick_numbers),
            "faction_tick_pairs": compared_faction_tick_pairs,
            "parallel_ticks": len(tick_left),
            "serial_ticks": len(tick_right),
            "parallel_events": len(events_left),
            "serial_events": len(events_right),
            "events_compared": compared_events,
            "rng_records_compared": min(
                len(parallel.get("rng_records", [])), len(serial.get("rng_records", []))
            ),
        },
        "first_divergence": first,
        "parallel_ledger_hash": parallel.get("ledger_hash"),
        "serial_ledger_hash": serial.get("ledger_hash"),
    }
    report["verification_hash"] = hash_without(report, "verification_hash")
    return report


def compare_to_truth(ledger: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    verify_ledger(ledger)
    truth_hash = truth.get("truth_hash")
    if truth_hash != hash_without(truth, "truth_hash"):
        raise PipelineError("E_LEDGER_HASH", "truth ledger self hash mismatch")
    left_ticks = _tick_map(ledger)
    right_ticks = _tick_map(truth)
    matched_pairs = 0
    first_mismatch: dict[str, Any] | None = None
    for tick in sorted(set(left_ticks) | set(right_ticks)):
        left = left_ticks.get(tick)
        right = right_ticks.get(tick)
        if left is None or right is None:
            first_mismatch = {"tick": tick, "reason": "missing_tick"}
            break
        if left.get("state_hash") != right.get("state_hash"):
            first_mismatch = {"tick": tick, "reason": "state_hash"}
            break
        for faction in sorted(left["faction_hashes"]):
            matched_pairs += 1
            if left["faction_hashes"].get(faction) != right["faction_hashes"].get(faction):
                first_mismatch = {"tick": tick, "faction": faction, "reason": "faction_hash"}
                break
        if first_mismatch:
            break
    business_equal = (
        ledger.get("events_hash") == truth.get("events_hash")
        and ledger.get("rules_hash") == truth.get("rules_hash")
        and ledger.get("exit_snapshot_hash") == truth.get("exit_snapshot_hash")
        and ledger.get("event_records") == truth.get("event_records")
        and ledger.get("rng_records") == truth.get("rng_records")
        and ledger.get("violations") == truth.get("violations")
        and ledger.get("budget_report") == truth.get("budget_report")
        and first_mismatch is None
    )
    report = {
        "full_consistency": business_equal,
        "result_code": "OK" if business_equal else "E_LEDGER_HASH",
        "explicit_conclusion": (
            "TRUTH_MATCH: generated event outcome ledger matches the independent truth ledger."
            if business_equal
            else "TRUTH_DIVERGENCE: generated truth ledger did not match the synthesized ledger."
        ),
        "matched_faction_tick_pairs": matched_pairs,
        "first_mismatch": first_mismatch,
        "truth_hash": truth_hash,
        "ledger_hash": ledger.get("ledger_hash"),
    }
    report["truth_comparison_hash"] = hash_value(report)
    return report
