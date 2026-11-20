from __future__ import annotations

import json
from typing import Any

from .engine import QuotaEngine
from .errors import ErrorRecord, ReplayMismatchError
from .loopback import NullSignalBroker
from .models import TraceEvent
from .storage import SQLiteLedger


POOL_COMPARE_FIELDS = (
    "own_active",
    "borrowed_in",
    "lent_out",
    "active_connections",
    "capacity_in_use",
    "active_with_lent",
    "remaining_connections",
    "warning",
)


def _trace_event(data: dict[str, Any]) -> TraceEvent:
    return TraceEvent(
        seq=int(data["seq"]),
        pool=str(data["pool"]),
        connection=str(data["connection"]),
        event_type=str(data["type"]),
        timestamp_ms=int(data["timestamp_ms"]),
        bytes_count=int(data["bytes"]),
        line=0,
        raw=data,
    )


def _mismatch(
    *,
    seq: int,
    pool_id: str,
    field: str,
    ledger_value: Any,
    replay_value: Any,
    reason: str,
) -> ReplayMismatchError:
    record = ErrorRecord(
        code="E_REPLAY",
        message=f"replay mismatch at event {seq}, pool {pool_id}, field {field}: {reason}",
        path=f"ledger.events[seq={seq}].pools[{pool_id}].{field}",
        business_code="E_REPLAY",
        details={
            "event_seq": seq,
            "pool_id": pool_id,
            "field": field,
            "ledger_value": ledger_value,
            "replay_value": replay_value,
            "basis": reason,
        },
    )
    return ReplayMismatchError(record)


def _compare_one(seq: int, ledger_audit: dict[str, Any], replay_audit: dict[str, Any]) -> None:
    ledger_totals = ledger_audit.get("totals", {})
    ledger_assignment_count = int(ledger_totals.get("assignment_count", 0))
    ledger_active_sum = sum(
        int(pool.get("active_connections", 0))
        for pool in ledger_audit.get("pools", {}).values()
    )
    if ledger_active_sum != ledger_assignment_count:
        raise _mismatch(
            seq=seq,
            pool_id="*",
            field="active_connections_sum",
            ledger_value=ledger_active_sum,
            replay_value=ledger_assignment_count,
            reason="ledger active-count sum differs from ledger ownership assignment count",
        )

    replay_totals = replay_audit.get("totals", {})
    replay_active_sum = sum(
        int(pool.get("active_connections", 0))
        for pool in replay_audit.get("pools", {}).values()
    )
    if replay_active_sum != ledger_assignment_count:
        raise _mismatch(
            seq=seq,
            pool_id="*",
            field="active_connections_sum",
            ledger_value=ledger_assignment_count,
            replay_value=replay_active_sum,
            reason="replayed active-count sum differs from ledger ownership assignment count",
        )

    if ledger_audit.get("parameter_version") != replay_audit.get("parameter_version"):
        raise _mismatch(
            seq=seq,
            pool_id="*",
            field="parameter_version",
            ledger_value=ledger_audit.get("parameter_version"),
            replay_value=replay_audit.get("parameter_version"),
            reason="parameter versions differ",
        )

    ledger_pools = ledger_audit.get("pools", {})
    replay_pools = replay_audit.get("pools", {})
    for pool_id in sorted(set(ledger_pools) | set(replay_pools)):
        if pool_id not in replay_pools:
            raise _mismatch(
                seq=seq,
                pool_id=pool_id,
                field="pool_presence",
                ledger_value=True,
                replay_value=False,
                reason="pool exists in ledger but not in replay",
            )
        if pool_id not in ledger_pools:
            raise _mismatch(
                seq=seq,
                pool_id=pool_id,
                field="pool_presence",
                ledger_value=False,
                replay_value=True,
                reason="pool exists in replay but not in ledger",
            )
        for field in POOL_COMPARE_FIELDS:
            if ledger_pools[pool_id].get(field) != replay_pools[pool_id].get(field):
                raise _mismatch(
                    seq=seq,
                    pool_id=pool_id,
                    field=field,
                    ledger_value=ledger_pools[pool_id].get(field),
                    replay_value=replay_pools[pool_id].get(field),
                    reason="ledger value differs from replayed state machine value",
                )

    ledger_ownership = ledger_audit.get("ownership", {})
    replay_ownership = replay_audit.get("ownership", {})
    if ledger_ownership != replay_ownership:
        for connection_id in sorted(set(ledger_ownership) | set(replay_ownership)):
            if ledger_ownership.get(connection_id) != replay_ownership.get(connection_id):
                raise _mismatch(
                    seq=seq,
                    pool_id=str(connection_id),
                    field="ownership",
                    ledger_value=ledger_ownership.get(connection_id),
                    replay_value=replay_ownership.get(connection_id),
                    reason="connection ownership differs from ledger",
                )

    ledger_warnings = ledger_audit.get("warning_conclusions", {})
    replay_warnings = replay_audit.get("warning_conclusions", {})
    for pool_id in sorted(set(ledger_warnings) | set(replay_warnings)):
        if ledger_warnings.get(pool_id) != replay_warnings.get(pool_id):
            raise _mismatch(
                seq=seq,
                pool_id=pool_id,
                field="warning_conclusion",
                ledger_value=ledger_warnings.get(pool_id),
                replay_value=replay_warnings.get(pool_id),
                reason="alert conclusion differs from ledger",
            )


def verify_replay(store: SQLiteLedger) -> dict[str, Any]:
    initial_snapshot = store.initial_snapshot()
    parameter_versions = store.parameter_versions()
    transitions = store.transitions()
    transition_by_version = {item["version"]: item for item in transitions}
    engine = QuotaEngine(initial_snapshot, signal_broker=NullSignalBroker())
    audits = store.all_event_audits()
    for entry in audits:
        event = _trace_event(entry["event"])
        ledger_result = entry["result"]
        ledger_audit = ledger_result["audit"]
        target_version = int(ledger_audit["parameter_version"])
        if target_version != engine.parameter_version:
            snapshot = parameter_versions.get(target_version)
            transition = transition_by_version.get(target_version)
            if snapshot is None or transition is None:
                raise _mismatch(
                    seq=event.seq,
                    pool_id="*",
                    field="parameter_version",
                    ledger_value=target_version,
                    replay_value=engine.parameter_version,
                    reason="ledger references a parameter version without a replayable transition",
                )
            engine.apply_parameters(
                snapshot,
                after_event_seq=int(transition["applied_after_seq"]),
            )
        replay_result = engine.process_event(event)
        _compare_one(event.seq, ledger_audit, replay_result.audit)
    trailing_versions: list[int] = []
    for transition in transitions:
        version = int(transition["version"])
        if version <= engine.parameter_version:
            continue
        snapshot = parameter_versions.get(version)
        if snapshot is None:
            raise _mismatch(
                seq=engine.last_event_seq,
                pool_id="*",
                field="parameter_version",
                ledger_value=version,
                replay_value=engine.parameter_version,
                reason="ledger has a trailing parameter version without a replayable snapshot",
            )
        engine.apply_parameters(
            snapshot,
            after_event_seq=int(transition["applied_after_seq"]),
        )
        trailing_versions.append(version)

    latest = store.latest_batch()
    if latest is not None:
        summary = json.loads(latest["summary_json"])
        actual_rows = {row["pool_id"]: row for row in summary.get("pool_summary", [])}
        expected_rows = {row["pool_id"]: row for row in engine.pool_rows()}
        fields = (
            "own_active",
            "borrowed_in",
            "lent_out",
            "active_connections",
            "active_with_lent",
            "capacity_in_use",
            "remaining_connections",
            "warning",
        )
        seq = int(latest["end_seq"])
        for pool_id in sorted(set(actual_rows) | set(expected_rows)):
            if pool_id not in actual_rows or pool_id not in expected_rows:
                raise _mismatch(
                    seq=seq,
                    pool_id=pool_id,
                    field="pool_presence",
                    ledger_value=pool_id in actual_rows,
                    replay_value=pool_id in expected_rows,
                    reason="final ledger pool set differs from replay",
                )
            for field in fields:
                if actual_rows[pool_id].get(field) != expected_rows[pool_id].get(field):
                    raise _mismatch(
                        seq=seq,
                        pool_id=pool_id,
                        field=field,
                        ledger_value=actual_rows[pool_id].get(field),
                        replay_value=expected_rows[pool_id].get(field),
                        reason="final persisted waterline differs from replay",
                    )
    return {
        "code": "OK",
        "consistent": True,
        "events_checked": len(audits),
        "first_mismatch": None,
        "active_sum_equals_assignment_count": True,
        "parameter_versions_checked": sorted(parameter_versions),
        "trailing_transitions_applied": trailing_versions,
    }
