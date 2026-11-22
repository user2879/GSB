"""Ledger replay, first-difference localization, and independent validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import canonical_json, digest_value
from .catalog import load_catalog
from .engine import Execution, build_execution
from .errors import EReplay, ERoute
from .ledger import Ledger
from .planner import RESCAN, RESIDENT


def _loads(value: str | None) -> Any:
    if value is None:
        return None
    return json.loads(value)


def first_difference(expected: Any, actual: Any, path: str = "") -> tuple[str, Any, Any] | None:
    if type(expected) is not type(actual):
        return path or "<root>", expected, actual
    if isinstance(expected, dict):
        keys = sorted(set(expected) | set(actual))
        for key in keys:
            if key not in expected:
                return f"{path}.{key}" if path else str(key), None, actual[key]
            if key not in actual:
                return f"{path}.{key}" if path else str(key), expected[key], None
            difference = first_difference(
                expected[key], actual[key], f"{path}.{key}" if path else str(key)
            )
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        common = min(len(expected), len(actual))
        for index in range(common):
            difference = first_difference(
                expected[index], actual[index], f"{path}[{index}]"
            )
            if difference is not None:
                return difference
        if len(expected) != len(actual):
            return f"{path}.length" if path else "length", len(expected), len(actual)
        return None
    if expected != actual:
        return path or "<root>", expected, actual
    return None


def _raise_replay(
    message: str,
    *,
    field: str,
    expected: Any,
    actual: Any,
    page_id: int | None = None,
    event_seq: int | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    payload = {
        "expected": expected,
        "actual": actual,
        "first_mismatch_field": field,
    }
    if details:
        payload.update(details)
    raise EReplay(
        message,
        page_id=page_id,
        event_seq=event_seq,
        field=field,
        details=payload,
    )


def load_snapshot_execution(
    ledger: Ledger,
    snapshot: dict[str, Any],
    *,
    catalog_path_override: str | Path | None = None,
) -> Execution:
    catalog_path = Path(catalog_path_override or snapshot["catalog_path"]).resolve()
    catalog = load_catalog(catalog_path, group_key_override=snapshot["group_key"])
    if catalog.digest != snapshot["catalog_digest"]:
        raise EReplay(
            "current disk catalog digest differs from the ledger snapshot",
            field="catalog_digest",
            details={"ledger": snapshot["catalog_digest"], "disk": catalog.digest},
        )
    return build_execution(
        catalog,
        int(snapshot["budget"]),
        force_route=snapshot["route"],
        batch_size=int(snapshot["batch_size"]),
    )


def compare_execution_with_ledger(
    ledger: Ledger,
    snapshot: dict[str, Any],
    execution: Execution,
) -> dict[str, Any]:
    snapshot_id = int(snapshot["snapshot_id"])
    cursor = int(snapshot["cursor"])
    committed_batches = ledger.read_batches(snapshot_id)
    expected_batches = [batch for batch in execution.batches if batch.cursor_end <= cursor]
    if len(committed_batches) != len(expected_batches):
        _raise_replay(
            "committed batch count differs from the replayed cursor",
            field="batches.length",
            expected=len(expected_batches),
            actual=len(committed_batches),
        )

    checked_outcomes = 0
    checked_evictions = 0
    checked_rescans = 0
    checked_group_states = 0
    for expected_batch, actual_batch in zip(expected_batches, committed_batches, strict=True):
        if int(actual_batch["batch_index"]) != expected_batch.batch_index:
            _raise_replay(
                "batch index mismatch",
                field="batch_index",
                expected=expected_batch.batch_index,
                actual=int(actual_batch["batch_index"]),
            )
        if int(actual_batch["cursor_start"]) != expected_batch.cursor_start or int(
            actual_batch["cursor_end"]
        ) != expected_batch.cursor_end:
            _raise_replay(
                "batch cursor mismatch",
                field="cursor",
                expected=[expected_batch.cursor_start, expected_batch.cursor_end],
                actual=[int(actual_batch["cursor_start"]), int(actual_batch["cursor_end"])],
            )
        if actual_batch["group_state_digest"] != expected_batch.group_state_digest:
            _raise_replay(
                "batch group-state digest mismatch",
                field="group_state_digest",
                expected=expected_batch.group_state_digest,
                actual=actual_batch["group_state_digest"],
                event_seq=expected_batch.cursor_end,
            )
        if actual_batch["batch_digest"] != expected_batch.batch_digest:
            _raise_replay(
                "batch digest mismatch",
                field="batch_digest",
                expected=expected_batch.batch_digest,
                actual=actual_batch["batch_digest"],
                event_seq=expected_batch.cursor_end,
            )

        actual_outcomes = [
            _loads(row["payload_json"])
            for row in ledger.read_page_outcomes(snapshot_id)
            if int(row["batch_index"]) == expected_batch.batch_index
        ]
        for expected, actual in zip(expected_batch.page_outcomes, actual_outcomes, strict=False):
            difference = first_difference(expected, actual)
            if difference is not None:
                field, expected_value, actual_value = difference
                _raise_replay(
                    "page outcome replay mismatch",
                    page_id=int(expected.get("page_id")) if isinstance(expected, dict) else None,
                    event_seq=int(expected.get("access_seq")) if isinstance(expected, dict) else None,
                    field=field,
                    expected=expected_value,
                    actual=actual_value,
                    details={"batch_index": expected_batch.batch_index},
                )
        if len(expected_batch.page_outcomes) != len(actual_outcomes):
            _raise_replay(
                "page outcome count mismatch",
                field="page_outcomes.length",
                expected=len(expected_batch.page_outcomes),
                actual=len(actual_outcomes),
                details={"batch_index": expected_batch.batch_index},
            )
        checked_outcomes += len(actual_outcomes)

        actual_evictions = [
            _loads(row["payload_json"])
            for row in ledger.read_evictions(snapshot_id)
            if int(row["batch_index"]) == expected_batch.batch_index
        ]
        for expected, actual in zip(expected_batch.eviction_events, actual_evictions, strict=False):
            difference = first_difference(expected, actual)
            if difference is not None:
                field, expected_value, actual_value = difference
                _raise_replay(
                    "eviction event replay mismatch",
                    page_id=int(expected.get("victim_page_id")),
                    event_seq=int(expected.get("event_seq")),
                    field=field,
                    expected=expected_value,
                    actual=actual_value,
                    details={"batch_index": expected_batch.batch_index},
                )
        if len(expected_batch.eviction_events) != len(actual_evictions):
            _raise_replay(
                "eviction event count mismatch",
                field="eviction_events.length",
                expected=len(expected_batch.eviction_events),
                actual=len(actual_evictions),
                details={"batch_index": expected_batch.batch_index},
            )
        checked_evictions += len(actual_evictions)

        actual_rescans = [
            _loads(row["payload_json"])
            for row in ledger.read_rescans(snapshot_id)
            if int(row["batch_index"]) == expected_batch.batch_index
        ]
        for expected, actual in zip(expected_batch.rescan_events, actual_rescans, strict=False):
            difference = first_difference(expected, actual)
            if difference is not None:
                field, expected_value, actual_value = difference
                _raise_replay(
                    "rescan event replay mismatch",
                    page_id=int(expected.get("page_id")),
                    event_seq=int(expected.get("event_seq")),
                    field=field,
                    expected=expected_value,
                    actual=actual_value,
                    details={"batch_index": expected_batch.batch_index},
                )
        if len(expected_batch.rescan_events) != len(actual_rescans):
            _raise_replay(
                "rescan event count mismatch",
                field="rescan_events.length",
                expected=len(expected_batch.rescan_events),
                actual=len(actual_rescans),
                details={"batch_index": expected_batch.batch_index},
            )
        checked_rescans += len(actual_rescans)

        actual_group_state = [
            _loads(row["payload_json"])
            for row in ledger.read_group_states(snapshot_id, expected_batch.batch_index)
        ]
        difference = first_difference(expected_batch.group_state, actual_group_state)
        if difference is not None:
            field, expected_value, actual_value = difference
            page_id = None
            if expected_batch.page_outcomes:
                page_id = int(expected_batch.page_outcomes[-1]["page_id"])
            _raise_replay(
                "aggregate group-state replay mismatch",
                page_id=page_id,
                event_seq=expected_batch.cursor_end,
                field=f"group_state.{field}",
                expected=expected_value,
                actual=actual_value,
                details={"batch_index": expected_batch.batch_index},
            )
        checked_group_states += len(actual_group_state)

    if int(snapshot["cursor"]) != cursor:
        _raise_replay(
            "snapshot cursor changed unexpectedly",
            field="cursor",
            expected=cursor,
            actual=int(snapshot["cursor"]),
        )
    snapshot_updated = ledger.get_active_snapshot()
    assert snapshot_updated is not None
    if snapshot_updated["status"] == "COMPLETE":
        stored_result = _loads(snapshot_updated["final_result_json"])
        difference = first_difference(execution.result, stored_result)
        if difference is not None:
            field, expected_value, actual_value = difference
            page_id = execution.page_outcomes[-1]["page_id"] if execution.page_outcomes else None
            _raise_replay(
                "final aggregate result replay mismatch",
                page_id=page_id,
                field=f"result.{field}",
                expected=expected_value,
                actual=actual_value,
            )
        if snapshot_updated["result_digest"] != execution.result_digest:
            _raise_replay(
                "final result digest mismatch",
                field="result_digest",
                expected=execution.result_digest,
                actual=snapshot_updated["result_digest"],
            )
    return {
        "checked_batches": len(committed_batches),
        "checked_page_outcomes": checked_outcomes,
        "checked_evictions": checked_evictions,
        "checked_rescans": checked_rescans,
        "checked_group_states": checked_group_states,
        "resume_cursor": cursor,
        "complete": snapshot_updated["status"] == "COMPLETE",
    }


def replay_database(
    db_path: str | Path,
    *,
    catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    with Ledger(db_path) as ledger:
        snapshot = ledger.get_active_snapshot()
        if snapshot is None:
            raise EReplay("ledger has no active snapshot", field="snapshot")
        execution = load_snapshot_execution(
            ledger,
            snapshot,
            catalog_path_override=catalog_path,
        )
        comparison = compare_execution_with_ledger(ledger, snapshot, execution)
        return {
            "ok": True,
            "command": "replay",
            "snapshot": {
                "snapshot_id": snapshot["snapshot_id"],
                "plan_version": snapshot["plan_version"],
                "route": snapshot["route"],
                "status": snapshot["status"],
            },
            **comparison,
            "ledger": ledger.ledger_summary(),
        }


def _route_self_check(ledger: Ledger, snapshot: dict[str, Any], execution: Execution) -> dict[str, Any]:
    snapshot_id = int(snapshot["snapshot_id"])
    stored_evictions = [_loads(row["payload_json"]) for row in ledger.read_evictions(snapshot_id)]
    expected_evictions = [
        event
        for event in execution.eviction_events
        if int(event["access_seq"]) <= int(snapshot["cursor"])
    ]
    difference = first_difference(expected_evictions, stored_evictions)
    if difference is not None:
        field, expected, actual = difference
        event_seq = None
        page_id = None
        if isinstance(expected, dict):
            event_seq = expected.get("event_seq")
            page_id = expected.get("victim_page_id")
            if field.endswith("victim_page_id") and isinstance(actual, int):
                page_id = actual
        elif stored_evictions:
            event_seq = stored_evictions[0].get("event_seq")
            page_id = stored_evictions[0].get("victim_page_id")
        raise ERoute(
            "eviction ledger is inconsistent with the canonical slot plan",
            page_id=page_id,
            event_seq=event_seq,
            field=f"eviction_events.{field}",
            details={"expected": expected, "actual": actual},
        )
    stored_rescans = [_loads(row["payload_json"]) for row in ledger.read_rescans(snapshot_id)]
    expected_rescans = [
        event
        for event in execution.rescan_events
        if int(event["access_seq"]) <= int(snapshot["cursor"])
    ]
    difference = first_difference(expected_rescans, stored_rescans)
    if difference is not None:
        field, expected, actual = difference
        event_seq = None
        page_id = None
        if isinstance(expected, dict):
            event_seq = expected.get("event_seq")
            page_id = expected.get("page_id")
        elif stored_rescans:
            event_seq = stored_rescans[0].get("event_seq")
            page_id = stored_rescans[0].get("page_id")
        raise ERoute(
            "rescan ledger is inconsistent with the canonical slot plan",
            page_id=page_id,
            event_seq=event_seq,
            field=f"rescan_events.{field}",
            details={"expected": expected, "actual": actual},
        )
    outcomes = ledger.read_page_outcomes(snapshot_id)
    cache_hits = sum(int(row["cache_hit"]) for row in outcomes)
    physical_scans = sum(int(row["physical_scan"]) for row in outcomes)
    if snapshot["route"] == RESIDENT:
        if stored_rescans:
            first = stored_rescans[0]
            raise ERoute(
                "resident route ledger contains rescan events",
                page_id=first.get("page_id"),
                event_seq=first.get("event_seq"),
                field="rescan_events",
            )
        if cache_hits != execution.cache_hits:
            raise ERoute(
                "resident cache hit count differs from the canonical slot plan",
                event_seq=len(outcomes),
                field="cache_hit_count",
                details={"expected": execution.cache_hits, "actual": cache_hits},
            )
    elif snapshot["route"] == RESCAN:
        if cache_hits != 0:
            raise ERoute(
                "rescan route counted cache hits",
                event_seq=len(outcomes),
                field="cache_hit_count",
                details={"expected": 0, "actual": cache_hits},
            )
        if len(stored_rescans) != len(outcomes):
            raise ERoute(
                "rescan route must emit one rescan event per scan access",
                event_seq=len(stored_rescans) + 1,
                field="rescan_events.length",
                details={"expected": len(outcomes), "actual": len(stored_rescans)},
            )
    return {
        "eviction_events": len(stored_evictions),
        "rescan_events": len(stored_rescans),
        "cache_hits": cache_hits,
        "physical_scans": physical_scans,
        "route": snapshot["route"],
    }


def validate_database(
    db_path: str | Path,
    *,
    catalog_path: str | Path | None = None,
) -> dict[str, Any]:
    with Ledger(db_path) as ledger:
        checks: list[dict[str, Any]] = []
        integrity = ledger.sqlite_integrity_check()
        checks.append({"name": "sqlite_integrity_check", "ok": integrity == "ok", "value": integrity})
        if integrity != "ok":
            raise EReplay(
                "SQLite integrity check failed",
                field="sqlite.integrity_check",
                details={"result": integrity},
            )
        digest = ledger.verify_content_digest()
        checks.append({"name": "ledger_content_digest", "ok": True, "value": digest})
        snapshot = ledger.get_active_snapshot()
        if snapshot is None:
            raise EReplay("ledger has no active snapshot", field="snapshot")
        execution = load_snapshot_execution(
            ledger,
            snapshot,
            catalog_path_override=catalog_path,
        )
        checks.append(
            {
                "name": "catalog_digest",
                "ok": True,
                "value": snapshot["catalog_digest"],
            }
        )
        route_check = _route_self_check(ledger, snapshot, execution)
        checks.append({"name": "eviction_rescan_route_consistency", "ok": True, "value": route_check})
        comparison = compare_execution_with_ledger(ledger, snapshot, execution)
        checks.append(
            {
                "name": "aggregate_replay_consistency",
                "ok": True,
                "value": comparison,
            }
        )
        expected_final = execution.final_group_state
        stored_batch_count = ledger.committed_batch_count(int(snapshot["snapshot_id"]))
        if stored_batch_count:
            stored_final = [
                _loads(row["payload_json"])
                for row in ledger.read_group_states(
                    int(snapshot["snapshot_id"]), stored_batch_count
                )
            ]
            if int(snapshot["cursor"]) == execution.progress_cursor:
                difference = first_difference(expected_final, stored_final)
                if difference is not None:
                    field, expected, actual = difference
                    raise EReplay(
                        "final group state is inconsistent with the aggregate view",
                        field=f"final_group_state.{field}",
                        expected=expected,
                        actual=actual,
                    )
        checks.append({"name": "final_group_state", "ok": True, "value": len(expected_final)})
        return {
            "ok": True,
            "command": "validate",
            "checks": checks,
            "snapshot": {
                "snapshot_id": snapshot["snapshot_id"],
                "plan_version": snapshot["plan_version"],
                "status": snapshot["status"],
                "route": snapshot["route"],
                "cursor": snapshot["cursor"],
                "result_digest": snapshot["result_digest"],
            },
            "ledger": ledger.ledger_summary(),
        }
