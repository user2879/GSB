from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import datetime, timezone
from itertools import zip_longest
from typing import Any

from .errors import BusinessError
from .locks import LockManager
from .models import (
    EditScript,
    ReplayReport,
    RestoreReport,
    RunResult,
    SchedulerParams,
    canonical_json,
)
from .scheduler import Engine
from .scripts import normalize_document
from .storage import SnapshotStore, Storage

_MISSING = object()


def _trace_projection(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_seq": int(event["event_seq"]),
        "virtual_ms": int(event["virtual_ms"]),
        "session_id": event.get("session_id"),
        "op_no": event.get("op_no"),
        "event_type": event["event_type"],
        "payload": event.get("payload", {}),
    }


def build_snapshot(
    snapshot_id: str,
    script: EditScript,
    result: RunResult,
    *,
    lock_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_lock_state = copy.deepcopy(lock_state if lock_state is not None else result.lock_state)
    wait_queues: list[dict[str, Any]] = []
    for row in final_lock_state.get("rows", []):
        for request in row.get("queue", []):
            request_with_row = dict(request)
            request_with_row["row_key"] = row["row_key"]
            wait_queues.append(request_with_row)
    return {
        "snapshot_id": snapshot_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script_id": result.script_id,
        "source_version": result.source_version,
        "normalized_version": result.normalized_version,
        "script": script.normalized_for_snapshot(),
        "script_digest": result.script_digest,
        "upgrade_changes": list(script.upgrade_changes),
        "params": result.params.to_dict(),
        "parameter_version": result.params.parameter_version,
        "params_digest": result.params.digest,
        "run_id": result.run_id,
        "lock_table": final_lock_state,
        "wait_queues": wait_queues,
        "lock_state": final_lock_state,
        "trace": [event.to_dict() for event in result.trace],
        "outcomes": [outcome.to_dict() for outcome in result.outcomes],
        "rows": copy.deepcopy(result.final_rows),
        "commits": [commit.to_dict() for commit in result.commits],
        "global_data_version": result.global_data_version,
    }


def save_snapshot(
    store: SnapshotStore,
    snapshot_id: str,
    script: EditScript,
    result: RunResult,
    *,
    lock_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = build_snapshot(snapshot_id, script, result, lock_state=lock_state)
    store.save(snapshot)
    return snapshot


def _changed_parameter_items(current: dict[str, Any], snapshot: dict[str, Any]) -> list[str]:
    changed: list[str] = []
    for key in sorted(set(current) | set(snapshot)):
        if current.get(key) != snapshot.get(key):
            changed.append(key)
    return changed


def _changed_sessions(current_outcomes: list[dict[str, Any]], snapshot_outcomes: list[dict[str, Any]]) -> list[str]:
    current_by_id = {item["session_id"]: item for item in current_outcomes}
    snapshot_by_id = {item["session_id"]: item for item in snapshot_outcomes}
    changed: list[str] = []
    for session_id in sorted(set(current_by_id) | set(snapshot_by_id)):
        if current_by_id.get(session_id) != snapshot_by_id.get(session_id):
            changed.append(session_id)
    return changed


def restore_snapshot(
    store: SnapshotStore,
    snapshot_id: str,
    *,
    target_storage: Storage,
    target_lock_manager: LockManager | None = None,
) -> RestoreReport:
    snapshot = store.load(snapshot_id)
    current = target_storage.get_context()
    cleared_lock_entries = target_lock_manager.entry_count() if target_lock_manager is not None else 0
    if target_lock_manager is not None:
        target_lock_manager.clear()

    script_match = current.get("script_digest") == snapshot["script_digest"]
    param_match = current.get("params_digest") == snapshot["params_digest"]
    parameter_version_match = (
        current.get("parameter_version") == snapshot.get("parameter_version")
        and snapshot.get("parameter_version") == snapshot.get("params", {}).get("parameter_version")
    )
    changed_parameters = _changed_parameter_items(
        current.get("params") or {},
        snapshot.get("params") or {},
    )
    changed_sessions = _changed_sessions(
        current.get("outcomes") or [],
        snapshot.get("outcomes") or [],
    )

    # Restoration is deliberately a whole-state replacement, independent of
    # whether the current context appears equal or not.
    target_storage.replace_state_from_snapshot(snapshot)
    trace_entries = len(snapshot.get("trace") or [])
    return RestoreReport(
        snapshot_id=snapshot_id,
        mode="replace",
        script_digest_match=script_match,
        param_digest_match=param_match,
        parameter_version_match=parameter_version_match,
        changed_sessions=changed_sessions,
        changed_parameter_items=changed_parameters,
        cleared_lock_entries=cleared_lock_entries,
        trace_entries=trace_entries,
        data_version=int(snapshot["global_data_version"]),
    )


def _first_diff(expected: Any, actual: Any, path: str) -> tuple[str, Any, Any] | None:
    if type(expected) is not type(actual) and not (expected is None or actual is None):
        return path or "$", expected, actual
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                return f"{path}.{key}", None, actual[key]
            if key not in actual:
                return f"{path}.{key}", expected[key], None
            difference = _first_diff(expected[key], actual[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        for index, (left, right) in enumerate(zip_longest(expected, actual, fillvalue=_MISSING)):
            if left is _MISSING:
                return f"{path}[{index}]", None, right
            if right is _MISSING:
                return f"{path}[{index}]", left, None
            difference = _first_diff(left, right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return path or "$", expected, actual
    return None


def _operation_projection(operation: dict[str, Any] | None) -> dict[str, Any] | None:
    if operation is None:
        return None
    return {
        key: value
        for key, value in operation.items()
        if key not in ("monotonic_start_ms", "monotonic_end_ms")
    }


def _compare_outcome(snapshot_outcome: dict[str, Any] | None, replay_outcome: dict[str, Any] | None, session_id: str) -> ReplayReport | None:
    if snapshot_outcome is not None:
        snapshot_outcome = json.loads(canonical_json(snapshot_outcome))
    if replay_outcome is not None:
        replay_outcome = json.loads(canonical_json(replay_outcome))
    if snapshot_outcome is None or replay_outcome is None:
        return ReplayReport(
            snapshot_id="",
            reproducible=False,
            session_id=session_id,
            field="outcome",
            expected=snapshot_outcome,
            actual=replay_outcome,
            message="session outcome is missing on one side",
        )
    scalar_fields = (
        "start_seq",
        "isolation_level",
        "concurrency_limit",
        "outcome",
        "business_code",
        "committed_data_version",
        "wait_used_ms",
        "started_virtual_ms",
        "finished_virtual_ms",
    )
    for field in scalar_fields:
        if snapshot_outcome.get(field) != replay_outcome.get(field):
            return ReplayReport(
                snapshot_id="",
                reproducible=False,
                session_id=session_id,
                field=f"outcome.{field}",
                expected=snapshot_outcome.get(field),
                actual=replay_outcome.get(field),
                message=f"session outcome field differs: {field}",
            )
    snapshot_ops = snapshot_outcome.get("operations") or []
    replay_ops = replay_outcome.get("operations") or []
    for index, (left, right) in enumerate(zip_longest(snapshot_ops, replay_ops, fillvalue=None)):
        op_no = (left or right or {}).get("op_no", index + 1)
        difference = _first_diff(
            _operation_projection(left),
            _operation_projection(right),
            f"operations[{index}]",
        )
        if difference is not None:
            field, expected, actual = difference
            return ReplayReport(
                snapshot_id="",
                reproducible=False,
                session_id=session_id,
                op_no=op_no,
                field=field,
                expected=expected,
                actual=actual,
                message=f"operation trace differs at {field}",
            )
    return None


def _compare_with_snapshot(snapshot: dict[str, Any], replay: RunResult) -> ReplayReport:
    snapshot_outcomes = snapshot.get("outcomes") or []
    replay_outcomes = replay.summary_dict()["sessions"]
    ordered_sessions = sorted(
        {
            outcome.get("session_id")
            for outcome in snapshot_outcomes + replay_outcomes
            if outcome.get("session_id") is not None
        }
    )
    snapshot_by_id = {outcome["session_id"]: outcome for outcome in snapshot_outcomes}
    replay_by_id = {outcome["session_id"]: outcome for outcome in replay_outcomes}
    for session_id in ordered_sessions:
        difference = _compare_outcome(snapshot_by_id.get(session_id), replay_by_id.get(session_id), session_id)
        if difference is not None:
            difference.snapshot_id = snapshot["snapshot_id"]
            return difference

    left_trace = [_trace_projection(event) for event in snapshot.get("trace") or []]
    right_trace = [event.deterministic_dict() for event in replay.trace]
    for index, (left, right) in enumerate(zip_longest(left_trace, right_trace, fillvalue=None)):
        if left is None or right is None:
            event = left or right or {}
            return ReplayReport(
                snapshot_id=snapshot["snapshot_id"],
                reproducible=False,
                session_id=event.get("session_id"),
                op_no=event.get("op_no"),
                field=f"trace[{index}].event",
                expected=left,
                actual=right,
                message="trace event is missing on one side",
            )
        difference = _first_diff(left, right, f"trace[{index}]")
        if difference is not None:
            field, expected, actual = difference
            return ReplayReport(
                snapshot_id=snapshot["snapshot_id"],
                reproducible=False,
                session_id=left.get("session_id") or right.get("session_id"),
                op_no=left.get("op_no") if left.get("op_no") is not None else right.get("op_no"),
                field=field,
                expected=expected,
                actual=actual,
                message=f"trace field differs: {field}",
            )

    if snapshot["global_data_version"] != replay.global_data_version:
        return ReplayReport(
            snapshot_id=snapshot["snapshot_id"],
            reproducible=False,
            field="global_data_version",
            expected=snapshot["global_data_version"],
            actual=replay.global_data_version,
            message="final data version differs",
        )
    difference = _first_diff(snapshot["rows"], replay.final_rows, "rows")
    if difference is not None:
        field, expected, actual = difference
        return ReplayReport(
            snapshot_id=snapshot["snapshot_id"],
            reproducible=False,
            field=field,
            expected=expected,
            actual=actual,
            message="final row state differs",
        )
    return ReplayReport(snapshot_id=snapshot["snapshot_id"], reproducible=True)


def replay_snapshot(
    store: SnapshotStore,
    snapshot_id: str,
    *,
    raise_on_error: bool = True,
) -> tuple[ReplayReport, RunResult]:
    snapshot = store.load(snapshot_id)
    try:
        script = normalize_document(snapshot["script"])
        params = SchedulerParams(
            global_concurrency_limit=int(snapshot["params"]["global_concurrency_limit"]),
            clock_quantum_ms=int(snapshot["params"]["clock_quantum_ms"]),
            parameter_version=int(snapshot["params"]["parameter_version"]),
        )
        # A replay uses the normalized semantics but retains the source marker
        # so run.start events can be compared without a false source-version diff.
        script = replace(script, source_version=int(snapshot["source_version"]))
    except BusinessError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise BusinessError(
            "E_SNAPSHOT",
            f"snapshot cannot be reconstructed as a script: {exc}",
            path=f"$.snapshots.{snapshot_id}.script",
            details={"snapshot_id": snapshot_id},
        ) from exc

    storage = Storage(":memory:")
    try:
        replay = Engine(storage, params).run(script)
        report = _compare_with_snapshot(snapshot, replay)
    finally:
        storage.close()
    if not report.reproducible and raise_on_error:
        raise BusinessError(
            "E_REPLAY",
            report.message or "snapshot is not reproducible",
            path=report.field or "$",
            session_id=report.session_id,
            op_no=report.op_no,
            details={
                "snapshot_id": snapshot_id,
                "field": report.field,
                "expected": report.expected,
                "actual": report.actual,
            },
        )
    return report, replay
