"""Whole-snapshot replay verification against an uninterrupted baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .declaration import FrameScript, SessionDeclaration
from .errors import ToolError
from .session import H2Session
from .snapshot import SnapshotDocument, load_snapshot_document, restore_snapshot
from .util import declaration_diff, digest_value, flatten_dict


COMPARED_FIELDS = (
    "direction",
    "frame_type",
    "flags",
    "target_stream_id",
    "payload_bytes",
    "flow_bytes",
    "wire_bytes",
    "conn_send_window_before",
    "conn_send_window_after",
    "stream_send_window_before",
    "stream_send_window_after",
    "conn_recv_window_before",
    "conn_recv_window_after",
    "stream_recv_window_before",
    "stream_recv_window_after",
    "state_before",
    "state_after",
    "code",
)


@dataclass
class ReplayResult:
    status: str
    code: str
    reason: str
    message: str
    snapshot_path: str
    snapshot_script_index: int
    first_difference: dict[str, Any] | None
    basis: dict[str, Any]
    replacement: dict[str, Any] | None
    tail_truncated: bool
    compared_record_count: int
    baseline_summary: dict[str, Any] | None
    restored_summary: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "code": self.code,
            "reason": self.reason,
            "message": self.message,
            "snapshot_path": self.snapshot_path,
            "snapshot_script_index": self.snapshot_script_index,
            "first_difference": self.first_difference,
            "basis": self.basis,
            "replacement": self.replacement,
            "tail_truncated": self.tail_truncated,
            "compared_record_count": self.compared_record_count,
            "baseline_summary": self.baseline_summary,
            "restored_summary": self.restored_summary,
        }


def _not_reproducible(
    document: SnapshotDocument,
    reason: str,
    message: str,
    basis: dict[str, Any],
    *,
    tail_truncated: bool = False,
) -> ReplayResult:
    return ReplayResult(
        status="NOT_REPRODUCIBLE",
        code="E_REPLAY",
        reason=reason,
        message=message,
        snapshot_path=str(document.path),
        snapshot_script_index=document.script_index,
        first_difference=None,
        basis=basis,
        replacement=None,
        tail_truncated=tail_truncated,
        compared_record_count=0,
        baseline_summary=None,
        restored_summary=None,
    )


def _record_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {field: record.get(field) for field in COMPARED_FIELDS}


def _run_baseline(
    document: SnapshotDocument,
    script: FrameScript,
    *,
    stop_after_events: int | None = None,
) -> H2Session:
    initial_strategy = str(
        document.state.get("initial_strategy", document.state.get("strategy", document.strategy))
    )
    baseline = H2Session(
        document.declaration,
        script,
        strategy=initial_strategy,
        starvation_rounds=int(document.state.get("starvation_rounds", 3)),
    )
    controls_by_index: dict[int, list[dict[str, Any]]] = {}
    for control in document.state.get("ledger", {}).get("controls", []):
        if control.get("kind") == "switch_strategy":
            controls_by_index.setdefault(int(control["script_index"]), []).append(control)
    baseline.start()
    target = len(script.events) if stop_after_events is None else min(
        len(script.events), max(0, stop_after_events)
    )
    while baseline.script_index < target:
        for control in sorted(
            controls_by_index.get(baseline.script_index, []), key=lambda item: item["control_seq"]
        ):
            baseline.set_strategy(str(control["new"]), script_index=baseline.script_index)
        baseline.process_next_event()
    if stop_after_events is None or baseline.script_index >= len(script.events):
        baseline.finalize()
    return baseline


LEDGER_COMPONENTS = (
    "records",
    "transitions",
    "priority_events",
    "empty_waits",
    "queue_waits",
    "schedule_rounds",
    "controls",
    "errors",
)


def _first_list_difference(
    expected_items: list[dict[str, Any]],
    actual_items: list[dict[str, Any]],
    *,
    label: str,
    basis: str = "snapshot_prefix",
) -> dict[str, Any] | None:
    limit = max(len(expected_items), len(actual_items))
    for index in range(limit):
        if index >= len(expected_items):
            actual = actual_items[index]
            return {
                "frame_index": actual.get("frame_seq", index + 1),
                "ledger_seq": actual.get("seq", actual.get("transition_seq", actual.get("wait_id"))),
                "stream_id": actual.get("stream_id", actual.get("target_stream_id")),
                "field": f"{label}.count",
                "expected": "<missing>",
                "actual": "extra_item",
                "basis": basis,
            }
        if index >= len(actual_items):
            expected = expected_items[index]
            return {
                "frame_index": expected.get("frame_seq", index + 1),
                "ledger_seq": expected.get("seq", expected.get("transition_seq", expected.get("wait_id"))),
                "stream_id": expected.get("stream_id", expected.get("target_stream_id")),
                "field": f"{label}.count",
                "expected": "item_present",
                "actual": "<missing>",
                "basis": basis,
            }
        expected = expected_items[index]
        actual = actual_items[index]
        for field in sorted(set(expected) | set(actual)):
            if expected.get(field) != actual.get(field):
                return {
                    "frame_index": actual.get("frame_seq") or expected.get("frame_seq") or index + 1,
                    "ledger_seq": actual.get("seq", actual.get("transition_seq", actual.get("wait_id"))),
                    "stream_id": actual.get("stream_id", actual.get("target_stream_id")),
                    "field": f"{label}.{field}",
                    "expected": expected.get(field),
                    "actual": actual.get(field),
                    "basis": basis,
                }
    return None


def _first_flat_difference(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    path_prefix: str,
    label: str,
) -> dict[str, Any] | None:
    left = flatten_dict(expected, path_prefix)
    right = flatten_dict(actual, path_prefix)
    for path in sorted(set(left) | set(right)):
        if left.get(path, "<missing>") != right.get(path, "<missing>"):
            return {
                "frame_index": None,
                "stream_id": None,
                "field": path,
                "expected": left.get(path, "<missing>"),
                "actual": right.get(path, "<missing>"),
                "basis": label,
            }
    return None


def _first_record_difference(
    expected_records: list[dict[str, Any]],
    actual_records: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any] | None:
    limit = max(len(expected_records), len(actual_records))
    for index in range(limit):
        if index >= len(expected_records):
            actual = actual_records[index]
            return {
                "frame_index": actual.get("frame_seq", index + 1),
                "ledger_seq": actual.get("seq"),
                "stream_id": actual.get("target_stream_id"),
                "field": "record_count",
                "expected": "<missing>",
                "actual": "extra_record",
                "basis": label,
            }
        if index >= len(actual_records):
            expected = expected_records[index]
            return {
                "frame_index": expected.get("frame_seq", index + 1),
                "ledger_seq": expected.get("seq"),
                "stream_id": expected.get("target_stream_id"),
                "field": "record_count",
                "expected": "record_present",
                "actual": "<missing>",
                "basis": label,
            }
        expected = expected_records[index]
        actual = actual_records[index]
        for field in COMPARED_FIELDS:
            if expected.get(field) != actual.get(field):
                return {
                    "frame_index": actual.get("frame_seq") or expected.get("frame_seq") or index + 1,
                    "ledger_seq": actual.get("seq"),
                    "expected_ledger_seq": expected.get("seq"),
                    "stream_id": actual.get("target_stream_id"),
                    "expected_stream_id": expected.get("target_stream_id"),
                    "field": field,
                    "expected": expected.get(field),
                    "actual": actual.get(field),
                    "basis": label,
                }
    return None


def replay_from_snapshot(
    snapshot_path: str,
    script: FrameScript,
    *,
    current_declaration: SessionDeclaration | None = None,
    current_strategy: str | None = None,
    current_session: H2Session | None = None,
) -> ReplayResult:
    document = load_snapshot_document(snapshot_path)
    comparison_declaration = (
        current_session.declaration if current_session is not None else current_declaration
    )
    comparison_strategy = (
        current_session.strategy if current_session is not None else current_strategy
    )
    basis: dict[str, Any] = {
        "snapshot_declaration_version": document.declaration.declaration_version,
        "snapshot_declaration_digest": digest_value(document.declaration.to_dict()),
        "snapshot_content_digest": document.content_digest,
        "snapshot_strategy": document.strategy,
        "snapshot_script_index": document.script_index,
        "snapshot_script_event_count": document.script_event_count,
        "current_script_event_count": len(script.events),
        "script_prefix_digest": document.script_prefix_digest,
        "current_script_prefix_digest": script.prefix_digest(min(document.script_index, len(script.events))),
    }
    if comparison_declaration is not None:
        changes = declaration_diff(
            comparison_declaration.to_dict(), document.declaration.to_dict()
        )
        basis["changed_declaration_fields"] = changes
        basis["current_declaration_version"] = comparison_declaration.declaration_version
        basis["current_declaration_digest"] = comparison_declaration.content_digest
        if comparison_declaration.declaration_version != document.declaration.declaration_version:
            return _not_reproducible(
                document,
                "DECLARATION_VERSION_MISMATCH",
                "snapshot declaration version differs from the current declaration",
                basis,
            )
        if comparison_declaration.content_digest != document.declaration.content_digest:
            return _not_reproducible(
                document,
                "DECLARATION_CONTENT_MISMATCH",
                "snapshot declaration content digest differs from the current declaration",
                basis,
            )
    if comparison_strategy is not None:
        basis["current_strategy"] = comparison_strategy
        if comparison_strategy != document.strategy:
            return _not_reproducible(
                document,
                "STRATEGY_CHANGED",
                "snapshot strategy differs from the current session strategy",
                basis,
            )
    if document.script_index > len(script.events):
        return _not_reproducible(
            document,
            "SCRIPT_PREFIX_TRUNCATED",
            "the supplied script ends before the snapshot cursor",
            basis,
            tail_truncated=True,
        )
    current_prefix = script.prefix_digest(document.script_index)
    if current_prefix != document.script_prefix_digest:
        return _not_reproducible(
            document,
            "SCRIPT_PREFIX_MISMATCH",
            "script events before the snapshot cursor are not identical",
            basis,
        )

    prefix_baseline = _run_baseline(
        document, script, stop_after_events=document.script_index
    )
    baseline = _run_baseline(document, script)

    try:
        restoration = restore_snapshot(
            snapshot_path,
            script,
            current_declaration=document.declaration,
            current_strategy=document.strategy,
        )
    except ToolError as exc:
        return _not_reproducible(
            document,
            "SNAPSHOT_RESTORE_FAILED",
            exc.message,
            {**basis, "restore_error": exc.to_dict()},
        )
    restored = restoration.session
    prefix_count = len(document.state.get("ledger", {}).get("records", []))
    prefix_difference: dict[str, Any] | None = None
    expected_prefix_state = {
        key: value
        for key, value in prefix_baseline.state_dict().items()
        if key != "ledger"
    }
    actual_prefix_state = {
        key: value for key, value in document.state.items() if key != "ledger"
    }
    prefix_difference = _first_flat_difference(
        expected_prefix_state,
        actual_prefix_state,
        path_prefix="$.state",
        label="snapshot_prefix_state",
    )
    if prefix_difference is not None:
        prefix_difference["frame_index"] = document.state.get("frame_seq", 0) + 1
    if prefix_difference is None:
        expected_ledger = prefix_baseline.ledger.to_dict()
        actual_ledger = document.state.get("ledger", {})
        for component in LEDGER_COMPONENTS:
            prefix_difference = _first_list_difference(
                expected_ledger.get(component, []),
                actual_ledger.get(component, []),
                label=component,
            )
            if prefix_difference is not None:
                break
    if prefix_difference is not None:
        return ReplayResult(
            status="MISMATCH",
            code="E_REPLAY",
            reason="SNAPSHOT_PREFIX_MISMATCH",
            message="persisted snapshot prefix does not reproduce from the declaration and script",
            snapshot_path=str(document.path),
            snapshot_script_index=document.script_index,
            first_difference=prefix_difference,
            basis=basis,
            replacement=restoration.to_dict(),
            tail_truncated=False,
            compared_record_count=prefix_count,
            baseline_summary=baseline.summary(),
            restored_summary=restored.summary(),
        )

    restored.run()
    expected_records = baseline.ledger.records[prefix_count:]
    actual_records = restored.ledger.records[prefix_count:]
    difference = _first_record_difference(
        expected_records, actual_records, label="post_snapshot_replay"
    )
    mismatch_reason = "FRAME_STATE_MISMATCH"
    mismatch_message = "restored continuation differs from the uninterrupted baseline"
    if difference is None:
        prefix_transitions = len(document.state.get("ledger", {}).get("transitions", []))
        difference = _first_list_difference(
            baseline.ledger.transitions[prefix_transitions:],
            restored.ledger.transitions[prefix_transitions:],
            label="transition",
            basis="post_snapshot_replay",
        )
        if difference is not None:
            mismatch_reason = "STATE_TRANSITION_MISMATCH"
            mismatch_message = "restored state-transition trace differs from the baseline"
    if difference is None:
        prefix_priority_events = len(
            document.state.get("ledger", {}).get("priority_events", [])
        )
        difference = _first_list_difference(
            baseline.ledger.priority_events[prefix_priority_events:],
            restored.ledger.priority_events[prefix_priority_events:],
            label="priority_event",
            basis="post_snapshot_replay",
        )
        if difference is not None:
            mismatch_reason = "PRIORITY_STATE_MISMATCH"
            mismatch_message = "restored priority evolution differs from the baseline"
    if difference is None:
        prefix_rounds = len(document.state.get("ledger", {}).get("schedule_rounds", []))
        difference = _first_list_difference(
            baseline.ledger.schedule_rounds[prefix_rounds:],
            restored.ledger.schedule_rounds[prefix_rounds:],
            label="schedule_round",
            basis="post_snapshot_replay",
        )
        if difference is not None:
            mismatch_reason = "SCHEDULE_ROUND_MISMATCH"
            mismatch_message = "restored schedule rounds differ from the baseline"
    if difference is None:
        difference = _first_flat_difference(
            baseline.diagnostics(),
            restored.diagnostics(),
            path_prefix="$.diagnostics",
            label="post_snapshot_replay",
        )
        if difference is not None:
            mismatch_reason = "DIAGNOSTIC_MISMATCH"
            mismatch_message = "restored fairness diagnostics differ from the baseline"
    tail_truncated = bool(
        len(script.events) < document.script_event_count
        or baseline.completion_report()["script_tail_truncated"]
        or restored.completion_report()["script_tail_truncated"]
    )
    if difference is not None:
        return ReplayResult(
            status="MISMATCH",
            code="E_REPLAY",
            reason=mismatch_reason,
            message=mismatch_message,
            snapshot_path=str(document.path),
            snapshot_script_index=document.script_index,
            first_difference=difference,
            basis=basis,
            replacement=restoration.to_dict(),
            tail_truncated=tail_truncated,
            compared_record_count=len(expected_records),
            baseline_summary=baseline.summary(),
            restored_summary=restored.summary(),
        )
    if tail_truncated:
        return ReplayResult(
            status="TRUNCATED",
            code="E_REPLAY",
            reason="SCRIPT_TAIL_TRUNCATED",
            message="the replay interval matched, but the frame script ends before session completion",
            snapshot_path=str(document.path),
            snapshot_script_index=document.script_index,
            first_difference=None,
            basis={**basis, "tail_attribution": "script_tail_truncated", "completion": baseline.completion_report()},
            replacement=restoration.to_dict(),
            tail_truncated=True,
            compared_record_count=len(expected_records),
            baseline_summary=baseline.summary(),
            restored_summary=restored.summary(),
        )
    return ReplayResult(
        status="MATCH",
        code="OK",
        reason="REPLAY_MATCH",
        message="restored continuation matches the uninterrupted baseline frame by frame",
        snapshot_path=str(document.path),
        snapshot_script_index=document.script_index,
        first_difference=None,
        basis=basis,
        replacement=restoration.to_dict(),
        tail_truncated=False,
        compared_record_count=len(expected_records),
        baseline_summary=baseline.summary(),
        restored_summary=restored.summary(),
    )
