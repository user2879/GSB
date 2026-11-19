from __future__ import annotations

from audit_forensics.canonical import canonical_json
from helpers import (
    BASE_TS,
    codes,
    event,
    import_workspace,
    rules_config,
    write_v1,
    write_v2,
)


def test_parent_link_errors_and_mixed_format_order_independence(tmp_path):
    v1_events = [
        event("e1", trace_id="T-1", seq=1, ts=BASE_TS + 1),
        event(
            "e2",
            trace_id="T-1",
            seq=2,
            ts=BASE_TS + 2,
            parent_event_id="e1",
        ),
        event(
            "e-orphan",
            trace_id="T-2",
            seq=1,
            ts=BASE_TS + 3,
            parent_event_id="missing",
        ),
        event(
            "e-self",
            trace_id="T-3",
            seq=1,
            ts=BASE_TS + 4,
            parent_event_id="e-self",
        ),
    ]
    v2_events = [
        event("x1", trace_id="T-9", seq=1, ts="2026-11-19T00:00:05Z"),
        event(
            "y1",
            trace_id="T-10",
            seq=1,
            ts="2026-11-19T00:00:06Z",
            parent_event_id="x1",
        ),
    ]
    v1 = write_v1(tmp_path / "v1.jsonl", v1_events)
    v2 = write_v2(tmp_path / "v2.json", v2_events)

    first = import_workspace(tmp_path / "ws1", [v1, v2], rules_config()).payload
    second = import_workspace(tmp_path / "ws2", [v2, v1], rules_config()).payload
    assert {"E_ORPHAN", "E_CYCLE", "E_CROSS_TRACE"} <= codes(first)
    assert canonical_json(first["traces"]) == canonical_json(second["traces"])

    trace_one = next(item for item in first["traces"] if item["trace_id"] == "T-1")
    e2 = next(node for node in trace_one["nodes"] if node["event_id"] == "e2")
    assert e2["parent_event_id"] == "e1"
    assert e2["depth"] == 1

    orphan_error = next(
        error for error in first["errors"] if error["code"] == "E_ORPHAN"
    )
    assert orphan_error["event_id"] == "e-orphan"
    cycle_error = next(
        error for error in first["errors"] if error["code"] == "E_CYCLE"
    )
    assert cycle_error["event_id"] == "e-self"
    cross_error = next(
        error for error in first["errors"] if error["code"] == "E_CROSS_TRACE"
    )
    assert cross_error["details"]["parent_trace_id"] == "T-9"


def test_duplicate_conflict_rollback_and_gap_are_independent(tmp_path):
    first_events = [
        event("e3", trace_id="T-1", seq=3, ts=BASE_TS + 3),
        event("e1", trace_id="T-1", seq=1, ts=BASE_TS + 1),
        event("e2", trace_id="T-1", seq=2, ts=BASE_TS + 2),
        event("e5", trace_id="T-1", seq=5, ts=BASE_TS + 5),
    ]
    duplicate_events = [event("e3", trace_id="T-1", seq=3, ts=BASE_TS + 3)]
    conflict_first = event("e-conflict", trace_id="T-2", actor="frank", seq=1)
    conflict_second = event(
        "e-conflict",
        trace_id="T-2",
        actor="mallory",
        ts="2026-11-19T00:00:00Z",
        seq=1,
    )
    v1_main = write_v1(tmp_path / "main.jsonl", first_events)
    v1_dup = write_v1(tmp_path / "dup.jsonl", duplicate_events)
    v1_conflict = write_v1(tmp_path / "conflict.jsonl", [conflict_first])
    v2_conflict = write_v2(tmp_path / "conflict.json", [conflict_second])

    result = import_workspace(
        tmp_path / "workspace",
        [v1_main, v1_dup, v1_conflict, v2_conflict],
        rules_config(),
    )
    report = result.payload
    found = codes(report)
    assert {"E_SEQ_ROLLBACK", "E_SEQ_GAP", "E_EVENT_CONFLICT"} <= found
    assert len(report["conflicts"]) == 1
    conflict = report["conflicts"][0]
    assert conflict["event_id"] == "e-conflict"
    assert set(conflict["source_formats"]) == {"v1", "v2"}
    assert "actor" in conflict["fields"]
    assert len([e for e in report["events"] if e["event"]["event_id"] == "e3"]) == 1
    duplicate = next(
        item for item in report["events"] if item["event"]["event_id"] == "e3"
    )
    assert duplicate["status"] == "duplicate"
    # A repeated delivery is not counted as a seq rollback.
    rollback_events = {
        error["event_id"]
        for error in report["errors"]
        if error["code"] == "E_SEQ_ROLLBACK"
    }
    assert "e3" not in rollback_events
    gap = next(error for error in report["errors"] if error["code"] == "E_SEQ_GAP")
    assert gap["details"]["missing_seq"] == [4]
