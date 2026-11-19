"""Stable, read-only projections used by the Tkinter dashboard and demos."""

from __future__ import annotations

from typing import Any

from .session import H2Session
from .snapshot import SnapshotStore
from .util import digest_value


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_viewmodel(
    session: H2Session,
    snapshot_store: SnapshotStore | None = None,
) -> dict[str, Any]:
    records = sorted(
        session.ledger.records,
        key=lambda item: (int(item.get("frame_seq") or 0), int(item.get("seq") or 0)),
    )
    frames: list[dict[str, Any]] = []
    for record in records:
        reason = record.get("reason")
        if isinstance(reason, dict):
            reason_text = str(reason.get("message", ""))
            reason_code = str(reason.get("code", ""))
        elif isinstance(reason, str):
            reason_text = reason
            reason_code = ""
        else:
            reason_text = ""
            reason_code = ""
        frames.append(
            {
                "seq": record.get("seq"),
                "frame_seq": record.get("frame_seq"),
                "clock_ms": record.get("clock_ms"),
                "script_index": record.get("script_index"),
                "direction": record.get("direction"),
                "frame_type": record.get("frame_type"),
                "stream_id": record.get("target_stream_id"),
                "payload_bytes": record.get("payload_bytes"),
                "flow_bytes": record.get("flow_bytes"),
                "wire_bytes": record.get("wire_bytes"),
                "conn_send_window_before": record.get("conn_send_window_before"),
                "conn_send_window_after": record.get("conn_send_window_after"),
                "stream_send_window_before": record.get("stream_send_window_before"),
                "stream_send_window_after": record.get("stream_send_window_after"),
                "conn_recv_window_after": record.get("conn_recv_window_after"),
                "stream_recv_window_after": record.get("stream_recv_window_after"),
                "state_before": record.get("state_before"),
                "state_after": record.get("state_after"),
                "code": record.get("code"),
                "reason_code": reason_code,
                "reason": reason_text,
            }
        )

    connection_points: list[dict[str, Any]] = []
    for record in records:
        connection_points.append(
            {
                "frame_seq": record.get("frame_seq"),
                "clock_ms": record.get("clock_ms"),
                "send_window": record.get("conn_send_window_after"),
                "recv_window": record.get("conn_recv_window_after"),
            }
        )
    stream_points: dict[int, list[dict[str, Any]]] = {sid: [] for sid in sorted(session.streams)}
    for stream_id in sorted(session.streams):
        for record in records:
            if record.get("target_stream_id") == stream_id and record.get("stream_send_window_after") is not None:
                stream_points[stream_id].append(
                    {
                        "frame_seq": record.get("frame_seq"),
                        "clock_ms": record.get("clock_ms"),
                        "send_window": record.get("stream_send_window_after"),
                        "recv_window": record.get("stream_recv_window_after"),
                    }
                )
    window_series = {
        "connection": connection_points,
        "streams": [
            {"stream_id": stream_id, "points": stream_points[stream_id]}
            for stream_id in sorted(stream_points)
        ],
    }

    transitions = sorted(
        session.ledger.transitions,
        key=lambda item: (int(item.get("transition_seq") or 0), int(item.get("stream_id") or 0)),
    )
    priority_rows = session.diagnostics()["priority_rows"]
    tree_rows = {row["stream_id"]: row for row in session.priority.rows()}
    priority_display: list[dict[str, Any]] = []
    for row in priority_rows:
        tree = tree_rows.get(row["stream_id"], {})
        depth = int(tree.get("depth", 0))
        priority_display.append(
            {
                **row,
                "depth": depth,
                "tree": f"{'  ' * depth}{row['stream_id']}",
                "share": row["share"],
                "expected_bytes": row["expected_bytes"],
                "actual_bytes": row["actual_bytes"],
                "deviation": row["deviation"],
            }
        )

    waits: list[dict[str, Any]] = []
    for wait in sorted(session.ledger.empty_waits, key=lambda item: int(item.get("wait_id", 0))):
        waits.append(
            {
                "kind": "empty_window",
                "wait_id": wait.get("wait_id"),
                "reason": wait.get("reason"),
                "streams": ",".join(str(value) for value in wait.get("affected_streams", [])),
                "start_ms": wait.get("start_ms"),
                "end_ms": wait.get("end_ms"),
                "closed_by": wait.get("closed_by"),
            }
        )
    for wait in sorted(session.ledger.queue_waits, key=lambda item: int(item.get("wait_id", 0))):
        detail = wait.get("detail", {})
        waits.append(
            {
                "kind": "queue",
                "wait_id": wait.get("wait_id"),
                "reason": wait.get("reason"),
                "streams": ",".join(str(value) for value in detail.get("queued_stream_ids", [])),
                "start_ms": wait.get("start_ms"),
                "end_ms": wait.get("end_ms"),
                "closed_by": wait.get("closed_by"),
            }
        )
    waits.sort(key=lambda item: (int(item.get("start_ms") or 0), item["kind"], int(item.get("wait_id") or 0)))

    snapshots: list[dict[str, Any]] = []
    if snapshot_store is not None:
        snapshots = snapshot_store.list_documents()

    diagnostics = session.diagnostics()
    frame_sequence_digest = digest_value(
        [
            {
                key: record.get(key)
                for key in (
                    "direction",
                    "frame_type",
                    "target_stream_id",
                    "payload_bytes",
                    "conn_send_window_after",
                    "stream_send_window_after",
                    "state_after",
                    "code",
                )
            }
            for record in records
        ]
    )
    return {
        "current_strategy": session.strategy,
        "strategy": session.strategy,
        "clock_mode": "virtual_monotonic_ms",
        "clock_ms": session.clock_ms,
        "script_index": session.script_index,
        "script_event_count": len(session.script.events),
        "declaration_version": session.declaration.declaration_version,
        "declaration_digest": session.declaration.content_digest,
        "frame_sequence_digest": frame_sequence_digest,
        "frame_timeline": frames,
        "window_series": window_series,
        "state_transitions": [
            {
                **transition,
                "transition": f"{transition.get('from')} -> {transition.get('to')}",
            }
            for transition in transitions
        ],
        "priority_rows": priority_display,
        "starvation": diagnostics["starvation"],
        "empty_waits": [item for item in waits if item["kind"] == "empty_window"],
        "queue_waits": [item for item in waits if item["kind"] == "queue"],
        "waits": waits,
        "snapshots": snapshots,
        "completion": session.completion_report(),
        "ledger_digest": digest_value(session.ledger.to_dict()),
        "frame_count": session.frame_seq,
    }
