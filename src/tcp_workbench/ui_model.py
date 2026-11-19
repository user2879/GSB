"""Deterministic, toolkit-independent view model used by Tkinter and CLI."""

from __future__ import annotations

from typing import Any

from .models import canonical_json
from .projection import ITEM_LABELS, ITEM_ORDER
from .service import SessionService


def _value_text(value: Any, limit: int = 160) -> str:
    text = canonical_json(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_ui_model(service: SessionService, *, log_limit: int = 30) -> dict[str, Any]:
    current_event_seq = service.events[-1].event_seq if service.events else 0
    conclusion_rows: list[dict[str, Any]] = []
    for key in ITEM_ORDER:
        item = service.items[key]
        conclusion_rows.append(
            {
                "item": key,
                "label": ITEM_LABELS[key],
                "value": _value_text(item.value),
                "revision": item.revision,
                "basis_event_seq": item.basis_event_seq,
                "depends_on_events": ",".join(
                    str(value) for value in item.depends_on_events[-12:]
                ),
                "affected": "YES"
                if item.last_affected_event_seq == current_event_seq
                and current_event_seq > 0
                else "-",
                "version": f"C{item.config_version}/S{item.script_version}",
                "stale": "YES" if item.stale else "-",
            }
        )

    snapshots = []
    for row in service.ledger.snapshots():
        snapshots.append(
            {
                "snapshot_id": int(row["snapshot_id"]),
                "name": str(row["name"]),
                "config_version": row["config_version"],
                "script_version": row["script_version"],
                "event_count": int(row["event_count"]),
                "created_mono_ms": int(row["created_mono_ms"]),
                "content_digest": str(row["content_digest"])[:12],
            }
        )

    logger = service.logger
    logs = list(logger.tail(log_limit)) if logger is not None else []
    recovery = (
        None if service.recovery_report is None else service.recovery_report.as_dict()
    )
    return {
        "header": {
            "session_id": service.config.session_id,
            "connection_state": service.items["connection_state"].value,
            "cwnd": service.items["cwnd"].value,
            "ssthresh": service.items["ssthresh"].value,
            "dup_ack_count": service.items["dup_ack_count"].value,
            "sent_unacked": service.items["sent_unacked"].value,
            "rtt": service.items["rtt"].value,
            "config_version": service.config.config_version,
            "script_version": service.script.script_version,
            "version": f"C{service.config.config_version}/S{service.script.script_version}",
            "event_count": len(service.events),
            "needs_rerun": service.needs_rerun,
        },
        "conclusion_rows": conclusion_rows,
        "snapshots": snapshots,
        "logs": logs,
        "recovery": recovery,
        "stable_item_order": list(ITEM_ORDER),
    }


def render_ui_text(model: dict[str, Any]) -> str:
    header = model["header"]
    rtt = header["rtt"]
    lines = [
        f"session={header['session_id']} version={header['version']} events={header['event_count']}",
        (
            f"state={header['connection_state']} cwnd={header['cwnd']} ssthresh={header['ssthresh']} "
            f"dup_ack={header['dup_ack_count']} sent_unacked={header['sent_unacked']['bytes']}B/"
            f"{header['sent_unacked']['segments']}seg srtt={rtt['srtt_ms']}ms rto={rtt['rto_ms']}ms"
        ),
    ]
    if model.get("recovery"):
        lines.append(f"recovery={canonical_json(model['recovery'])}")
    lines.append("ITEM | VALUE | REV | BASIS | AFFECTED | VERSION | DEPENDS")
    for row in model["conclusion_rows"]:
        lines.append(
            f"{row['item']} | {row['value']} | {row['revision']} | {row['basis_event_seq']} | "
            f"{row['affected']} | {row['version']} | {row['depends_on_events']}"
        )
    lines.append("SNAPSHOTS")
    for snapshot in model["snapshots"]:
        lines.append(
            f"#{snapshot['snapshot_id']} {snapshot['name']} C{snapshot['config_version']}/S{snapshot['script_version']} "
            f"events={snapshot['event_count']} digest={snapshot['content_digest']}"
        )
    return "\n".join(lines)
