from __future__ import annotations

from pathlib import Path

from .ledger import AuditLedger
from .replay import replay_ledger


def build_audit_view(ledger_path: str | Path) -> dict:
    """Build a stable, read-only projection for the Tkinter audit UI."""

    replay = replay_ledger(ledger_path, None)
    ledger = AuditLedger(ledger_path, readonly=True, verify=False)
    try:
        meta = ledger.meta()
        events = ledger.events()
        unknown = ledger.unknown_leases()
    finally:
        ledger.close()

    unknown_by_source: dict[str, list[dict]] = {}
    for item in unknown:
        unknown_by_source.setdefault(item["source_id"], []).append(item)

    source_rows: list[dict] = []
    for source in sorted(replay["final_state"]["sources"], key=lambda item: item["source_id"]):
        active = sorted(source["active_leases"], key=lambda item: (item["expires_at_ms"], item["lease_id"]))
        source_rows.append(
            {
                "source_id": source["source_id"],
                "lease_share": source["lease_share"],
                "reserve_quota": source["reserve_quota"],
                "retained_quota": source["retained_quota"],
                "active_granted": source["active_granted"],
                "watermark_ratio_ppm": source["last_ratio_ppm"],
                "watermark_percent": f"{source['last_ratio_ppm'] / 10_000:.2f}%",
                "reclaimed_total": source["reclaimed_total"],
                "last_reclaim_basis": source["last_reclaim_basis"],
                "low_streak": source["low_streak"],
                "active_leases": active,
                "active_expiries_ms": [lease["expires_at_ms"] for lease in active],
                "unknown_leases": sorted(
                    unknown_by_source.get(source["source_id"], []),
                    key=lambda item: (item.get("event_seq") or 0, item["unknown_id"]),
                ),
                "unknown_marker": bool(unknown_by_source.get(source["source_id"])),
            }
        )

    event_rows: list[dict] = []
    for event in events:
        event_rows.append(
            {
                "seq": event.seq,
                "mono_ms": event.mono_ms,
                "event_type": event.event_type,
                "source_id": event.source_id or "",
                "lease_id": event.lease_id or "",
                "amount": "" if event.amount is None else event.amount,
                "decision": event.payload.get("decision", ""),
                "basis": event.payload.get("basis", ""),
                "ownership_digest": event.payload.get("ownership_digest", ""),
                "event_hash": event.event_hash,
            }
        )

    return {
        "format_version": meta["format_version"],
        "content_digest": replay["content_digest"],
        "event_count": replay["event_count"],
        "last_seq": meta["last_seq"],
        "params_version": meta["params_version"],
        "params_digest": meta["params_digest"],
        "scenario_version": meta["scenario_version"],
        "scenario_digest": meta["scenario_digest"],
        "external_digest": meta["external_digest"],
        "global_allocatable": replay["final_state"]["global_allocatable"],
        "total_active": replay["final_state"]["total_active"],
        "retained_total": replay["final_state"]["retained_total"],
        "cluster_capacity": replay["final_state"]["cluster_capacity"],
        "ownership_digest": replay["final_state"]["ownership_digest"],
        "sources": source_rows,
        "events": event_rows,
        "unknown_leases": sorted(
            unknown,
            key=lambda item: (item.get("event_seq") or 0, item["source_id"], item["unknown_id"]),
        ),
    }


def view_as_text(view: dict) -> str:
    lines = [
        f"ledger_format={view['format_version']} content_digest={view['content_digest']}",
        f"params_version={view['params_version']} scenario_version={view['scenario_version']}",
        f"capacity={view['cluster_capacity']} retained_total={view['retained_total']} "
        f"global_allocatable={view['global_allocatable']} total_active={view['total_active']}",
        "sources:",
    ]
    for source in view["sources"]:
        lines.append(
            "  "
            f"source={source['source_id']} share={source['lease_share']} "
            f"reserve={source['reserve_quota']} retained={source['retained_quota']} "
            f"active={source['active_granted']} watermark_ppm={source['watermark_ratio_ppm']} "
            f"reclaimed={source['reclaimed_total']} unknown={int(source['unknown_marker'])} "
            f"expiries={','.join(str(value) for value in source['active_expiries_ms']) or '-'}"
        )
    lines.append("events:")
    for event in view["events"]:
        lines.append(
            "  "
            f"seq={event['seq']} mono_ms={event['mono_ms']} type={event['event_type']} "
            f"source={event['source_id'] or '-'} lease={event['lease_id'] or '-'} "
            f"amount={event['amount']} decision={event['decision']}"
        )
    return "\n".join(lines)
