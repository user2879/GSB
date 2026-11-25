from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .errors import ArbitrationError


def _read_only_connection(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def build_view_model(db_path: str | Path) -> dict[str, Any]:
    conn = _read_only_connection(db_path)
    try:
        meta = {row["key"]: row["value"] for row in conn.execute("SELECT key,value FROM meta ORDER BY key")}
        nodes = [dict(row) for row in conn.execute("SELECT * FROM node_state ORDER BY node_id")]
        for node in nodes:
            node["remaining_slots"] = node["capacity_slots"] - node["used_slots"]
            node["lease_remaining"] = node["lease_limit"] - node["active_leases"]
            scale = int(meta.get("watermark_scale", "1000"))
            high = int(meta.get("high_watermark_fixed", str(scale * 8 // 10)))
            node["watermark_fixed"] = (node["used_slots"] * scale) // node["capacity_slots"]
            node["watermark_scale"] = scale
            node["high_watermark"] = node["watermark_fixed"] >= high

        arbitrations = []
        for row in conn.execute(
            """
            SELECT event_seq, artifact_key, shard_index, node_id, quota_version, payload_json
              FROM conclusions
             WHERE kind='arbitration' AND status='current'
             ORDER BY artifact_key ASC, shard_index ASC
            """
        ):
            payload = json.loads(row["payload_json"])
            arbitrations.append(
                {
                    "event_seq": row["event_seq"],
                    "artifact_key": row["artifact_key"],
                    "shard_index": row["shard_index"],
                    "winner_node_id": row["node_id"],
                    "quota_version": row["quota_version"],
                    "indistinguishable_node_ids": payload.get("indistinguishable_node_ids", []),
                    "basis": payload.get("basis"),
                }
            )

        reservations = _reservation_rows(conn)
        queue_events = [
            {
                "event_seq": row["event_seq"],
                "node_id": row["node_id"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in conn.execute(
                """
                SELECT event_seq,node_id,payload_json FROM records
                 WHERE record_type='queue_metric'
                 ORDER BY event_seq ASC, id ASC
                """
            )
        ]
        unknown = [dict(row) for row in conn.execute("SELECT batch_id,event_seq,status FROM unknown_batches ORDER BY event_seq ASC")]
        stale_count = int(conn.execute("SELECT COUNT(*) AS count FROM conclusions WHERE status='stale'").fetchone()["count"])
        return {
            "ledger_path": str(Path(db_path)),
            "ledger_format_version": int(meta.get("ledger_format_version", "0")),
            "ledger_digest": meta.get("head_digest", ""),
            "current_quota_version": int(meta.get("current_quota_version", "1")),
            "nodes": nodes,
            "arbitrations": arbitrations,
            "reservations": reservations,
            "queue_events": queue_events,
            "unknown_batches": unknown,
            "stale_conclusion_count": stale_count,
        }
    finally:
        conn.close()


def _reservation_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT event_seq, artifact_key, node_id, reservation_id, payload_json
          FROM records
         WHERE record_type='reserve'
         ORDER BY event_seq ASC, id ASC
        """
    ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        reservation_id = row["reservation_id"]
        outcome_row = conn.execute(
            """
            SELECT record_type FROM records
             WHERE reservation_id=? AND record_type IN ('rollback','lease_grant')
             ORDER BY id DESC LIMIT 1
            """,
            (reservation_id,),
        ).fetchone()
        result.append(
            {
                "event_seq": row["event_seq"],
                "artifact_key": row["artifact_key"],
                "node_id": row["node_id"],
                "reservation_id": reservation_id,
                "outcome": outcome_row["record_type"] if outcome_row else "open_or_unknown",
                "payload": json.loads(row["payload_json"]),
            }
        )
    return result


def render_text_snapshot(model: dict[str, Any]) -> str:
    lines = [
        f"Ledger v{model['ledger_format_version']} quota=v{model['current_quota_version']} digest={model['ledger_digest'][:16]}",
        f"Unknown: {len([item for item in model['unknown_batches'] if item['status'] == 'open'])} open; stale conclusions archived: {model['stale_conclusion_count']}",
        "[Watermarks]",
    ]
    for node in model["nodes"]:
        high_mark = " HIGH" if node.get("high_watermark") else ""
        lines.append(
            f"  {node['node_id']}: wm={node['watermark_fixed']}/{node['watermark_scale']}{high_mark} "
            f"used={node['used_slots']} remaining={node['remaining_slots']} "
            f"leases={node['active_leases']}/{node['lease_limit']} lease_remaining={node['lease_remaining']}"
        )
    lines.append("[Pre-reservation Attribution]")
    for item in model["reservations"]:
        lines.append(f"  seq={item['event_seq']} {item['artifact_key']} -> {item['node_id']} {item['reservation_id']} outcome={item['outcome']}")
    lines.append("[Unique Arbitration / Indistinguishable Marks]")
    for item in model["arbitrations"]:
        marks = ",".join(item["indistinguishable_node_ids"]) or "-"
        lines.append(f"  {item['artifact_key']}#{item['shard_index']} -> {item['winner_node_id']} indistinguishable=[{marks}] basis={item['basis']}")
    lines.append("[Queue Events]")
    for item in model["queue_events"]:
        metric = item["payload"]
        lines.append(f"  seq={item['event_seq']} node={item['node_id']} queued={metric.get('queued')} wait_ms={metric.get('wait_ms')}")
    lines.append("[Unknown Marks]")
    for item in model["unknown_batches"]:
        lines.append(f"  seq={item['event_seq']} {item['batch_id']} status={item['status']}")
    return "\n".join(lines)


def run_tk(db_path: str | Path) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:
        raise ArbitrationError(
            "Tkinter is unavailable in this Python installation; install the OS Tk runtime and retry",
            code="E_UI",
            field="tkinter",
        ) from exc

    root = tk.Tk()
    root.title("Artifact Cache Lease Arbitration Diagnostics")
    root.geometry("1180x720")

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    title = ttk.Label(top, text="")
    title.pack(side="left")
    status = ttk.Label(top, text="Read-only refresh; ledger is never written by UI")
    status.pack(side="right")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)
    tables: dict[str, ttk.Treeview] = {}

    def make_table(name: str, columns: list[str]) -> ttk.Treeview:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=name)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=160, anchor="w")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        tables[name] = tree
        return tree

    node_table = make_table("Watermarks", ["node", "wm", "used", "remaining", "leases", "lease_remaining", "quota"])
    reserve_table = make_table("Reservations", ["seq", "artifact", "node", "reservation", "outcome"])
    arbitration_table = make_table("Arbitration", ["artifact", "shard", "winner", "indistinguishable", "version", "basis"])
    queue_table = make_table("Queue", ["seq", "node", "queued", "wait_start", "wait_end", "wait_ms"])
    unknown_table = make_table("Unknown", ["seq", "batch", "status"])

    def refresh() -> None:
        model = build_view_model(db_path)
        title.configure(text=f"Ledger v{model['ledger_format_version']}  {model['ledger_digest'][:24]}")
        for tree in tables.values():
            for item in tree.get_children():
                tree.delete(item)
        for node in model["nodes"]:
            node_table.insert("", "end", values=(
                node["node_id"],
                f"{node['watermark_fixed']}/{node['watermark_scale']}",
                node["used_slots"],
                node["remaining_slots"],
                f"{node['active_leases']}/{node['lease_limit']}",
                node["lease_remaining"],
                model["current_quota_version"],
            ))
        for item in model["reservations"]:
            reserve_table.insert("", "end", values=(item["event_seq"], item["artifact_key"], item["node_id"], item["reservation_id"], item["outcome"]))
        for item in model["arbitrations"]:
            arbitration_table.insert("", "end", values=(
                item["artifact_key"], item["shard_index"], item["winner_node_id"],
                ",".join(item["indistinguishable_node_ids"]), item["quota_version"], item["basis"],
            ))
        for item in model["queue_events"]:
            metric = item["payload"]
            queue_table.insert("", "end", values=(
                item["event_seq"], item["node_id"], metric.get("queued"),
                metric.get("wait_start_monotonic_ms"), metric.get("wait_end_monotonic_ms"), metric.get("wait_ms"),
            ))
        for item in model["unknown_batches"]:
            unknown_table.insert("", "end", values=(item["event_seq"], item["batch_id"], item["status"]))
        status.configure(text=f"Refreshed read-only at quota v{model['current_quota_version']}; stale archived={model['stale_conclusion_count']}")

    controls = ttk.Frame(root, padding=(8, 0, 8, 8))
    controls.pack(fill="x")
    ttk.Button(controls, text="Refresh (does not write ledger)", command=refresh).pack(side="left")
    refresh()
    root.mainloop()
