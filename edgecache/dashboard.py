"""Stable, read-only dashboard snapshot and optional Tkinter viewer."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .errors import UiUnavailable


def dashboard_snapshot(db_path: str | Path) -> dict[str, Any]:
    path = Path(db_path)
    if not path.exists():
        return {"ok": False, "error": "ledger does not exist", "path": str(path)}
    uri = path.resolve().as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
    except sqlite3.DatabaseError as exc:
        return {"ok": False, "error": f"ledger cannot be opened read-only: {exc}", "path": str(path)}
    try:
        state_row = conn.execute("SELECT * FROM state WHERE id=1").fetchone()
        state = dict(state_row) if state_row is not None else {}
        capacity_rows = conn.execute(
            "SELECT node_id,COUNT(*) AS active_leases,COALESCE(SUM(bytes),0) AS active_bytes,GROUP_CONCAT(slot_index) AS slots FROM current_view GROUP BY node_id ORDER BY node_id"
        ).fetchall()
        capacities = [
            {
                "node_id": row["node_id"],
                "active_leases": int(row["active_leases"]),
                "active_bytes": int(row["active_bytes"]),
                "used_slots": sorted(int(value) for value in (row["slots"] or "").split(",") if value != ""),
            }
            for row in capacity_rows
        ]
        leases = [
            dict(row)
            for row in conn.execute(
                "SELECT lease_id,node_id,content_key,shard_index,slot_index,bytes,expires_ms,lease_version,quota_version,digest,status FROM current_view ORDER BY node_id,slot_index,lease_id"
            ).fetchall()
        ]
        queued = [
            dict(row)
            for row in conn.execute(
                "SELECT queue_id,event_seq,node_id,content_key,shard_index,wait_start_ms,attempt FROM queue_waits WHERE status='WAITING' ORDER BY event_seq,node_id,shard_index,queue_id"
            ).fetchall()
        ]
        unknown = [
            dict(row)
            for row in conn.execute(
                "SELECT lease_id,node_id,content_key,shard_index,slot_index,batch_id,claim_basis,claim_event_seq FROM leases WHERE status='UNKNOWN' ORDER BY node_id,slot_index,lease_id"
            ).fetchall()
        ]
        stale = [
            dict(row)
            for row in conn.execute(
                "SELECT id,quota_version,reason,event_seq,created_ms,payload_json FROM stale_views ORDER BY id"
            ).fetchall()
        ]
        batches = [
            dict(row)
            for row in conn.execute(
                "SELECT batch_id,event_seq,node_id,status,created_ms,closed_ms,error_code,attempt FROM batches ORDER BY created_ms,batch_id"
            ).fetchall()
        ]
        quota_transitions = [
            dict(row)
            for row in conn.execute(
                "SELECT quota_version,event_seq,node_ids_json,added_nodes_json,removed_nodes_json,new_leases_json,revoked_leases_json,retained_leases_json,created_ms FROM quota_history ORDER BY event_seq,quota_version"
            ).fetchall()
        ]
        for transition in quota_transitions:
            for key in (
                "node_ids_json",
                "added_nodes_json",
                "removed_nodes_json",
                "new_leases_json",
                "revoked_leases_json",
                "retained_leases_json",
            ):
                transition[key.removesuffix("_json")] = json.loads(transition.pop(key))
        return {
            "ok": True,
            "state": state,
            "watermarks": {
                "global_active_leases": len(leases),
                "global_active_bytes": sum(int(lease["bytes"]) for lease in leases),
                "nodes": capacities,
            },
            "leases": leases,
            "queued": queued,
            "unknown_leases": unknown,
            "stale_views": stale,
            "quota_transitions": quota_transitions,
            "batches": batches,
        }
    except sqlite3.DatabaseError as exc:
        return {"ok": False, "error": f"ledger read failed: {exc}", "path": str(path)}
    finally:
        conn.close()


def launch_ui(db_path: str | Path) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # Tk is optional at runtime; headless mode remains complete.
        raise UiUnavailable(f"Tkinter is unavailable: {exc}") from exc

    root = tk.Tk()
    root.title("Edge Cache Lease Ledger")
    root.geometry("1180x780")
    toolbar = ttk.Frame(root, padding=6)
    toolbar.pack(fill="x")
    ttk.Label(toolbar, text=f"Ledger: {Path(db_path).resolve()}").pack(side="left")
    body = ttk.Frame(root, padding=6)
    body.pack(fill="both", expand=True)

    sections = [
        ("Lease watermarks", "watermarks"),
        ("Active leases", "leases"),
        ("Queued applications", "queued"),
        ("Unknown leases", "unknown_leases"),
        ("Stale conclusions", "stale_views"),
        ("Quota transitions", "quota_transitions"),
        ("Ledger batches", "batches"),
    ]
    widgets: dict[str, tk.Text] = {}
    for index, (title, key) in enumerate(sections):
        frame = ttk.LabelFrame(body, text=title, padding=4)
        frame.grid(row=index // 2, column=index % 2, sticky="nsew", padx=3, pady=3)
        body.grid_columnconfigure(index % 2, weight=1)
        body.grid_rowconfigure(index // 2, weight=1)
        text = tk.Text(frame, height=10, width=62, wrap="none", font=("Consolas", 9))
        scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        widgets[key] = text

    def refresh() -> None:
        snapshot = dashboard_snapshot(db_path)
        for _title, key in sections:
            widget = widgets[key]
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", json.dumps(snapshot.get(key, snapshot), ensure_ascii=False, sort_keys=True, indent=2))
            widget.configure(state="disabled")

    ttk.Button(toolbar, text="Refresh (read-only)", command=refresh).pack(side="right")
    refresh()
    root.mainloop()
