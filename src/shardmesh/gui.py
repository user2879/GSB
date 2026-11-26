"""Optional Tkinter viewer. Importing tkinter is deferred to the gui command."""

from __future__ import annotations

import json
from pathlib import Path

from .ledger import Ledger
from .view import build_ledger_view


def run_tk_app(spec_path: Path, ledger_path: Path) -> int:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:
        from .errors import ShardMeshError

        raise ShardMeshError(
            "E_GUI",
            "tkinter is unavailable; install the system python3-tk package",
            category="gui",
            field_name="tkinter",
            details={"error": str(exc)},
        ) from exc

    root = tk.Tk()
    root.title("ShardMesh current view")
    root.geometry("1100x650")
    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    status = ttk.Label(top, text=f"ledger: {ledger_path}")
    status.pack(side="left")

    frame = ttk.Frame(root, padding=8)
    frame.pack(fill="both", expand=True)
    columns = ("shard", "node", "version", "state", "total", "checksum", "dominant")
    tree = ttk.Treeview(frame, columns=columns, show="headings")
    for column in columns:
        tree.heading(column, text=column)
        tree.column(column, width=140)
    tree.pack(side="left", fill="both", expand=True)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")

    detail = tk.Text(root, height=14, wrap="word")
    detail.pack(fill="x", padx=8, pady=8)

    def refresh() -> None:
        for item in tree.get_children():
            tree.delete(item)
        if not Path(ledger_path).exists():
            status.config(text=f"ledger not found: {ledger_path}")
            return
        ledger = Ledger(ledger_path)
        try:
            view = build_ledger_view(ledger)
        finally:
            ledger.close()
        status.config(text=f"current={view['current_count']} stale={view['stale_count']} cursor={view['cursor_seq']}")
        for record in view["records"]:
            summary = record.get("outcome", {}).get("summary", {})
            tree.insert(
                "",
                "end",
                iid=record["shard_id"],
                values=(
                    record["shard_id"],
                    record.get("node_id"),
                    record.get("capacity_version"),
                    record.get("state", "committed"),
                    summary.get("total"),
                    str(summary.get("checksum", ""))[:16],
                    record.get("error_attribution", {}).get("dominant_cause"),
                ),
            )

    def show_detail(_event=None) -> None:
        selected = tree.selection()
        if not selected:
            return
        ledger = Ledger(ledger_path)
        try:
            record = ledger.get_record(selected[0])
        finally:
            ledger.close()
        detail.delete("1.0", "end")
        detail.insert("1.0", json.dumps(record, sort_keys=True, indent=2, default=str))

    tree.bind("<<TreeviewSelect>>", show_detail)
    refresh_button = ttk.Button(top, text="Refresh whole view", command=refresh)
    refresh_button.pack(side="right")
    refresh()
    root.mainloop()
    return 0
