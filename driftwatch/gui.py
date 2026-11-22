from __future__ import annotations

import json
import sys
from typing import Any

from .ledger import Ledger


def _load(db_path: str) -> dict[str, Any]:
    with Ledger(db_path) as ledger:
        return {
            "status": ledger.status(),
            "view": ledger.active_view(),
        }


def main(db_path: str) -> int:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:
        print(json.dumps({"ok": False, "code": "E_GUI", "message": str(exc)}), file=sys.stderr)
        return 2

    root = tk.Tk()
    root.title("DriftWatch Ledger")
    root.geometry("1180x720")
    frame = ttk.Frame(root, padding=8)
    frame.pack(fill="both", expand=True)
    summary = tk.Text(frame, height=10, wrap="none")
    summary.pack(fill="x")
    columns = ("snapshot", "host", "metric", "decision", "observed", "expected", "tolerance", "error")
    table = ttk.Treeview(frame, columns=columns, show="headings")
    for column in columns:
        table.heading(column, text=column)
        table.column(column, width=120 if column not in {"host", "error"} else 210)
    table.pack(fill="both", expand=True, pady=(8, 0))

    def refresh() -> None:
        payload = _load(db_path)
        summary.delete("1.0", "end")
        summary.insert("end", json.dumps(payload["status"], ensure_ascii=False, indent=2))
        for item in table.get_children():
            table.delete(item)
        for row in payload["view"]["drift_view"]:
            table.insert(
                "",
                "end",
                values=(
                    row.get("snapshot_seq"),
                    row.get("host_id"),
                    row.get("metric_id"),
                    row.get("decision"),
                    row.get("observed"),
                    row.get("expected"),
                    row.get("tolerance_bps"),
                    row.get("reason"),
                ),
            )

    controls = ttk.Frame(frame)
    controls.pack(fill="x", pady=(8, 0))
    ttk.Button(controls, text="Refresh", command=refresh).pack(side="left")
    ttk.Label(controls, text=f"SQLite: {db_path}").pack(side="left", padx=12)
    refresh()
    root.mainloop()
    return 0
