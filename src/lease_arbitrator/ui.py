from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .envcheck import check_tkinter


def launch_report_viewer(report_path: str | Path) -> int:
    tk_info = check_tkinter()
    if not tk_info.get("available"):
        raise RuntimeError(f"tkinter is unavailable: {tk_info.get('error')}")
    import tkinter as tk
    from tkinter import ttk

    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    root = tk.Tk()
    root.title("Lease Arbitration Report")
    root.geometry("1100x650")
    paned = ttk.PanedWindow(root, orient="horizontal")
    paned.pack(fill="both", expand=True)

    columns = ("lease", "node", "state", "generation", "wait_ms", "version")
    tree = ttk.Treeview(paned, columns=columns, show="headings")
    for column, text in zip(
        columns, ("Lease", "Node", "State", "Gen", "Wait ms", "Ver"), strict=True
    ):
        tree.heading(column, text=text)
        tree.column(column, width=130)
    paned.add(tree, weight=1)

    text = tk.Text(paned, wrap="none")
    paned.add(text, weight=2)

    leases: list[dict[str, Any]] = report.get("current_view", {}).get("leases", [])
    for lease in leases:
        tree.insert(
            "",
            "end",
            iid=lease["lease_id"],
            values=(
                lease["lease_id"],
                lease["node_id"],
                lease["state"],
                lease["generation_current"],
                lease.get("wait_ms"),
                lease.get("quota_version"),
            ),
        )

    def show_lease(_event: Any = None) -> None:
        selection = tree.selection()
        if not selection:
            return
        lease_id = selection[0]
        lease = next((item for item in leases if item["lease_id"] == lease_id), None)
        text.delete("1.0", "end")
        text.insert("1.0", json.dumps(lease, ensure_ascii=False, indent=2, sort_keys=True))

    tree.bind("<<TreeviewSelect>>", show_lease)
    text.insert("1.0", json.dumps(report.get("current_view", {}).get("water", {}), indent=2))
    root.mainloop()
    return 0
