"""Optional Tkinter report viewer.

Tkinter ships with the official Windows Python installer. On minimal Linux
images without the system Tk extension the CLI degrades gracefully with an
E_GRID-style environment message instead of crashing.
"""
from __future__ import annotations

import json
from pathlib import Path


def view_report(report_path: str | Path) -> bool:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("Tkinter is unavailable in this Python installation.")
        print("On Windows use the official installer from python.org (Tcl/Tk is")
        print("included by default); on Debian/Ubuntu install python3-tk.")
        return False

    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    root = tk.Tk()
    root.title("graymirror fixed point report")
    columns = ("rule_id", "total", "stable", "gray", "mirrored",
               "route_bound", "mirror_bound", "e2e_bound")
    tree = ttk.Treeview(root, columns=columns, show="headings")
    for column in columns:
        tree.heading(column, text=column)
        tree.column(column, width=130, anchor="center")
    for bucket in payload["statistics"]["per_rule"].values():
        bounds = bucket["bounds"]
        tree.insert("", "end", values=(
            bucket["rule_id"], bucket["total"], bucket["stable"],
            bucket["gray"], bucket["mirrored"],
            bounds["route_share_bound_per_10000"],
            bounds["mirror_abs_bound_per_10000"],
            bounds["end_to_end_bound_per_10000"],
        ))
    tree.pack(expand=True, fill="both")
    summary = tk.Label(
        root,
        text=(f"version {payload['rule_version']} | scale 10**"
              f"{payload['scale_exponent']} | "
              f"e2e bound {payload['statistics']['global']['global_end_to_end_bound_per_10000']} per-10000"),
        anchor="w",
    )
    summary.pack(fill="x")
    root.mainloop()
    return True


if __name__ == "__main__":
    import sys

    view_report(sys.argv[1] if len(sys.argv) > 1 else "samples/report-k2.json")
