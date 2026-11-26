"""Optional Tkinter viewer for specs and run reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import DiagnosticError


def launch(spec_path: str | None = None, report_path: str | None = None) -> int:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:
        raise DiagnosticError(
            "E_GUI",
            "tkinter is unavailable; install the system Tk package for your Python 3.11 runtime "
            "(on Ubuntu 22.04 with Python 3.11 from the feedback toolchain, provide libtk8.6 and "
            "the matching tkinter package; `sudo apt install python3-tk` covers distro Python 3.10).",
        ) from exc

    root = tk.Tk()
    root.title("Shard Dispatch Diagnostics")
    root.geometry("1180x720")

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    path_var = tk.StringVar(value=report_path or spec_path or "")
    ttk.Label(top, text="Report/spec:").pack(side="left")
    ttk.Entry(top, textvariable=path_var, width=90).pack(side="left", padx=6)

    paned = ttk.PanedWindow(root, orient="horizontal")
    paned.pack(fill="both", expand=True, padx=8, pady=8)

    left = ttk.Frame(paned)
    right = ttk.Frame(paned)
    paned.add(left, weight=3)
    paned.add(right, weight=2)

    columns = ("shard", "status", "node", "digest", "wait_ms")
    tree = ttk.Treeview(left, columns=columns, show="headings")
    for column, text, width in (
        ("shard", "Shard", 180),
        ("status", "Status", 100),
        ("node", "Node", 90),
        ("digest", "Summary digest", 260),
        ("wait_ms", "Wait ms", 80),
    ):
        tree.heading(column, text=text)
        tree.column(column, width=width, anchor="w")
    tree.pack(fill="both", expand=True)

    text = tk.Text(right, wrap="word", height=40)
    text.pack(fill="both", expand=True)

    def load_path(path: str) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def refresh() -> None:
        for item in tree.get_children():
            tree.delete(item)
        path = path_var.get().strip()
        if not path:
            return
        data = load_path(path)
        shards = data.get("shards", {})
        if isinstance(shards, dict):
            for shard_id, row in sorted(shards.items()):
                tree.insert(
                    "",
                    "end",
                    values=(
                        shard_id,
                        row.get("status"),
                        row.get("node_id"),
                        (row.get("summary_digest") or "")[:24],
                        row.get("wait_ms"),
                    ),
                )
        text.delete("1.0", "end")
        text.insert("1.0", json.dumps(data.get("water_report", data), indent=2, sort_keys=True)[:200000])

    ttk.Button(top, text="Load", command=refresh).pack(side="left", padx=4)
    root.after(50, refresh)
    root.mainloop()
    return 0
