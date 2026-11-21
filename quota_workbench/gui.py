"""Tkinter diagnostics interface (read-only over the ledger).

Shows tenant quota balances and in-flight counts, instance health and
transition reasons, probe rounds, request outcome distribution, affected
sets, current versions and the snapshot list. Rendering order is stable
(sorted keys / ORDER BY) and refreshing never mutates the ledger.
"""

from __future__ import annotations

import os
import sqlite3


def run_gui(db_path: str, snapshot_dir: str) -> None:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Quota Arbitration Workbench")
    root.geometry("980x720")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    trees = {}

    def add_tree(title, columns):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True)
        trees[title] = tree
        return tree

    add_tree("Tenants", ("tenant_id", "share", "consumed", "remaining", "inflight"))
    add_tree("Instances", ("instance_id", "to_state", "probe_round", "consec_failures", "reason", "virt_ms"))
    add_tree("Outcomes", ("outcome", "count"))
    add_tree("Propagation", ("seq", "change", "affected"))
    add_tree("Versions", ("key", "value"))
    add_tree("Snapshots", ("snapshot", "status"))

    status_var = tk.StringVar(value="")
    status = ttk.Label(root, textvariable=status_var, anchor=tk.W)
    status.pack(fill=tk.X)

    def refresh():
        for tree in trees.values():
            for item in tree.get_children():
                tree.delete(item)
        if not os.path.exists(db_path):
            status_var.set(f"ledger not found: {db_path}")
            return
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            for row in conn.execute(
                "SELECT tenant_id, share, consumed, share - consumed, inflight"
                " FROM tenants ORDER BY tenant_id"
            ):
                trees["Tenants"].insert("", tk.END, values=row)
            for row in conn.execute(
                "SELECT instance_id, to_state, probe_round, consec_failures, reason, virt_ms"
                " FROM transitions ORDER BY seq"
            ):
                trees["Instances"].insert("", tk.END, values=row)
            for row in conn.execute(
                "SELECT COALESCE(outcome, state), COUNT(*) FROM requests"
                " GROUP BY COALESCE(outcome, state) ORDER BY 1"
            ):
                trees["Outcomes"].insert("", tk.END, values=row)
            for row in conn.execute(
                "SELECT seq, change, affected FROM propagation ORDER BY seq"
            ):
                trees["Propagation"].insert("", tk.END, values=row)
            for row in conn.execute("SELECT key, value FROM meta ORDER BY key"):
                trees["Versions"].insert("", tk.END, values=row)
        finally:
            conn.close()
        if os.path.isdir(snapshot_dir):
            for name in sorted(os.listdir(snapshot_dir)):
                if name.endswith(".json"):
                    trees["Snapshots"].insert("", tk.END, values=(name, "present"))
        status_var.set(f"db={db_path} snapshots={snapshot_dir} (read-only refresh)")

    refresh_btn = ttk.Button(root, text="Refresh", command=refresh)
    refresh_btn.pack(anchor=tk.E)
    refresh()
    root.mainloop()
