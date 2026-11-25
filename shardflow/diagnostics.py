"""Tkinter diagnostic window.

Tkinter is imported lazily so the rest of the package and all headless tests
work on machines without the Tk native libraries. Refreshing the window only
rebuilds the read-only view model and never writes to the ledger.
"""

from .viewmodel import build_view_model, render_text


class DiagnosticApp:
    def __init__(self, root, spec_path, ledger_path, events_path=None):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.spec_path = spec_path
        self.ledger_path = ledger_path
        self.events_path = events_path
        root.title("Shardflow Diagnostics")
        root.geometry("980x720")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.text_tab = tk.Frame(self.notebook)
        self.node_tab = tk.Frame(self.notebook)
        self.shard_tab = tk.Frame(self.notebook)
        self.invalidation_tab = tk.Frame(self.notebook)
        self.snapshot_tab = tk.Frame(self.notebook)
        self.notebook.add(self.text_tab, text="Overview")
        self.notebook.add(self.node_tab, text="Nodes")
        self.notebook.add(self.shard_tab, text="Shards")
        self.notebook.add(self.invalidation_tab, text="Invalidations")
        self.notebook.add(self.snapshot_tab, text="Ledger Snapshots")

        self.text_widget = tk.Text(self.text_tab, wrap=tk.NONE)
        self.text_widget.pack(fill=tk.BOTH, expand=True)

        self.node_tree = self._tree(
            self.node_tab,
            ("node_id", "port", "lease_events", "active_leases", "lease_limit",
             "concurrency",
             "capacity_version", "units"),
        )
        self.shard_tree = self._tree(
            self.shard_tab,
            ("shard_id", "node_id", "status", "attempt", "config_version",
             "capacity_version", "unknown"),
        )
        self.invalid_tree = self._tree(
            self.invalidation_tab, ("revision", "shard_id", "reason", "basis")
        )
        self.snapshot_tree = self._tree(
            self.snapshot_tab, ("note", "wall_ms", "monotonic_ms", "digest")
        )

        button_bar = tk.Frame(root)
        button_bar.pack(fill=tk.X)
        ttk.Button(button_bar, text="Refresh (read-only)",
                   command=self.refresh).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(button_bar, text="Close",
                   command=root.destroy).pack(side=tk.RIGHT, padx=4, pady=4)
        self.refresh()
        self.root.after(2000, self._auto_refresh)

    def _tree(self, parent, columns):
        import tkinter as tk
        from tkinter import ttk

        tree = ttk.Treeview(parent, columns=columns, show="headings")
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=140, anchor=tk.W)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL,
                                  command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        return tree

    def refresh(self):
        model = build_view_model(
            self.spec_path, self.ledger_path, self.events_path
        )
        self.text_widget.delete("1.0", "end")
        self.text_widget.insert("1.0", render_text(model))
        self._replace_rows(
            self.node_tree,
            [
                (
                    row["node_id"],
                    row["port"],
                    row["lease_event_count"],
                    row["active_lease_count"],
                    row["lease_limit"],
                    row["concurrency"],
                    row["capacity_version"],
                    row["resource_units_seen"],
                )
                for row in model["nodes"]
            ],
        )
        self._replace_rows(
            self.shard_tree,
            [
                (
                    row["shard_id"],
                    row["node_id"],
                    row["status"],
                    row["attempt"],
                    row["config_version"],
                    row["capacity_version"],
                    "yes" if row["unknown"] else "no",
                )
                for row in model["shards"]
            ],
        )
        self._replace_rows(
            self.invalid_tree,
            [
                (
                    row["revision"],
                    row["shard_id"],
                    row["reason"],
                    str(row["basis"]),
                )
                for row in model["invalidations"]
            ],
        )
        self._replace_rows(
            self.snapshot_tree,
            [
                (
                    row["note"],
                    row["wall_ms"],
                    row["monotonic_ms"],
                    row["content_digest"],
                )
                for row in model["snapshots"]
            ],
        )
        return model

    def _auto_refresh(self):
        if self.root.winfo_exists():
            self.refresh()
            self.root.after(2000, self._auto_refresh)

    @staticmethod
    def _replace_rows(tree, rows):
        for item in tree.get_children():
            tree.delete(item)
        for row in rows:
            tree.insert("", "end", values=row)


def launch_tk(spec_path, ledger_path, events_path=None):
    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Tkinter is unavailable in this Python installation: %s. On "
            "Windows use the official installer, which includes Tkinter." % exc
        ) from exc
    root = tk.Tk()
    DiagnosticApp(root, spec_path, ledger_path, events_path)
    root.mainloop()
    return 0
