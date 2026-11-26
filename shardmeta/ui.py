"""Tkinter fast view over the cached merge state (requirement 6).

The UI only performs one cached read per query and shows the monotonic-clock
millisecond latency. It never invokes authoritative recompute; that path is a
separate command run outside the interactive read.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

from .engine import AccountingEngine


class FastViewApp:
    def __init__(self, engine: AccountingEngine, keys: list[str],
                 title: str = "shardmeta fast view") -> None:
        self.engine = engine
        self.keys = sorted(keys)
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("860x520")
        self._build()
        if self.keys:
            self.key_var.set(self.keys[0])
            self.refresh()

    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Object key:").pack(side=tk.LEFT)
        self.key_var = tk.StringVar()
        self.selector = ttk.Combobox(
            top, textvariable=self.key_var, values=self.keys, state="readonly", width=28,
        )
        self.selector.pack(side=tk.LEFT, padx=6)
        self.selector.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        ttk.Button(top, text="Cached read", command=self.refresh).pack(side=tk.LEFT, padx=6)
        self.latency_var = tk.StringVar(value="latency: - ms (cache)")
        ttk.Label(top, textvariable=self.latency_var).pack(side=tk.RIGHT)

        columns = ("shard", "merge_id", "digest", "node", "meta_ver", "fmt")
        self.tree = ttk.Treeview(self.root, columns=columns, show="headings")
        headers = {
            "shard": "Shard", "merge_id": "Merge ID", "digest": "Shard digest",
            "node": "Node", "meta_ver": "Meta ver", "fmt": "Format",
        }
        widths = {"shard": 60, "merge_id": 160, "digest": 160,
                  "node": 90, "meta_ver": 80, "fmt": 70}
        for column in columns:
            self.tree.heading(column, text=headers[column])
            self.tree.column(column, width=widths[column], anchor=tk.W)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.status_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status_var, padding=6,
                  anchor=tk.W).pack(fill=tk.X)

    def refresh(self) -> None:
        key = self.key_var.get()
        result = self.engine.fast_read(key)
        view = result["view"]
        for item in self.tree.get_children():
            self.tree.delete(item)
        for row in view["adopted"]:
            self.tree.insert(
                "", tk.END,
                values=(row["shard_index"], row["merge_id"], row["shard_digest"],
                        row["node_id"], row["meta_version"], row["source_format"]),
            )
        watermark = view["lease_watermark"]
        self.status_var.set(
            f"status={view['status']}  rejected={view['rejected_count']}  "
            f"leases active={watermark['active']}/{watermark['total']}  "
            f"min_expires_ms={watermark['min_expires_ms']}  "
            f"max_lease_version={watermark['max_lease_version']}  "
            f"(served from view cache, no recompute)"
        )
        self.latency_var.set(f"latency: {result['elapsed_ms']} ms (cache)")

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        self.root.destroy()


def launch_or_smoke(engine: AccountingEngine, keys: list[str],
                    smoke: bool = False) -> Optional[dict]:
    """Launch the UI, or perform a headless smoke cycle without mainloop."""
    app = FastViewApp(engine, keys)
    if smoke:
        # Exercise every cached read once, then close without entering mainloop.
        snapshots = []
        for key in app.keys:
            app.key_var.set(key)
            app.refresh()
            snapshots.append({
                "key": key,
                "latency_ms": app.engine.read_observations[-1].elapsed_ms,
                "status": app.status_var.get(),
            })
        app.close()
        return {"smoke": True, "reads": snapshots}
    app.run()
    return None


__all__ = ["FastViewApp", "launch_or_smoke"]
