"""Read-only Tkinter diagnostics dashboard with stable ordering."""
from __future__ import annotations

import json
from typing import Any

from .errors import BusinessError


def launch_dashboard(report: dict[str, Any], *, title: str = "Quota Ledger Diagnostics") -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # includes a Python build without the _tkinter module
        raise BusinessError("E_UI", f"Tkinter unavailable: {exc}", field="tkinter") from exc

    root = tk.Tk()
    root.title(title)
    root.geometry("1180x760")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)

    nodes_tab = ttk.Frame(notebook)
    queue_tab = ttk.Frame(notebook)
    invalidation_tab = ttk.Frame(notebook)
    lease_tab = ttk.Frame(notebook)
    ledger_tab = ttk.Frame(notebook)
    notebook.add(nodes_tab, text="Nodes / Resources")
    notebook.add(queue_tab, text="Ready / Queued")
    notebook.add(invalidation_tab, text="Invalidation")
    notebook.add(lease_tab, text="Lease Ownership")
    notebook.add(ledger_tab, text="Ledger / Snapshots")

    node_tree = _tree(
        nodes_tab,
        (
            "node",
            "concurrency",
            "lease_limit",
            "active_shards",
            "active_units",
            "active_leases",
            "max_shards",
            "max_units",
            "max_leases",
            "active_lease_ids",
        ),
    )
    queue_tree = _tree(
        queue_tab,
        ("class", "shard", "node", "status", "generation", "wait_start", "wait_end", "cancel_reason"),
    )
    inv_tree = _tree(invalidation_tab, ("event_seq", "rank", "shard", "direct", "reason"))
    lease_tree = _tree(
        lease_tab,
        ("lease_id", "shard", "node", "state", "version", "config", "generation", "units"),
    )
    snap_tree = _tree(ledger_tab, ("item", "value"))
    snap_frame = snap_tree.master

    def refresh() -> None:
        scheduler = report.get("scheduler", {})
        limits = scheduler.get("node_limits", {})
        active = scheduler.get("node_active", {})
        active_units = scheduler.get("node_units", {})
        leases = scheduler.get("leases", {}).get("node_active", {})
        max_shards = scheduler.get("max_node_active", {})
        max_units = scheduler.get("max_node_units", {})
        max_leases = scheduler.get("leases", {}).get("max_node_leases", {})
        active_lease_ids: dict[str, list[str]] = {}
        for lease in scheduler.get("leases", {}).get("active", []):
            active_lease_ids.setdefault(lease["node_id"], []).append(lease["lease_id"])
        for item in node_tree.get_children():
            node_tree.delete(item)
        for node_id in sorted(limits):
            values = limits[node_id]
            node_tree.insert(
                "",
                "end",
                values=(
                    node_id,
                    values.get("concurrency"),
                    values.get("lease_limit"),
                    active.get(node_id, 0),
                    active_units.get(node_id, 0),
                    leases.get(node_id, 0),
                    max_shards.get(node_id, 0),
                    max_units.get(node_id, 0),
                    max_leases.get(node_id, 0),
                    ",".join(sorted(active_lease_ids.get(node_id, []))) or "-",
                ),
            )

        for item in queue_tree.get_children():
            queue_tree.delete(item)
        tasks = sorted(scheduler.get("tasks", []), key=lambda row: row["shard_id"])
        for task in tasks:
            if task["status"] in {"unrequested", "completed"}:
                continue
            queue_tree.insert(
                "",
                "end",
                values=(
                    "blocked" if task["status"] in {"blocked", "cancelled_dependency"} else task["status"],
                    task["shard_id"],
                    task["node_id"],
                    task["status"],
                    f'{task["result_generation"]}/{task["requested_generation"]}',
                    task.get("wait_start_ms"),
                    task.get("wait_end_ms"),
                    task.get("cancel_reason") or "-",
                ),
            )

        for item in inv_tree.get_children():
            inv_tree.delete(item)
        for row in sorted(report.get("invalidations", []), key=lambda item: (item["event_seq"], item["topo_rank"], item["shard_id"])):
            inv_tree.insert(
                "",
                "end",
                values=(row["event_seq"], row["topo_rank"], row["shard_id"], bool(row["direct"]), row["reason"]),
            )

        for item in lease_tree.get_children():
            lease_tree.delete(item)
        for lease in sorted(report.get("leases", []), key=lambda item: item["lease_id"]):
            lease_tree.insert(
                "",
                "end",
                values=(
                    lease["lease_id"],
                    lease["shard_id"],
                    lease["node_id"],
                    lease["state"],
                    lease["version"],
                    lease["config_version"],
                    lease["generation"],
                    lease["resource_units"],
                ),
            )

        for item in snap_tree.get_children():
            snap_tree.delete(item)
        rows: list[tuple[str, Any]] = [
            ("ledger_format_version", report.get("ledger_format_version")),
            ("active_revision", report.get("active_revision")),
            ("ledger_digest", report.get("ledger_digest")),
            ("unknown_shards_on_recovery", ",".join(report.get("unknown_shards_on_recovery", [])) or "-"),
            ("completed_count", report.get("completed_count")),
            ("interrupted", report.get("interrupted")),
        ]
        for snapshot in report.get("snapshots", []):
            rows.append(
                (
                    f'snapshot#{snapshot["snapshot_id"]} event={snapshot["after_event_seq"]}',
                    snapshot["business_digest"],
                )
            )
        for key, value in rows:
            snap_tree.insert("", "end", values=(key, value))

    toolbar = ttk.Frame(root)
    toolbar.pack(fill="x", padx=8, pady=(0, 8))
    ttk.Button(toolbar, text="Refresh (read-only)", command=refresh).pack(side="left")
    ttk.Label(toolbar, text="Refresh reads the immutable run report and never writes the ledger.").pack(side="left", padx=12)
    refresh()
    root.mainloop()
    _ = snap_frame


def _tree(parent: Any, columns: tuple[str, ...]):
    import tkinter as tk
    from tkinter import ttk

    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True)
    tree = ttk.Treeview(frame, columns=columns, show="headings")
    for column in columns:
        tree.heading(column, text=column)
        tree.column(column, width=max(90, 900 // len(columns)), anchor="w")
    y = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    x = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=y.set, xscrollcommand=x.set)
    tree.grid(row=0, column=0, sticky="nsew")
    y.grid(row=0, column=1, sticky="ns")
    x.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    return tree
