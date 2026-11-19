"""Optional Tkinter diagnostics window (standard-library only)."""

from __future__ import annotations

from typing import Any

from .engine import diagnose, manifest
from .snapshot import list_snapshots
from .storage import load_collection_specs, open_session


def _first_names(session: str) -> tuple[str, str]:
    conn = open_session(session)
    try:
        specs = load_collection_specs(conn)
    finally:
        conn.close()
    if not specs:
        raise RuntimeError("session has no collections")
    collection = specs[0]
    if not collection.indexes:
        raise RuntimeError("collection has no indexes")
    return collection.name, collection.indexes[0].path


def run_gui(
    session: str,
    *,
    collection: str | None = None,
    index_path: str | None = None,
) -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    default_collection, default_index = _first_names(session)
    collection = collection or default_collection
    index_path = index_path or default_index

    root = tk.Tk()
    root.title(f"Document Range Index Diagnostics - {session}")
    root.geometry("1180x720")

    status = tk.StringVar(value="Ready")
    toolbar = ttk.Frame(root, padding=(8, 6))
    toolbar.pack(fill=tk.X)
    ttk.Label(
        toolbar,
        text=f"collection={collection}  index={index_path}",
    ).pack(side=tk.LEFT)
    ttk.Label(toolbar, textvariable=status).pack(side=tk.LEFT, padx=20)

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    segment_frame = ttk.Frame(notebook)
    ledger_frame = ttk.Frame(notebook)
    isolation_frame = ttk.Frame(notebook)
    snapshot_frame = ttk.Frame(notebook)
    notebook.add(segment_frame, text="Segments")
    notebook.add(ledger_frame, text="Stale ledger")
    notebook.add(isolation_frame, text="Isolation filters")
    notebook.add(snapshot_frame, text="Snapshots")

    segment_columns = (
        "segment",
        "low_key",
        "low_pk",
        "high_key",
        "high_pk",
        "count",
        "version",
        "digest",
        "stale",
        "filtered",
    )
    segment_tree = ttk.Treeview(
        segment_frame,
        columns=segment_columns,
        show="headings",
        height=20,
    )
    for column in segment_columns:
        segment_tree.heading(column, text=column)
        segment_tree.column(column, width=110, anchor=tk.W)
    segment_tree.column("segment", width=130)
    segment_tree.column("digest", width=240)
    segment_tree.pack(fill=tk.BOTH, expand=True)

    ledger_columns = (
        "primary_key",
        "stale_key",
        "reason",
        "original_segment",
        "discovered_version",
        "last_seen_version",
    )
    ledger_tree = ttk.Treeview(
        ledger_frame,
        columns=ledger_columns,
        show="headings",
        height=20,
    )
    for column in ledger_columns:
        ledger_tree.heading(column, text=column)
        ledger_tree.column(column, width=150)
    ledger_tree.column("discovered_version", width=180)
    ledger_tree.column("last_seen_version", width=180)
    ledger_tree.pack(fill=tk.BOTH, expand=True)

    isolation_columns = (
        "segment",
        "stale_entry_count",
        "filtered_count",
        "stale_keys",
    )
    isolation_tree = ttk.Treeview(
        isolation_frame,
        columns=isolation_columns,
        show="headings",
        height=20,
    )
    for column in isolation_columns:
        isolation_tree.heading(column, text=column)
    isolation_tree.column("segment", width=150)
    isolation_tree.column("stale_entry_count", width=160)
    isolation_tree.column("filtered_count", width=160)
    isolation_tree.column("stale_keys", width=650)
    isolation_tree.pack(fill=tk.BOTH, expand=True)

    snapshot_columns = (
        "label",
        "collection_versions",
        "index_versions",
        "segments",
        "ledger",
        "content_digest",
    )
    snapshot_tree = ttk.Treeview(
        snapshot_frame,
        columns=snapshot_columns,
        show="headings",
        height=20,
    )
    for column in snapshot_columns:
        snapshot_tree.heading(column, text=column)
    snapshot_tree.column("label", width=150)
    snapshot_tree.column("collection_versions", width=180)
    snapshot_tree.column("index_versions", width=210)
    snapshot_tree.column("segments", width=100)
    snapshot_tree.column("ledger", width=100)
    snapshot_tree.column("content_digest", width=420)
    snapshot_tree.pack(fill=tk.BOTH, expand=True)

    def refresh() -> None:
        try:
            report = diagnose(session, collection, index_path)
            state = manifest(session)
            snaps = list_snapshots(session)
        except Exception as exc:  # display diagnostics even when the DB is faulty
            status.set(f"ERROR: {exc}")
            messagebox.showerror("Diagnostics failed", str(exc))
            return
        for tree in (segment_tree, ledger_tree, isolation_tree, snapshot_tree):
            for item in tree.get_children():
                tree.delete(item)
        for segment in report["segments"]:
            segment_tree.insert(
                "",
                tk.END,
                values=(
                    segment["segment_id"],
                    segment["low_key"],
                    segment["low_pk"],
                    segment["high_key"],
                    segment["high_pk"],
                    segment["entry_count"],
                    segment["version"],
                    segment["digest"],
                    segment["stale_entry_count"],
                    segment["filtered_count"],
                ),
            )
        for row in report["stale_ledger"]:
            ledger_tree.insert(
                "",
                tk.END,
                values=(
                    row["primary_key"],
                    row["stale_key"],
                    row["reason"],
                    row["original_segment"],
                    row["discovered_collection_version"],
                    row["last_seen_collection_version"],
                ),
            )
        for segment in report["segments"]:
            stale_text = ", ".join(
                f'{item["primary_key"]}@{item["value"]}'
                for item in segment["stale_keys"]
            )
            isolation_tree.insert(
                "",
                tk.END,
                values=(
                    segment["segment_id"],
                    segment["stale_entry_count"],
                    segment["filtered_count"],
                    stale_text,
                ),
            )
        for snapshot in snaps:
            snapshot_tree.insert(
                "",
                tk.END,
                values=(
                    snapshot["label"],
                    snapshot["collection_versions"],
                    snapshot["index_versions"],
                    snapshot["segment_count"],
                    snapshot["ledger_count"],
                    snapshot["content_digest"],
                ),
            )
        consistent = "consistent" if report["consistent"] else "STALE VERSIONS"
        status.set(
            f"{consistent}; collection_version={report['collection_version']} "
            f"index_version={report['index_version']}; "
            f"segments={len(state['segments'])}; ledger={report['stale_ledger_count']}"
        )

    ttk.Button(toolbar, text="Refresh", command=refresh).pack(side=tk.RIGHT)
    refresh()
    root.mainloop()
