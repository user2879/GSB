"""Optional Tkinter diagnostic interface."""

from __future__ import annotations

import json
from typing import Any

from .diagnostics import dashboard_snapshot
from .errors import DiagnosticError


def launch_dashboard(store: Any) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except (ImportError, OSError) as exc:
        raise DiagnosticError(
            "E_TK_UNAVAILABLE",
            "Tkinter is unavailable; install the Python tcl/tk component",
            field="tkinter",
            details={"error": str(exc)},
        ) from exc

    try:
        root = tk.Tk()
    except Exception as exc:
        raise DiagnosticError(
            "E_TK_UNAVAILABLE",
            "Tk could not initialize a display",
            field="tkinter",
            details={"error": str(exc)},
        ) from exc
    root.title("Cacheproof diagnostics")
    root.geometry("1200x780")

    status_var = tk.StringVar(value="Loading...")
    header = ttk.Frame(root, padding=(10, 8))
    header.pack(fill="x")
    ttk.Label(header, textvariable=status_var, anchor="w").pack(
        side="left", fill="x", expand=True
    )

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    segments_frame = ttk.Frame(notebook)
    entries_frame = ttk.Frame(notebook)
    views_frame = ttk.Frame(notebook)
    unknown_frame = ttk.Frame(notebook)
    notebook.add(segments_frame, text="Segments & proofs")
    notebook.add(entries_frame, text="Entries")
    notebook.add(views_frame, text="Read views")
    notebook.add(unknown_frame, text="Unknown entries")

    segment_columns = (
        "segment",
        "capacity",
        "used",
        "remaining",
        "entries",
        "version",
        "proof_ok",
        "proof_digest",
    )
    segment_tree = ttk.Treeview(
        segments_frame, columns=segment_columns, show="headings", height=12
    )
    widths = (160, 90, 90, 90, 80, 80, 80, 520)
    for column, width in zip(segment_columns, widths):
        segment_tree.heading(column, text=column)
        segment_tree.column(column, width=width, anchor="w")
    segment_scroll = ttk.Scrollbar(
        segments_frame, orient="vertical", command=segment_tree.yview
    )
    segment_tree.configure(yscrollcommand=segment_scroll.set)
    segment_tree.pack(side="left", fill="both", expand=True)
    segment_scroll.pack(side="right", fill="y")

    entry_columns = ("segment", "key", "size", "digest", "actual", "ok")
    entry_tree = ttk.Treeview(
        entries_frame, columns=entry_columns, show="headings", height=18
    )
    for column, width in zip(entry_columns, (130, 170, 80, 480, 480, 70)):
        entry_tree.heading(column, text=column)
        entry_tree.column(column, width=width, anchor="w")
    entry_scroll = ttk.Scrollbar(
        entries_frame, orient="vertical", command=entry_tree.yview
    )
    entry_tree.configure(yscrollcommand=entry_scroll.set)
    entry_tree.pack(side="left", fill="both", expand=True)
    entry_scroll.pack(side="right", fill="y")

    views_text = tk.Text(views_frame, wrap="none", font=("TkFixedFont", 10))
    views_scroll = ttk.Scrollbar(
        views_frame, orient="vertical", command=views_text.yview
    )
    views_text.configure(yscrollcommand=views_scroll.set)
    views_text.pack(side="left", fill="both", expand=True)
    views_scroll.pack(side="right", fill="y")

    unknown_columns = (
        "segment",
        "key",
        "reason",
        "expected",
        "observed",
        "size",
        "operation",
    )
    unknown_tree = ttk.Treeview(
        unknown_frame, columns=unknown_columns, show="headings", height=18
    )
    for column, width in zip(
        unknown_columns, (130, 170, 220, 470, 470, 80, 220)
    ):
        unknown_tree.heading(column, text=column)
        unknown_tree.column(column, width=width, anchor="w")
    unknown_scroll = ttk.Scrollbar(
        unknown_frame, orient="vertical", command=unknown_tree.yview
    )
    unknown_tree.configure(yscrollcommand=unknown_scroll.set)
    unknown_tree.pack(side="left", fill="both", expand=True)
    unknown_scroll.pack(side="right", fill="y")

    def refresh() -> None:
        dashboard = dashboard_snapshot(store)
        ledger = dashboard["ledger"]
        status_var.set(
            "batch={} segment_set_version={} cursor={} code={} ledger={}".format(
                ledger.get("batch_id"),
                ledger.get("segment_set_version"),
                ledger.get("completed_cursor"),
                dashboard["code"],
                ledger.get("content_digest"),
            )
        )
        for tree in (segment_tree, entry_tree, unknown_tree):
            tree.delete(*tree.get_children())
        for segment in dashboard["segments"]:
            segment_tree.insert(
                "",
                "end",
                values=(
                    segment["segment_id"],
                    segment["capacity_bytes"],
                    segment["used_bytes"],
                    segment["remaining_bytes"],
                    segment["entry_count"],
                    segment["segment_version"],
                    segment["proof_ok"],
                    segment["proof_content_digest"],
                ),
            )
            for entry in segment["entries"]:
                entry_tree.insert(
                    "",
                    "end",
                    values=(
                        segment["segment_id"],
                        entry["key"],
                        entry["size_bytes"],
                        entry["digest"],
                        entry["actual_digest"],
                        entry["digest_ok"],
                    ),
                )
        views_text.delete("1.0", "end")
        views_text.insert(
            "end",
            json.dumps(
                {
                    "views": dashboard["views"],
                    "recovery": dashboard["recovery"],
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
        )
        for unknown in dashboard["unknown_entries"]:
            unknown_tree.insert(
                "",
                "end",
                values=(
                    unknown["segment_id"],
                    unknown["entry_key"],
                    unknown["reason"],
                    unknown["expected_digest"],
                    unknown["observed_digest"],
                    unknown["observed_size_bytes"],
                    unknown["operation_id"],
                ),
            )

    controls = ttk.Frame(header)
    controls.pack(side="right")
    ttk.Button(controls, text="Refresh", command=refresh).pack(side="right")
    refresh()
    root.mainloop()
