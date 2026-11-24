"""Tkinter diagnostics window (Tk is imported lazily).

On machines without a Tk runtime, :func:`launch_tk_ui` raises a clear
:class:`RuntimeError` and callers can fall back to the headless view model /
text table. Refreshing only re-reads engine state and never writes the
ledger.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .viewmodel import build_window_rows

if TYPE_CHECKING:  # pragma: no cover
    from ..engine import ObsFlowEngine


def launch_tk_ui(engine: "ObsFlowEngine", title: str = "obsflow diagnostics"
                 ) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Tkinter is not available in this Python runtime; use the "
            "headless diagnostics view model instead") from exc

    root = tk.Tk()
    root.title(title)
    columns = ("metric", "win", "start", "end", "count", "sum_q", "mean_q",
               "val_err<", "sum_err<", "ver", "status", "lease")
    tree = ttk.Treeview(root, columns=columns, show="headings")
    for column in columns:
        tree.heading(column, text=column)
        tree.column(column, width=90, anchor="center")
    tree.pack(fill="both", expand=True)

    info = ttk.Label(root, text="", anchor="w")
    info.pack(fill="x")

    def refresh() -> None:
        for item in tree.get_children():
            tree.delete(item)
        rows = build_window_rows(engine.query_windows(),
                                 engine.lease_manager)
        for row in rows:  # stable render order from the view model
            tree.insert("", "end", values=(
                row.metric_key, row.window_index, row.start_seq, row.end_seq,
                row.count, row.sum_q, row.mean_q, row.value_error_bound_q,
                row.sum_error_bound_q, row.window_version, row.status,
                row.lease_state))
        unknown = sum(1 for row in rows if row.status == "unknown")
        info.config(text=f"windows={len(rows)} unknown={unknown} "
                         f"(refresh is read-only)")
        root.after(1000, refresh)

    refresh()
    root.mainloop()
