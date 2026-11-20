"""Tkinter diagnostic workbench and a headless, deterministic view builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import EdgeError
from .ledger import Ledger


def build_view_model(db_path: str | Path) -> dict[str, Any]:
    """Read the current ledger view without appending any ledger content."""

    ledger = Ledger(db_path, read_only=True)
    try:
        view = ledger.view_model()
    finally:
        ledger.close()
    view["entries"] = sorted(
        view.get("entries", []),
        key=lambda item: (item["node_id"], item["path"], item["etag"], item["cache_key"]),
    )
    view["evictions"] = sorted(
        view.get("evictions", []),
        key=lambda item: (item["node_id"], item["event_seq"]),
    )
    view["propagation"] = sorted(view.get("propagation", []), key=lambda item: item["seq"])
    view["versions"] = sorted(view.get("versions", []), key=lambda item: item["ledger_version"])
    return view


def run_gui(db_path: str | Path) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:
        raise EdgeError(
            code="E_TK",
            message="tkinter is not available in this Python installation",
            details={"hint": "Reinstall Python 3.11/3.12 with the Tcl/Tk option enabled."},
            cause=type(exc).__name__,
        ) from exc

    try:
        root = tk.Tk()
    except Exception as exc:  # TclError when no display is available
        raise EdgeError(
            code="E_TK",
            message="unable to create the Tk window",
            details={"error": str(exc)},
            cause=type(exc).__name__,
        ) from exc
    root.title("EdgeCache Consistency Workbench")
    root.geometry("1180x760")

    version_text = tk.StringVar(value="")
    status_text = tk.StringVar(value="")
    header = ttk.Frame(root, padding=(10, 8))
    header.pack(fill="x")
    ttk.Label(header, textvariable=version_text, font=("TkDefaultFont", 11, "bold")).pack(side="left")
    ttk.Label(header, textvariable=status_text).pack(side="left", padx=(20, 0))
    ttk.Button(header, text="Refresh", command=lambda: refresh()).pack(side="right")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    entries_frame, entries_tree = _make_tree(
        notebook,
        ["node_id", "path", "etag", "content_version", "remaining_ttl_ms", "entry_version", "stale"],
    )
    evictions_frame, evictions_tree = _make_tree(
        notebook,
        ["node_id", "event_seq", "path", "etag", "reason", "rule"],
    )
    propagation_frame, propagation_tree = _make_tree(
        notebook,
        ["seq", "trigger_path", "affected_path", "depth", "content_version", "policy_version"],
    )
    versions_frame, versions_tree = _make_tree(
        notebook,
        ["ledger_version", "status", "round_kind", "content_version", "policy_version", "stale"],
    )
    notebook.add(entries_frame, text="Cache entries")
    notebook.add(evictions_frame, text="Evictions")
    notebook.add(propagation_frame, text="Invalidation chains")
    notebook.add(versions_frame, text="Ledger versions")

    def refresh() -> None:
        view = build_view_model(db_path)
        ledger_version = view.get("ledger_version")
        version_text.set(
            f"ledger={ledger_version or '-'}  content={view.get('content_version') or '-'}  policy={view.get('policy_version') or '-'}"
        )
        stats = view.get("current_stats", {})
        status_text.set(
            f"current entries={stats.get('entry_count', 0)}  evictions={stats.get('eviction_count', 0)}  "
            f"propagation={stats.get('propagation_count', 0)}  unknown rounds={len(view.get('unknown_rounds', []))}"
        )
        _fill_tree(entries_tree, view.get("entries", []), current_only=False)
        _fill_tree(evictions_tree, view.get("evictions", []))
        _fill_tree(propagation_tree, view.get("propagation", []))
        _fill_tree(versions_tree, view.get("versions", []), current_only=False)

    refresh()
    root.mainloop()


def _make_tree(parent: Any, columns: list[str]):
    from tkinter import ttk

    frame = ttk.Frame(parent)
    tree = ttk.Treeview(frame, columns=columns, show="headings")
    for column in columns:
        tree.heading(column, text=column)
        width = 150 if column in {"path", "cache_key", "content_version"} else 105
        tree.column(column, width=width, stretch=True, anchor="w")
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    return frame, tree


def _fill_tree(tree: Any, rows: list[dict[str, Any]], *, current_only: bool = True) -> None:
    for item in tree.get_children():
        tree.delete(item)
    for row in rows:
        if current_only and row.get("stale"):
            continue
        values = [row.get(column, "") for column in tree["columns"]]
        tree.insert("", "end", values=values)
