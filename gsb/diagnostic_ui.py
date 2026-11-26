from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def tkinter_status() -> dict[str, Any]:
    try:
        import tkinter  # noqa: F401
        from tkinter import ttk  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - optional system package/display
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "install_hint": "Ubuntu/WSL: sudo apt install python3-tk tk8.6 && export DISPLAY=:0",
        }
    return {"available": True, "error": None}


def launch(state_dir: str | Path) -> int:
    status = tkinter_status()
    if not status["available"]:
        print(json.dumps({"ok": False, "ui": "tkinter", **status}, ensure_ascii=False, indent=2))
        return 2

    import tkinter as tk
    from tkinter import ttk

    from .store import SnapshotStore

    state = SnapshotStore(state_dir).load_current()
    if state is None:
        raise FileNotFoundError(f"no committed snapshot under {state_dir}")

    root = tk.Tk()
    root.title("GSB Ledger Diagnostics")
    root.geometry("1100x700")

    top = ttk.Frame(root, padding=8)
    top.pack(fill=tk.X)
    manifest = state["manifest"]
    ttk.Label(
        top,
        text=(
            f"version={manifest['global_version']} status={manifest['status']} "
            f"config={manifest['config_version']} hash={manifest['config_hash'][:12]}"
        ),
    ).pack(side=tk.LEFT)

    body = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
    body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    tree_frame = ttk.Frame(body)
    detail_frame = ttk.Frame(body)
    body.add(tree_frame, weight=1)
    body.add(detail_frame, weight=2)

    columns = ("stage", "subject", "disposition", "cache_key")
    tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="browse")
    for column, width in zip(columns, (130, 220, 110, 220)):
        tree.heading(column, text=column)
        tree.column(column, width=width)
    tree.pack(fill=tk.BOTH, expand=True)

    detail = tk.Text(detail_frame, wrap=tk.NONE)
    detail.pack(fill=tk.BOTH, expand=True)

    entries = state.get("ledger", {}).get("entries", [])
    for entry in entries:
        tree.insert(
            "",
            tk.END,
            iid=entry["entry_id"],
            values=(
                entry["stage"],
                entry["subject_id"],
                entry.get("disposition", ""),
                entry["cache_key"][:24],
            ),
        )

    def show_detail(_event: Any = None) -> None:
        selection = tree.selection()
        if not selection:
            return
        entry_id = selection[0]
        entry = next(item for item in entries if item["entry_id"] == entry_id)
        detail.delete("1.0", tk.END)
        detail.insert(tk.END, json.dumps(entry, ensure_ascii=False, indent=2, sort_keys=True))

    tree.bind("<<TreeviewSelect>>", show_detail)
    if entries:
        tree.selection_set(entries[0]["entry_id"])
        show_detail()
    root.mainloop()
    return 0
