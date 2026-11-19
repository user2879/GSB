"""Optional Tkinter desktop view.

Tkinter is imported lazily so the headless pipeline (and this WSL environment
without python3-tk) keeps working.  On the standard Windows python.org installer
Tkinter is bundled and ``python -m cfgdist.gui`` launches the window.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _import_tk():  # pragma: no cover - GUI-only
    import tkinter as tk
    from tkinter import ttk

    return tk, ttk


def launch(snapshot_path: str | Path) -> None:  # pragma: no cover - GUI-only
    tk, ttk = _import_tk()
    payload = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))

    root = tk.Tk()
    root.title(f"cfgdist session {payload['session_id']} - v{payload['version']}")
    columns = ("key", "version", "derived_summary", "verdict", "stale", "source_version")
    tree = ttk.Treeview(root, columns=columns, show="headings")
    for column in columns:
        tree.heading(column, text=column)
        tree.column(column, width=150, anchor="center")
    for row in payload["view"]["rows"]:
        tree.insert(
            "",
            "end",
            values=(
                row["key"],
                row["version"],
                row["derived_summary"],
                row["verdict"],
                row["stale"],
                row["source_version"],
            ),
        )
    tree.pack(expand=True, fill="both")

    status = tk.Label(
        root,
        text=f"converged={payload['convergence']['converged']}  "
        f"digest={payload['view']['view_digest'][:16]}...",
    )
    status.pack(side="bottom", fill="x")
    root.mainloop()


def render_text(snapshot_path: str | Path) -> str:
    """Headless rendering of the same rows the GUI would display."""
    payload: dict[str, Any] = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    lines = [
        f"session={payload['session_id']} version={payload['version']} "
        f"converged={payload['convergence']['converged']}"
    ]
    for row in payload["view"]["rows"]:
        lines.append(
            f"  {row['key']:<12} {row['derived_summary']:<16} "
            f"{row['verdict']:<10} stale={row['stale']} src=v{row['source_version']}"
        )
    return "\n".join(lines)
