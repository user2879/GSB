"""Tkinter diagnostic window with a strictly read-only refresh path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .dashboard import render_dashboard_text
from .errors import DiagnosticError


def launch_gui(db_path: str | Path) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover - platform-dependent
        raise DiagnosticError(
            "E_UI",
            f"tkinter is unavailable in this Python runtime: {exc}",
            field="tkinter",
        ) from exc

    root = tk.Tk()
    root.title("Declarative Drift Diagnostic - read-only ledger")
    root.geometry("1280x820")

    frame = ttk.Frame(root, padding=8)
    frame.pack(fill=tk.BOTH, expand=True)
    toolbar = ttk.Frame(frame)
    toolbar.pack(fill=tk.X, pady=(0, 6))
    status = tk.StringVar(value="Read-only view")
    ttk.Label(toolbar, textvariable=status).pack(side=tk.LEFT)

    text = tk.Text(frame, wrap=tk.NONE, font=("Consolas", 10))
    text.pack(fill=tk.BOTH, expand=True)
    scroll_x = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=text.xview)
    scroll_y = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
    text.configure(xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)
    scroll_x.place(relx=0, rely=1, relwidth=1, anchor=tk.SW)
    scroll_y.place(relx=1, rely=0, relheight=1, anchor=tk.NE)

    def refresh() -> None:
        try:
            content = render_dashboard_text(db_path)
        except Exception as exc:
            content = f"refresh failed: {exc}\n"
            status.set("Refresh failed; ledger was not modified")
        else:
            status.set("Refreshed from read-only SQLite connection")
        text.configure(state=tk.NORMAL)
        text.delete("1.0", tk.END)
        text.insert("1.0", content)
        text.configure(state=tk.DISABLED)

    refresh_button = ttk.Button(toolbar, text="Refresh", command=refresh)
    refresh_button.pack(side=tk.RIGHT)
    close_button = ttk.Button(toolbar, text="Close", command=root.destroy)
    close_button.pack(side=tk.RIGHT, padx=(0, 6))

    refresh()
    root.mainloop()


def ui_environment() -> dict[str, Any]:
    try:
        import tkinter

        return {"tkinter_available": True, "tk_version": tkinter.TkVersion}
    except Exception as exc:
        return {"tkinter_available": False, "error": str(exc)}
