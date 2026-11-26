"""Optional Tkinter diagnostic dashboard.

The core acceptance path is headless.  This module is imported lazily so a
missing libtk shared object never prevents validation or worker execution.
"""

from __future__ import annotations

import json

from .errors import DiagnosticError
from .model import SessionSpec


def show_dashboard(session: SessionSpec, *, ledger_path: str | None = None) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover - depends on WSL packages
        raise DiagnosticError(
            "E_UI",
            "Tkinter is unavailable; install it with: sudo apt install python3-tk",
            details={"error": str(exc)},
        ) from exc
    try:
        root = tk.Tk()
    except Exception as exc:  # pragma: no cover - requires a display
        raise DiagnosticError(
            "E_UI",
            "no graphical display is available for Tkinter",
            details={"error": str(exc)},
        ) from exc
    root.title("GSB Training Checkpoint Diagnostics")
    root.geometry("1060x700")
    frame = ttk.Frame(root, padding=12)
    frame.pack(fill=tk.BOTH, expand=True)
    ttk.Label(
        frame,
        text=f"shards={len(session.shards)} checkpoints={len(session.checkpoints)} "
        f"batches={len(session.batches)} events={len(session.events)}",
        font=("TkDefaultFont", 11, "bold"),
    ).pack(anchor=tk.W)
    ttk.Label(
        frame,
        text=f"global_budget={session.global_budget}  source={session.source}  ledger={ledger_path or '-'}",
    ).pack(anchor=tk.W, pady=(4, 10))
    notebook = ttk.Notebook(frame)
    notebook.pack(fill=tk.BOTH, expand=True)

    def add_text_tab(title: str, value: object) -> None:
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=title)
        text = tk.Text(tab, wrap="none", font=("TkFixedFont", 10))
        scroll_y = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=text.yview)
        scroll_x = ttk.Scrollbar(tab, orient=tk.HORIZONTAL, command=text.xview)
        text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        text.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        text.insert("1.0", json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))
        text.configure(state="disabled")

    add_text_tab("Shards", [shard.to_dict() for shard in session.shards])
    add_text_tab("Checkpoints", [cp.to_dict() for cp in session.checkpoints])
    add_text_tab("Batches", [batch.to_dict() for batch in session.batches])
    add_text_tab("Events", [event.to_dict() for event in session.events])
    root.mainloop()

