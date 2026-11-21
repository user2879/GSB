"""Optional Tk status viewer; the core never depends on tkinter."""

from __future__ import annotations

from pathlib import Path


def show_ledger(ledger_path: Path | str) -> None:
    import tkinter as tk
    from tkinter import ttk

    from .ledger import Ledger
    from .util import canonical_json_dumps

    path = Path(ledger_path)
    with Ledger(path) as ledger:
        report = ledger.current_report()
    root = tk.Tk()
    root.title(f"Replica Repair Ledger - {path.name}")
    root.geometry("980x720")
    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)
    text = tk.Text(frame, wrap="none", font=("Consolas", 10))
    text.pack(fill="both", expand=True)
    text.insert("1.0", canonical_json_dumps(report, pretty=True))
    text.configure(state="disabled")
    root.mainloop()
