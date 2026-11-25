from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .ownership import adjudicate_events
from .spec import load_declaration, load_events


def run_tk(_args: Optional[Any] = None) -> int:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("LeaseMesh 租约与副本段裁决台")
    root.geometry("900x620")

    controls = ttk.Frame(root, padding=8)
    controls.pack(fill=tk.X)
    output = tk.Text(root, wrap=tk.NONE, font=("DejaVu Sans Mono", 10))
    output.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def show(payload: Any) -> None:
        output.delete("1.0", tk.END)
        output.insert("1.0", json.dumps(payload, ensure_ascii=False, indent=2))

    def adjudicate_sample() -> None:
        root_dir = Path(__file__).resolve().parent.parent
        spec = load_declaration(root_dir / "samples" / "declarations.json")
        events = load_events(root_dir / "samples" / "events.json")
        show(adjudicate_events(events, spec))

    ttk.Label(controls, text="LeaseMesh").pack(side=tk.LEFT)
    ttk.Button(controls, text="重算样本归属", command=adjudicate_sample).pack(side=tk.LEFT, padx=12)
    ttk.Button(controls, text="退出", command=root.destroy).pack(side=tk.RIGHT)
    root.after(50, adjudicate_sample)
    root.mainloop()
    return 0
