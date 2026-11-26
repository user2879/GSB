from __future__ import annotations

import json
from pathlib import Path

from .snapshot import SessionSnapshot


def launch(snapshot_path: str | Path) -> int:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:
        raise RuntimeError(
            "Tkinter is unavailable. On Ubuntu/WSL install python3-tk "
            "(for example: sudo apt install python3-tk), then restart WSL."
        ) from exc

    snapshot = SessionSnapshot.load(snapshot_path).to_dict()
    root = tk.Tk()
    root.title(f"GSB snapshot v{snapshot['capacity_version']}")
    root.geometry("960x640")

    header = ttk.Label(
        root,
        text=(
            f"format={snapshot['snapshot_format']} "
            f"capacity_version={snapshot['capacity_version']} "
            f"digest={snapshot['snapshot_digest'][:16]} "
            f"stale={snapshot['stale']}"
        ),
        padding=8,
    )
    header.pack(anchor="w")

    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True, padx=8, pady=8)
    text = tk.Text(frame, wrap="none", font=("DejaVu Sans Mono", 10))
    yscroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    xscroll = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
    text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    text.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)

    text.insert("1.0", json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True))
    text.configure(state="disabled")
    root.mainloop()
    return 0
