"""Optional Tkinter viewer; import failure degrades to a clear message.

The experiment bench never requires a display: all decision logic lives in the
CLI/library. This viewer only renders already computed ledger conclusions.
"""

from __future__ import annotations

import json
from pathlib import Path

TKINTER_AVAILABLE = False
IMPORT_ERROR: Exception | None = None
try:
    import tkinter as tk
    from tkinter import ttk

    TKINTER_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on OS python packaging
    tk = None
    ttk = None
    IMPORT_ERROR = exc


def launch_viewer(report_path: str | Path) -> None:
    if not TKINTER_AVAILABLE:
        raise RuntimeError(
            "Tkinter is not available in this Python installation; "
            f"use the CLI instead. Underlying error: {IMPORT_ERROR!r}"
        )
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    root = tk.Tk()
    root.title("DR Decision Bench Viewer")
    frame = ttk.Frame(root, padding=8)
    frame.grid(row=0, column=0, sticky="nsew")
    text = tk.Text(frame, width=110, height=36, wrap="none")
    text.insert(
        "1.0",
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False),
    )
    text.grid(row=0, column=0, sticky="nsew")
    root.mainloop()
