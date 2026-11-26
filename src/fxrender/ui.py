"""Optional Tk status panel.

Tk is imported lazily so headless runs and tests never require a display.
When no display is available the panel degrades to a printed text summary.
"""

from __future__ import annotations


def show_summary(summary: dict) -> bool:
    try:
        import tkinter as tk  # noqa: F401
    except Exception:
        print("[fxrender] tkinter unavailable; text summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return False
    try:
        import tkinter as tk
        root = tk.Tk()
        root.title("fxrender status")
        root.withdraw()
        text = tk.Text(root, width=72, height=18)
        text.pack()
        for key, value in summary.items():
            text.insert("end", f"{key}: {value}\n")
        root.update()
        root.destroy()
        return True
    except Exception:
        print("[fxrender] no display; text summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return False
